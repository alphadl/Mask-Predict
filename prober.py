#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a MT model prober to probe NMT decoder layers.
"""

import collections
import torch
import subprocess
import itertools
import math
import os
import random
import time
import pickle

from argparse import Namespace
from itertools import chain
from torch import nn
import torch.nn.functional as F

from fairseq import distributed_utils, bleu, optim, models, options, progress_bar, tasks, utils, bleu
from fairseq.data import iterators, Dictionary, IndexedCachedDataset
from fairseq.utils import convert_padding_direction, get_perplexity
from fairseq.meters import AverageMeter, StopwatchMeter, TimeMeter
from fairseq.optim import lr_scheduler
from fairseq.sequence_generator import SequenceGenerator
from torch.nn.utils.rnn import pad_sequence
from fairseq.models.bert_seq2seq import SelfTransformerDecoder as TransformerDecoder
from fairseq.data.data_utils import collate_tokens

def parse():
    parser = options.get_training_parser()
    parser.add_argument('--nmt_model', type=str)
    parser.add_argument('--prob_layer', type=int, default=6)
    parser.add_argument('--prob_self_attn', action='store_true')
    parser.add_argument('--prob_ed_attn', action='store_true')
    parser.add_argument('--prob_ed_norm', action='store_true')
    parser.add_argument('--prob_ffn', action='store_true')
    parser.add_argument('--eval_model_path', type=str, default=None)
    parser.add_argument('--prob', type=str, default='src')
    parser.add_argument('--accum_grad', type=int, default=1)
    return options.parse_args_and_arch(parser)

def main(args, init_distributed=False):
    print(args)
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    #define new task
    task = tasks.setup_task(args)
    #cross entropy loss
    criterion = task.build_criterion(args)
    load_dataset_splits(args, task)

    # Load nmt_model
    print('| loading model(s) from {}'.format(args.nmt_model))
    models, nmt_args = utils.load_ensemble_for_inference(args.nmt_model.split(':'), task)
    nmt_model = models[0]
    nmt_model.eval()
    nmt_model.make_generation_fast_(raise_exception=False)
    if torch.cuda.is_available() and not args.cpu:
        nmt_model.cuda()
    for p in nmt_model.parameters():
        p.requires_grad = False

    # load prob_dict if any
    if args.prob == 'src': # probing for source tokens
        prob_decoder_dict = task.src_dict
        prob_embed_tokens = nmt_model.encoder.embed_tokens
    elif args.prob == 'tgt': # probing for target tokens
        prob_decoder_dict = task.tgt_dict
        prob_embed_tokens = nmt_model.decoder.embed_tokens

    # build probing model
    prob_model = decoder_prob(args, prob_decoder_dict, nmt_model, prob_embed_tokens)

    print("="*20)
    print(prob_model)
    print("="*20)
    print('| criterion {}'.format(criterion.__class__.__name__))
    print('| num. model params:{} (num. trained: {})'.format(
        sum(p.numel() for p in prob_model.parameters()),
        sum(p.numel() for p in prob_model.parameters() if p.requires_grad),
    ))

    # load prob_model if eval_model_path
    if args.eval_model_path is not None:
        eval_state_dict = torch.load(args.eval_model_path, torch.device('cpu'))
        prob_model.load_state_dict(eval_state_dict)

    if torch.cuda.is_available() and not args.cpu:
        prob_model.cuda()
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(args.max_tokens, args.max_sentences))

    # Initialize train_subset dataloader
    if args.eval_model_path is None:
        epoch_itr = task.get_batch_iterator(
            dataset=task.dataset(args.train_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=(256,256),
            ignore_invalid_inputs=True,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        )

    # Initialize distributed training (after data loading)
    if init_distributed:
        import socket
        args.distributed_rank = distributed_utils.distributed_init(args)
        print('| initialized host {} as rank {}'.format(socket.gethostname(), args.distributed_rank))
    else:
        distributed_utils.suppress_output(args.distributed_rank==0)

    # build optim & lr_sche
    params = list(filter(lambda p: p.requires_grad, prob_model.parameters()))
    prob_optim = optim.build_optimizer(args, params)
    # prob_lr_sche = lr_scheduler.build_lr_scheduler(args, prob_optim)

    if args.eval_model_path is None:
        train_probing_model(epoch_itr, args, task, prob_model, prob_optim, criterion)
    else:
        eval_loss = eval_probing_model(args, task, criterion, prob_model)
        print('eval_loss: %s' % eval_loss)


def train_probing_model(epoch_itr, args, task, prob_model, prob_optim, criterion):
    prob_model.train()
    num_updates = 0
    train_loss = []

    while True:  # each epoch
        train_itr = epoch_itr.next_epoch_itr(shuffle=True)
        progress = progress_bar.build_progress_bar(args, train_itr, epoch_itr.epoch, default='tqdm')

        for i,sample in enumerate(progress):
            # build probing targets for loss calculation
            if args.prob == 'src': # source information reconstruction
                sample['target'] = sample['net_input']['src_tokens']
            elif args.prob == 'tgt': # target information reconstruction
                pass

            sample = utils.move_to_cuda(sample)
            sample['net_input']['prev_output_tokens'] = (sample['net_input']['prev_output_tokens'],
                                                        sample['target'])
            loss, sample_size, logging_output = criterion(prob_model, sample)
            loss = loss / sample_size
            (loss/args.accum_grad).backward()
            if (i+1) % args.accum_grad== 0:
                prob_optim.step()
                prob_optim.zero_grad()
                num_updates += 1
            train_loss.append(loss.item())

            if num_updates % 907 == 0:
                print('train_loss at step %d: %f' % (num_updates, sum(train_loss)/len(train_loss)))
                train_loss = []
                print('valid nll_loss at step %d: %s' % (num_updates,
                            eval_probing_model(args, task, criterion, prob_model)))

            if num_updates >= args.max_update:
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                torch.save(prob_model.state_dict(), args.save_dir+'/checkpoint_l%d.pt' % (args.prob_layer))
                return


@torch.no_grad()
def eval_probing_model(args, task, criterion, prob_model):
    prob_model.eval()
    criterion.eval()
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.valid_subset),
        max_tokens=args.max_tokens,
        max_positions=(1024, 1024),
        ignore_invalid_inputs=True,
    ).next_epoch_itr(shuffle=True)
    valid_loss, sample_num = 0, 0

    for i,sample in enumerate(itr):
        # build probing targets for loss calculation
        if args.prob == 'src':
            sample['target'] = sample['net_input']['src_tokens']
        elif args.prob == 'tgt':
            pass

        sample = utils.move_to_cuda(sample)
        sample['net_input']['prev_output_tokens'] = (sample['net_input']['prev_output_tokens'],
                                                    sample['target'])
        loss, sample_size, logging_output = criterion(prob_model, sample)
        valid_loss += logging_output['nll_loss']
        sample_num += sample_size

        if i >= 100: # max eval step for eval on training set
            break

    prob_model.train()
    criterion.train()
    return valid_loss/sample_num


class decoder_prob(nn.Module):
    def __init__(self, args, tgt_dict, nmt_model, embed_tokens):
        super().__init__()
        self.args = args
        self.nmt_model = nmt_model
        self.prober = TransformerDecoder(args, tgt_dict, embed_tokens)

        # monkey-patch
        self.get_normalized_probs = nmt_model.get_normalized_probs
        self.get_targets = nmt_model.get_targets

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        all_sub_layers = self.args.prob_self_attn or self.args.prob_ed_attn or self.args.prob_ed_norm or self.args.prob_ffn
        prev_output_tokens, target = prev_output_tokens
        self.nmt_model.eval()
        with torch.no_grad():
            encoder_out = self.nmt_model.encoder(src_tokens, src_lengths)
            decoder_out = self.nmt_model.decoder(prev_output_tokens, encoder_out, all_sub_layers=all_sub_layers)

        # prob sub_layers
        if self.args.prob_self_attn:
            nmt_out = decoder_out[1]['self_attn'][self.args.prob_layer-1]
        elif self.args.prob_ed_attn:
            nmt_out = decoder_out[1]['ed_attn'][self.args.prob_layer-1]
        elif self.args.prob_ed_norm:
            nmt_out = decoder_out[1]['ed_norm'][self.args.prob_layer-1]
        elif self.args.prob_ffn:
            nmt_out = decoder_out[1]['ffn'][self.args.prob_layer-1]
        else:
            nmt_out = decoder_out[1]['inner_states'][self.args.prob_layer]

        encoder_padding_mask = prev_output_tokens.eq(1)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        prob_input = {'encoder_out': nmt_out,
                      'encoder_padding_mask': encoder_padding_mask}
        prob_prev_output_tokens = collate_tokens([s[s.ne(1)] for s in target],1, 2, False, True)
        return self.prober(prob_prev_output_tokens, prob_input)


def load_dataset_splits(args, task):
    task.load_dataset(args.train_subset, combine=True)
    for split in args.valid_subset.split(','):
        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            try:
                task.load_dataset(split_k, combine=False)
            except FileNotFoundError as e:
                if k > 0:
                    break
                raise e

def distributed_main(i, args):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = i
    main(args, init_distributed=True)


def cli_main():
    args = parse()

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()