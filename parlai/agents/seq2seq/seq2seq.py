#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.torch_generator_agent import TorchGeneratorAgent
from .modules import Seq2seq, opt_to_kwargs

import torch
import torch.nn as nn

import json


class Seq2seqAgent(TorchGeneratorAgent):
    """Agent which takes an input sequence and produces an output sequence.

    This model supports encoding the input and decoding the output via one of
    several flavors of RNN. It then uses a linear layer (whose weights can
    be shared with the embedding layer) to convert RNN output states into
    output tokens. This model supports greedy decoding, selecting the
    highest probability token at each time step, as well as beam
    search.

    For more information, see the following papers:
    - Neural Machine Translation by Jointly Learning to Align and Translate
      `(Bahdanau et al. 2014) <arxiv.org/abs/1409.0473>`_
    - Sequence to Sequence Learning with Neural Networks
      `(Sutskever et al. 2014) <arxiv.org/abs/1409.3215>`_
    - Effective Approaches to Attention-based Neural Machine Translation
      `(Luong et al. 2015) <arxiv.org/abs/1508.04025>`_
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Seq2Seq Arguments')
        agent.add_argument('-hs', '--hiddensize', type=int, default=128,
                           help='size of the hidden layers')
        agent.add_argument('-esz', '--embeddingsize', type=int, default=128,
                           help='size of the token embeddings')
        agent.add_argument('-nl', '--numlayers', type=int, default=2,
                           help='number of hidden layers')
        agent.add_argument('-dr', '--dropout', type=float, default=0.1,
                           help='dropout rate')
        agent.add_argument('-bi', '--bidirectional', type='bool',
                           default=False,
                           help='whether to encode the context with a '
                                'bidirectional rnn')
        agent.add_argument('-att', '--attention', default='none',
                           choices=['none', 'concat', 'general', 'dot',
                                    'local'],
                           help='Choices: none, concat, general, local. '
                                'If set local, also set attention-length. '
                                '(see arxiv.org/abs/1508.04025)')
        agent.add_argument('-attl', '--attention-length', default=48, type=int,
                           help='Length of local attention.')
        agent.add_argument('--attention-time', default='post',
                           choices=['pre', 'post'],
                           help='Whether to apply attention before or after '
                                'decoding.')
        agent.add_argument('-rnn', '--rnn-class', default='lstm',
                           choices=Seq2seq.RNN_OPTS.keys(),
                           help='Choose between different types of RNNs.')
        agent.add_argument('-dec', '--decoder', default='same',
                           choices=['same', 'shared'],
                           help='Choose between different decoder modules. '
                                'Default "same" uses same class as encoder, '
                                'while "shared" also uses the same weights. '
                                'Note that shared disabled some encoder '
                                'options--in particular, bidirectionality.')
        agent.add_argument('-lt', '--lookuptable', default='unique',
                           choices=['unique', 'enc_dec', 'dec_out', 'all'],
                           help='The encoder, decoder, and output modules can '
                                'share weights, or not. '
                                'Unique has independent embeddings for each. '
                                'Enc_dec shares the embedding for the encoder '
                                'and decoder. '
                                'Dec_out shares decoder embedding and output '
                                'weights. '
                                'All shares all three weights.')
        agent.add_argument('-soft', '--numsoftmax', default=1, type=int,
                           help='default 1, if greater then uses mixture of '
                                'softmax (see arxiv.org/abs/1711.03953).')
        agent.add_argument('-idr', '--input-dropout', type=float, default=0.0,
                           help='Probability of replacing tokens with UNK in training.')

        
        agent.add_argument('-ord', '--order', default='no',
                           choices=['no', '1_order', '2_order', '3_order','full'],
                           help='Choices: no_order, 1_order, 2_order, 3_order,full_order.')
        agent.add_argument('-dli_in_dim','--dli_input_dim',default=128,type=int, help='size of the dli input dim')
        agent.add_argument('-dli_rnn_hid','--dli_rnn_hiddensize',default=64,type=int, help='size of the dli rnn hidden dim')
        agent.add_argument('-dli_ffn_dim','--dli_ffn_dimension',default=128,type=int, help='size of the dli ffn dim')

        super(Seq2seqAgent, cls).add_cmdline_args(argparser)
        return agent

    @staticmethod
    def model_version():
        """Return current version of this model, counting up from 0.

        Models may not be backwards-compatible with older versions.
        Version 1 split from version 0 on Aug 29, 2018.
        Version 2 split from version 1 on Nov 13, 2018
        To use version 0, use --model legacy:seq2seq:0
        To use version 1, use --model legacy:seq2seq:1
        (legacy agent code is located in parlai/agents/legacy_agents).
        """
        return 2

    def __init__(self, opt, shared=None):
        """Set up model."""
        super().__init__(opt, shared)
        self.id = 'Seq2Seq'

    def build_model(self, states=None):
        """Initialize model, override to change model setup."""
        opt = self.opt
        if not states:
            states = {}

        kwargs = opt_to_kwargs(opt)
        self.model = Seq2seq(
            opt['order'],len(self.dict), opt['embeddingsize'], opt['hiddensize'],
            padding_idx=self.NULL_IDX, start_idx=self.START_IDX,
            end_idx=self.END_IDX, unknown_idx=self.dict[self.dict.unk_token],
            longest_label=states.get('longest_label', 1),
            **kwargs)

        if (opt.get('dict_tokenizer') == 'bpe' and
                opt['embedding_type'] != 'random'):
            print('skipping preinitialization of embeddings for bpe')
        elif not states and opt['embedding_type'] != 'random':
            # `not states`: only set up embeddings if not loading model
            self._copy_embeddings(self.model.decoder.lt.weight,
                                  opt['embedding_type'])
            if opt['lookuptable'] in ['unique', 'dec_out']:
                # also set encoder lt, since it's not shared
                self._copy_embeddings(self.model.encoder.lt.weight,
                                      opt['embedding_type'], log=False)

        if states:
            # set loaded states if applicable
            self.model.load_state_dict(states['model'])

        if self.use_cuda:
            self.model.cuda()

        if opt['embedding_type'].endswith('fixed'):
            print('Seq2seq: fixing embedding weights.')
            self.model.decoder.lt.weight.requires_grad = False
            self.model.encoder.lt.weight.requires_grad = False
            if opt['lookuptable'] in ['dec_out', 'all']:
                self.model.decoder.e2s.weight.requires_grad = False

        if self.use_cuda:
            self.model.cuda()

        return self.model

    def build_criterion(self):
        # set up criteria
        if self.opt.get('numsoftmax', 1) > 1:
            self.criterion = nn.NLLLoss(
                ignore_index=self.NULL_IDX, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.NULL_IDX, size_average=False)

        if self.use_cuda:
            self.criterion.cuda()

    def vectorize(self, *args, **kwargs):
        """Override vectorize for seq2seq."""
        kwargs['add_start'] = False  # model does this in module code
        kwargs['add_end'] = True  # we do want this
        return super().vectorize(*args, **kwargs)

    def batchify(self, *args, **kwargs):
        """Override batchify options for seq2seq."""
        kwargs['sort'] = True  # need sorted for pack_padded
        return super().batchify(*args, **kwargs)

    def save(self, path=None):
        """Save model parameters if model_file is set."""
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'model'):
            model = {}
            if hasattr(self.model, 'module'):
                model['model'] = self.model.module.state_dict()
                model['longest_label'] = self.model.module.longest_label
            else:
                model['model'] = self.model.state_dict()
                model['longest_label'] = self.model.longest_label
            model['optimizer'] = self.optimizer.state_dict()
            model['optimizer_type'] = self.opt['optimizer']

            with open(path, 'wb') as write:
                torch.save(model, write)

            # save opt file
            with open(path + '.opt', 'w') as handle:
                # save version string
                self.opt['model_version'] = self.model_version()
                json.dump(self.opt, handle)

    def load(self, path):
        """Return opt and model states."""
        states = torch.load(path, map_location=lambda cpu, _: cpu)
        # set loaded states if applicable
        self.model.load_state_dict(states['model'])
        if 'longest_label' in states:
            self.model.longest_label = states['longest_label']
        return states
