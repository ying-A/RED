# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.agents import Agent
from parlai.core.utils import warn_once
from parlai.core.utils import padded_3d
from parlai.core.torch_generator_agent import TorchGeneratorAgent

from .modules import RecosaGeneratorModel

import torch


warn_once(
    "Public release transformer models are currently in beta. The name of "
    "command line options may change or disappear before a stable release. We "
    "welcome your feedback. Please file feedback as issues at "
    "https://github.com/facebookresearch/ParlAI/issues/new"
)


def add_common_cmdline_args(argparser):
    argparser.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
    argparser.add_argument('-nl', '--n-layers', type=int, default=2)
    argparser.add_argument('-hid', '--ffn-size', type=int, default=300,
                           help='Hidden size of the FFN layers')
    argparser.add_argument('--attention-dropout', type=float, default=0.0)
    argparser.add_argument('--relu-dropout', type=float, default=0.0)
    argparser.add_argument('--n-heads', type=int, default=2,
                           help='Number of multihead attention heads')
    argparser.add_argument('--learn-positional-embeddings', type='bool', default=False)
    argparser.add_argument('--embeddings-scale', type='bool', default=True)
    argparser.add_argument('--n-positions', type=int, default=None, hidden=True,
                           help='Number of positional embeddings to learn. Defaults '
                                'to truncate or 1024 if not provided.')

class Recosa(Agent):
    """
    Placeholder class, which just throws an error telling the user to specify
    whether they want the ranker or the generator.
    """
    def __init__(self, opt, shared=None):
        raise RuntimeError(
            "`--model recosa` is not a valid choice. Please select "
            " `--model recosa/generator' "
        )


class RecosaGeneratorAgent(TorchGeneratorAgent):
    @classmethod
    def add_cmdline_args(cls, argparser):
        """Add command-line arguments specifically for this agent."""
        agent = argparser.add_argument_group('Transformer Arguments')
        
        agent.add_argument('-ord', '--order', default='no',
                           choices=['no', '1_order', '2_order', '3_order','full'],
                           help='Choices: no_order, 1_order, 2_order, 3_order,full_order.')
        agent.add_argument('-dli_in_dim','--dli_input_dim',default=300,type=int, help='size of the dli input dim')
        agent.add_argument('-dli_rnn_hid','--dli_rnn_hiddensize',default=64,type=int, help='size of the dli rnn hidden dim')
        agent.add_argument('-dli_ffn_dim','--dli_ffn_dimension',default=128,type=int, help='size of the dli ffn dim')

        agent.add_argument('-rnn_hid','--rnn_hiddensize',default=300,type=int, help='size of the rnn input embedding')
        agent.add_argument('-rnn_esz','--rnn_embeddingsize',default=300,type=int, help='size of the rnn hidden layers')
        agent.add_argument('-rnn_nlayers','--rnn_numlayers',default=2,type=int, help='the number of rnn hidden layers')
        agent.add_argument('-rnn_cls','--rnn_class',default='gru',choices=['lstm','gru','rnn'], help='rnn class for utterance encoder')
        agent.add_argument('-rnn_bi','--rnn_bidirectional',default=False,type=bool, help='whether use bi-dir rnn')
        agent.add_argument('--rnn_dropout',default=0.0,type=float, help='dropout for rnn hidden layers')
        agent.add_argument('--input_dropout',default=0.0,type=float, help='input dropout for inputs')
        agent.add_argument('--max_turns',default=30,type=int, help='the max number of history turns')
        agent.add_argument('--max_single_seq_len',default=50,type=int, help='the max length of single history utterance')

        add_common_cmdline_args(agent)
      
        cls.dictionary_class().add_cmdline_args(argparser)
        
        super(RecosaGeneratorAgent, cls).add_cmdline_args(argparser)
        return agent

    def build_model(self, states=None):
        self.model = RecosaGeneratorModel(self.opt, self.dict)
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                self.model.encoder.embeddings.weight, self.opt['embedding_type']
            )
        if self.use_cuda:
            self.model.cuda()
        return self.model 
