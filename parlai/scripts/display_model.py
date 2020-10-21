#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Basic example which iterates through the tasks specified and runs the given
model on them.

Examples
--------

.. code-block:: shell

  python examples/display_model.py -t babi:task1k:1 -m "repeat_label"
  python examples/display_model.py -t "#MovieDD-Reddit" -m "ir_baseline" -mp "-lp 0.5" -dt test
"""  # noqa: E501

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task

import random


def setup_args():
    parser = ParlaiParser(True, True, 'Display model predictions.')
    parser.add_argument('-n', '--num-examples', default=10)
    parser.add_argument('--display-ignore-fields', type=str, default='his_turn_end_ids,eval_labels_choice,label_candidates')
    # by default we want to display info about the validation set
    parser.set_defaults(datatype='valid')
    return parser


def display_model(opt):
    random.seed(42)

    sample_ids = [i for i in range(5000)]
    random.shuffle(sample_ids)
    ids = [sample_ids[id] for id in range(1,51)]
    
    # Create model and assign it to the specified task
    agent = create_agent(opt)
    world = create_task(opt, agent)
    cnt = 0
    # Show some example dialogs.
    while not world.epoch_done():
        cnt += opt.get('batchsize', 1)
        world.parley()
        if cnt in ids:
            print(world.display() + "\n~~~~~~~~~~~")


if __name__ == '__main__':
    # Get command line arguments
    parser = setup_args()
    opt = parser.parse_args(print_args=False)
    display_model(opt)
