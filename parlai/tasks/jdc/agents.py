#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
JD dialog Challenge
"""

import os
import json
from parlai.core.teachers import FixedDialogTeacher
from .build import build


START_ENTRY =  '_SILENCE__'


class DefaultTeacher(FixedDialogTeacher):
    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        if shared:
            self.data = shared['data']
        else:
            build(opt)
            fold = opt.get('datatype', 'train').split(':')[0]
            self._setup_data(fold)

        self.num_exs = sum(len(d) for d in self.data)

        # we learn from both sides of every conversation
        self.num_eps = 2 * len(self.data)
        self.reset()

    def num_episodes(self):
        return self.num_eps

    def num_examples(self):
        return self.num_exs

    def _setup_data(self, fold):
        self.data = []
        fpath = os.path.join(self.opt['datapath'], 'jdc', fold + '.json')
        with open(fpath) as f:
            self.data = json.load(f)

    def get(self, episode_idx, entry_idx=0):
        # Sometimes we're speaker 1 and sometimes we're speaker 2
        speaker_id = episode_idx % 2
        full_eps = self.data[episode_idx // 2]

        entries = [START_ENTRY] + full_eps
        their_turn = entries[speaker_id + 2 * entry_idx]
        my_turn = entries[1 + speaker_id + 2 * entry_idx]

        episode_done = 2 * entry_idx + speaker_id + 1 >= len(full_eps) - 1

        action = {
            'text': their_turn,
            'labels': [my_turn],
            'episode_done': episode_done,
        }
        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared
