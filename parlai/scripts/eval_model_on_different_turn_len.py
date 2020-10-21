#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.logs import TensorboardLogger
from parlai.core.worlds import create_task
from parlai.core.utils import TimeLogger
from nltk.translate import bleu_score as nltkbleu
from collections import defaultdict
import random
import math
import re


re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s)))) 

def compute_bleu(guess, answers):
    """Compute approximate BLEU score between guess and a set of answers."""
    if nltkbleu is None:
        raise ImportError
    # Warning: BLEU calculation *should* include proper tokenization and
    # punctuation etc. We're using the normalize_answer for everything though,
    # so we're over-estimating our BLEU scores.  Also note that NLTK's bleu is
    # going to be slower than fairseq's (which is written in C), but fairseq's
    # requires that everything be in arrays of ints (i.e. as tensors). NLTK's
    # works with strings, which is better suited for this module.
    return nltkbleu.sentence_bleu(
        [normalize_answer(a).split(" ") for a in answers],
        normalize_answer(guess).split(" "),
        smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1,
    )


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Evaluate a model')
    # Get command line arguments
    parser.add_argument('-ne', '--num-examples', type=int, default=-1)
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.add_argument('-ltim', '--log-every-n-secs', type=float, default=2)
    parser.add_argument('--metrics', type=str, default="all",
                        help="list of metrics to show/compute, e.g. "
                             "ppl,f1,accuracy,hits@1."
                             "If 'all' is specified [default] all are shown.")
    parser.add_argument('-pb', '--perturb', type=str, default="None")
    parser.add_argument('-sft', '--skip_first_turn', type='bool', default=False) 
    TensorboardLogger.add_cmdline_args(parser)
    parser.set_defaults(datatype='valid')
    return parser


def load_eval_world(agent, opt, datatype):
    if 'stream' in opt['datatype']:
        datatype += ':stream'
    opt = opt.copy()
    opt['datatype'] = datatype
    if opt.get('pytorch_teacher_task'):
        # never use pytorch teachers for evaluation
        # but don't forget what we were normally using
        opt['task'] = opt['pytorch_teacher_task']
        del opt['pytorch_teacher_task']
    if opt.get('evaltask'):
        # if a different eval task is specified, use it.
        opt['task'] = opt['evaltask']
    if opt.get('eval_batchsize'):
        # override eval time batchsize
        opt['batchsize'] = opt['eval_batchsize']
    if opt.get('validation_share_agent', False):
        valid_agent = create_agent_from_shared(agent.share())
    else:
        valid_agent = agent

    valid_world = create_task(opt, valid_agent)
    return valid_world


def run_eval(valid_world, opt, datatype, max_exs=-1):
    """
    Eval on validation/test data.

    :param valid_world: the pre-created validation world.
    :param opt: the options that specific the task, eval_task, etc
    :param datatype: the datatype to use, such as "valid" or "test"
    :param bool write_log: specifies to write metrics to file if the model_file is set
    :param int max_exs: limits the number of examples if max_exs > 0
    """
    if valid_world is None:
        # This isn't the primary worker, so we can just skip evaluation
        return None
    turns_dict = defaultdict(int)
    word_statistics_1 = {
        # 'his_list': [],
        # 'pred_list': [],
        # 'gt_list':[],
        # 'processed_pred_list':[],
        'unique_1_words':set(),
        'unique_2_words':set(),
        'all_1_cnts':0.0,
        'unique_1_cnts':0.0,
        'all_2_cnts':0.0,
        'unique_2_cnts':0.0,
        'ex_nums':0.0,
        'ex_ppl_loss':0.0,
        'ex_num_tgt_toks':0.0,
        'ex_bleu':0.0
    }
    word_statistics_2 = {
        # 'his_list': [],
        # 'pred_list': [],
        # 'gt_list':[],
        # 'processed_pred_list':[],
        'unique_1_words':set(),
        'unique_2_words':set(),
        'all_1_cnts':0.0,
        'unique_1_cnts':0.0,
        'all_2_cnts':0.0,
        'unique_2_cnts':0.0,
        'ex_nums':0.0,
        'ex_ppl_loss':0.0,
        'ex_num_tgt_toks':0.0,
        'ex_bleu':0.0
    }
    word_statistics_3 = {
        # 'his_list': [],
        # 'pred_list': [],
        # 'gt_list':[],
        # 'processed_pred_list':[],
        'unique_1_words':set(),
        'unique_2_words':set(),
        'all_1_cnts':0.0,
        'unique_1_cnts':0.0,
        'all_2_cnts':0.0,
        'unique_2_cnts':0.0,
        'ex_nums':0.0,
        'ex_ppl_loss':0.0,
        'ex_num_tgt_toks':0.0,
        'ex_bleu':0.0
    }

    def process_metrics(ground_truths,prediction,ex_ppl_loss,ex_num_tgt_toks,word_statistics):
        processed_pred = normalize_answer(prediction)
        all_1_in_pred = processed_pred.split(" ")
        for word in all_1_in_pred:
            if word in word_statistics['unique_1_words']:
                pass
            elif word not in word_statistics['unique_1_words']:
                word_statistics['unique_1_words'].add(word)
                word_statistics['unique_1_cnts']+=1
            word_statistics['all_1_cnts']+=1
        
        if len(all_1_in_pred)>=2:
            left_all_2_in_pred = all_1_in_pred[:-1]
            right_all_2_in_pred = all_1_in_pred[1:]
            assert len(left_all_2_in_pred)==len(right_all_2_in_pred)
            all_2_in_pred = [a+" "+b for (a,b) in zip(left_all_2_in_pred,right_all_2_in_pred)]
            for bi_word in all_2_in_pred:
                if bi_word in word_statistics['unique_2_words']:
                    pass
                elif bi_word not in word_statistics['unique_2_words']:
                    word_statistics['unique_2_words'].add(bi_word)
                    word_statistics['unique_2_cnts']+=1
                word_statistics['all_2_cnts']+=1
        
        # word_statistics['processed_pred_list'].append(processed_pred)
        word_statistics['ex_nums']+=1
        word_statistics['ex_ppl_loss']+= ex_ppl_loss
        word_statistics['ex_num_tgt_toks']+= ex_num_tgt_toks
        word_statistics['ex_bleu']+=compute_bleu(prediction,ground_truths)


        return word_statistics
    
    print('[ running eval: ' + datatype + ' ]')
    valid_world.reset()
    cnt = 0

    while not valid_world.epoch_done():
        valid_world.parley()
        ex_his_num_turns = valid_world.get_agents()[-1].metrics['this_example_his_turns_num']
        ex_ppl_loss = valid_world.get_agents()[-1].metrics['this_example_ppl_loss']
        ex_num_tgt_toks = valid_world.get_agents()[-1].metrics['this_example_num_tokens']
        turns_dict[str(ex_his_num_turns)]+=1
        
        if opt['batchsize'] == 1:
            prediction = valid_world.acts[-1]['text']
            if 'eval_labels' in valid_world.acts[0]:
                ground_truths = valid_world.acts[0]['eval_labels']
            elif 'eval_labels' not in valid_world.acts[0] and 'labels' in valid_world.acts[0]:
                ground_truths = valid_world.acts[0]['labels']
            elif 'eval_labels' not in valid_world.acts[0] and 'labels' not in valid_world.acts[0]:
                print(valid_world.acts[0])
                ground_truths = None
            # word_statistics['his_list'].append(valid_world.acts[0]['text'])
            # word_statistics['gt_list'].append(valid_world.acts[0]['eval_labels'][0])
            # word_statistics['pred_list'].append(prediction)
            if ground_truths is not None:
                if ex_his_num_turns<=10:
                    word_statistics_1 = process_metrics(ground_truths,prediction,ex_ppl_loss,ex_num_tgt_toks,word_statistics_1)
                elif ex_his_num_turns>10 and ex_his_num_turns<=15:
                    word_statistics_2 = process_metrics(ground_truths,prediction,ex_ppl_loss,ex_num_tgt_toks,word_statistics_2)
                else:
                    word_statistics_3 = process_metrics(ground_truths,prediction,ex_ppl_loss,ex_num_tgt_toks,word_statistics_3)

        cnt += valid_world.opt['batchsize']
        if max_exs > 0 and cnt > max_exs + opt.get('numthreads', 1):
            # note this max_exs is approximate--some batches won't always be
            # full depending on the structure of the data
            break
    valid_report = valid_world.report()
    valid_world.reset()  # this makes sure agent doesn't remember valid data

    metrics = datatype + ':' + str(valid_report)
    print(metrics)
    print("turns_distribution")
    print(turns_dict)
    print("\n###### exs_nums: 6-10,11-15,16- ########")
    print(word_statistics_1['ex_nums'],word_statistics_2['ex_nums'],word_statistics_3['ex_nums'])
    print("\n###### ppl #########")
    print("6-10")
    print("ppl: ",math.exp(word_statistics_1['ex_ppl_loss']/word_statistics_1['ex_num_tgt_toks']))
    print("11-15")
    print("ppl: ",math.exp(word_statistics_2['ex_ppl_loss']/word_statistics_2['ex_num_tgt_toks']))
    print("16-")
    print("ppl: ",math.exp(word_statistics_3['ex_ppl_loss']/word_statistics_3['ex_num_tgt_toks']))

    print("\n###### bleu ########")
    print("5-10")
    print("bleu: ",word_statistics_1['ex_bleu']/word_statistics_1['ex_nums'])
    print("11-15")
    print("bleu: ",word_statistics_2['ex_bleu']/word_statistics_2['ex_nums'])
    print("16-")
    print("bleu: ",word_statistics_3['ex_bleu']/word_statistics_3['ex_nums'])


    print("\n###### distinct #########")
    print("5-10")
    print("distinct_1: ",word_statistics_1['unique_1_cnts']/word_statistics_1['all_1_cnts'])
    if word_statistics_1['all_2_cnts']!=0:
        print("distinct_2: ",word_statistics_1['unique_2_cnts']/word_statistics_1['all_2_cnts'])
    print("11-15")
    print("distinct_1: ",word_statistics_2['unique_1_cnts']/word_statistics_2['all_1_cnts'])
    if word_statistics_2['all_2_cnts']!=0:
        print("distinct_2: ",word_statistics_2['unique_2_cnts']/word_statistics_2['all_2_cnts'])
    print("16-")
    print("distinct_1: ",word_statistics_3['unique_1_cnts']/word_statistics_3['all_1_cnts'])
    if word_statistics_3['all_2_cnts']!=0:
        print("distinct_2: ",word_statistics_3['unique_2_cnts']/word_statistics_3['all_2_cnts'])


    return valid_report

def evalmodel_test(opt, printargs=None, print_parser=None):
    """Evaluates model on test dataset.

    :param opt: tells the evaluation function how to run
    :param bool print_parser: if provided, prints the options that are set within the
        model after loading the model
    :return: the final result of calling report()
    """
    if printargs is not None:
        print('[ Deprecated Warning: eval_model no longer uses `printargs` ]')
        print_parser = printargs
    if print_parser is not None:
        if print_parser is True and isinstance(opt, ParlaiParser):
            print_parser = opt
        elif print_parser is False:
            print_parser = None
    if isinstance(opt, ParlaiParser):
        print('[ Deprecated Warning: eval_model should be passed opt not Parser ]')
        opt = opt.parse_args()

    random.seed(42)

    # Create model and assign it to the specified task
    agent = create_agent(opt, requireModelExists=True)

    if print_parser:
        # Show arguments after loading model
        print_parser.opt = agent.opt
        print_parser.print_args()

    test_world = load_eval_world(agent, opt, 'test')
    t_report = run_eval(test_world, opt, 'test')
    test_world.shutdown()

if __name__ == '__main__':
    parser = setup_args()
    evalmodel_test(parser.parse_args(print_args=True), print_parser=parser)
        