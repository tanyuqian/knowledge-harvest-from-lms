from collections import defaultdict
import os
import fire
import json
from glob import glob

import numpy as np

from matplotlib import pyplot as plt

# Set the default text font size
plt.rc('font', size=16)
# Set the axes title font size
plt.rc('axes', titlesize=16)
# Set the axes labels font size
plt.rc('axes', labelsize=16)
# Set the font size for x tick labels
plt.rc('xtick', labelsize=16)
# Set the font size for y tick labels
plt.rc('ytick', labelsize=16)
# Set the legend font size
plt.rc('legend', fontsize=12)
# Set the font size of the figure title
plt.rc('figure', titlesize=20)

MODELS = [
    'roberta-large', 'roberta-base',
    'bert-large-cased', 'bert-base-cased',
    'distilbert-base-cased']


LABEL_TRANS = {
    'autoprompt': 'Prompt Tuning',
    'comet': 'COMET',
    'LPAQA-manual_paraphrase': 'LPAQA-MP',
    'LPAQA-mine': 'LPAQA-M',
    'LPAQA-paraphrase': 'LPAQA-P',
    'init prompts': 'Initial Prompt',
    '1 prompts': 'Best Prompt (ours)',
    '20 prompts': 'Multiple Prompts (ours)'
}


def reorder(xs, ys):
    combined = list(zip(xs, ys))
    combined.sort(key=lambda x: x[0])
    return list(zip(*combined))


def main(rel_sets='all', prompt_temp=1.):
    # all_prec, all_recall = {}, {}
    pr_list = []
    int_r = np.arange(0., 1., 0.001)
    int_p = defaultdict(list)

    if rel_sets == 'all':
        rel_sets = ['conceptnet', 'lama']
    else:
        rel_sets = [rel_sets]

    for model in MODELS:
        for rel_set in rel_sets:
            print(f'curves/{model}-temp{prompt_temp}/{rel_set}/*.json')
            for rel_pr in glob(
                    f'curves/{model}-temp{prompt_temp}/{rel_set}/*.json'):
                print(rel_pr)

                curves = json.load(open(rel_pr))
                for label in curves:
                    recall, precision = reorder(
                        curves[label]['recall'], curves[label]['precision'])
                    cur_int_p = np.interp(int_r, recall, precision)
                    int_p[label].append(cur_int_p)

            # all_prec[label].extend(curves[label]['precision'])
            # all_recall[label].extend(curves[label]['recall'])
            # pr_list.append((int_p))
            
    for label in int_p:
        if label not in LABEL_TRANS:
            continue
        aggr = np.array(int_p[label]).mean(0)
        plt.plot(int_r, aggr, label=LABEL_TRANS[label])
        # plt.plot(x, y, label=label)

    # plt.ylim(0.5, 1)
    # plt.xlim(0, 1)
    rel_sets = 'all' if len(rel_sets) == 2 else rel_sets[0]
    # plt.title(f'{rel_sets} Relations - {model}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(f'outputs/{rel_sets}_{model}_temp{prompt_temp}.png')
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)