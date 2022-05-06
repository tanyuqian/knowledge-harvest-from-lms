from collections import defaultdict
import os
import fire
import json
from glob import glob

import numpy as np

from matplotlib import pyplot as plt


MODEL_NAMES = [
    'roberta-large',
    'roberta-base',
    'bert-large-cased',
    'distilbert-base-cased']


def reorder(xs, ys):
    combined = list(zip(xs, ys))
    combined.sort(key=lambda x: x[0])
    return list(zip(*combined))


def main(rel_set='lama', prompt_temp=1., setting='20 prompts'):
    int_r = np.arange(0., 1., 0.001)
    int_p = defaultdict(list)

    for model_name in MODEL_NAMES:
        for rel_pr in glob(f'curves/{model_name}-temp1.0/{rel_set}/*.json'):
            curves = json.load(open(rel_pr))

            if curves == []:
                continue

            # print(rel_pr)
            # print(curves[setting]['recall'])
            # print(curves[setting]['precision'])
            # print(reorder(curves[setting]['recall'], curves[setting]['precision']))
            # exit()

            recall, precision = reorder(curves[setting]['recall'], curves[setting]['precision'])
            cur_int_p = np.interp(int_r, recall, precision)
            int_p[model_name].append(cur_int_p)

            # all_prec[label].extend(curves[label]['precision'])
            # all_recall[label].extend(curves[label]['recall'])
            # pr_list.append((int_p))
            
    for model_name in int_p:
        aggr = np.array(int_p[model_name]).mean(0)
        plt.plot(int_r, aggr, label=model_name)
        # plt.plot(x, y, label=label)

    # plt.ylim(0.5, 1)
    # plt.xlim(0, 1)
    plt.title(f'{rel_set} - All Relations')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(f'outputs/{rel_set}_{setting}_temp{prompt_temp}.png')
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)