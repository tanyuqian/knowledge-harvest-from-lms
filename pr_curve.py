from collections import defaultdict
import os
import fire
import json
from glob import glob

import numpy as np

from matplotlib import pyplot as plt

def reorder(xs, ys):
    combined = list(zip(xs, ys))
    combined.sort(key=lambda x: x[0])
    return list(zip(*combined))

def main(rel_set='lama', model='distilbert-base-uncased'):
    # all_prec, all_recall = {}, {}
    pr_list = []
    int_r = np.arange(0., 1., 0.001)
    int_p = defaultdict(list)
    for rel_pr in glob(f'curves/{model}/{rel_set}/*.json'):
        curves = json.load(open(rel_pr))
        for label in curves:
            recall, precision = reorder(curves[label]['recall'], curves[label]['precision'])
            cur_int_p = np.interp(int_r, recall,\
                precision)
            int_p[label].append(cur_int_p)

            # all_prec[label].extend(curves[label]['precision'])
            # all_recall[label].extend(curves[label]['recall'])
            # pr_list.append((int_p))
            
    for label in int_p:
        aggr = np.array(int_p[label]).mean(0)
        plt.plot(int_r, aggr, label=label)
        # plt.plot(x, y, label=label)

    # plt.ylim(0.5, 1)
    # plt.xlim(0, 1)
    plt.title(f'{rel_set} - All Relations')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(f'outputs/{model}_{rel_set}.png')
    # plt.show()


if __name__ == '__main__':
    fire.Fire(main)