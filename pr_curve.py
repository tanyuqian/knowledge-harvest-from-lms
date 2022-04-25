import os
import fire
import json
from glob import glob

import numpy as np

from matplotlib import pyplot as plt


def main(rel_set='conceptnet'):
    all_prec, all_recall = {}, {}

    for rel_pr in glob(f'curves/{rel_set}/*.json'):
        curves = json.load(open(rel_pr))

        for label in curves:
            if label not in all_prec:
                all_prec[label] = []
                all_recall[label] = []

            all_prec[label].extend(curves[label]['precision'])
            all_recall[label].extend(curves[label]['recall'])

    for label in all_prec:
        x = np.arange(0., 1., 0.001)

        poly_fn = np.poly1d(np.polyfit(all_recall[label], all_prec[label], 2))

        y = poly_fn(x)

        plt.plot(x, y, label=label)

    # plt.ylim(0, 1)

    plt.title(f'{rel_set} - All Relations')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(f'outputs/{rel_set}.png')
    plt.show()


if __name__ == '__main__':
    fire.Fire(main)