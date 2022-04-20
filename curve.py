import fire
import os
import json
from glob import glob

import numpy as np
from matplotlib import pyplot as plt


WINDOW_WIDTH = 200


OUTPUT_DIRS = {
    'init prompt': 'outputs_1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_initprompts',
    '1 prompt': 'outputs_1000tuples_1prompts_5seeds_maxsubwords2_maxrepeat5',
    '20 prompts': 'outputs_1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5'
}


def main(metric='ckbc'):
    os.makedirs(f'curves_outputs_{metric}/', exist_ok=True)

    relation_info = json.load(open(f'data/relation_info_5seeds.json'))

    for rel in relation_info:
        for setting, output_dir in OUTPUT_DIRS.items():
            if not os.path.exists(f'{output_dir}/{rel}/scores.json'):
                continue

            result = json.load(open(f'{output_dir}/{rel}/scores.json'))
            if result == []:
                continue

            scores = []
            for knowledge_term in result:
                score = knowledge_term[f'{metric} score']

                scores.append(score)

            y = []
            for i in range(WINDOW_WIDTH, len(scores)):
                y.append(sum(scores[i - WINDOW_WIDTH: i]) / WINDOW_WIDTH)

            plt.plot(np.arange(WINDOW_WIDTH, len(scores)), y, label=setting)

        plt.title(f'{rel}: {metric} score')
        plt.legend()

        plt.savefig(f'curves_outputs_{metric}/{rel}.png')
        plt.figure().clear()


if __name__ == '__main__':
    fire.Fire(main)