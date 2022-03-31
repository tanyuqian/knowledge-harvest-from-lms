import os
import json

import numpy as np
from matplotlib import pyplot as plt

from data_utils.data_utils import conceptnet_relation_init_prompts


def main():
    window_width = 100

    for relation in conceptnet_relation_init_prompts:
        if not os.path.exists(f'outputs/{relation}/result.json'):
            continue

        result = json.load(open(f'outputs/{relation}/result.json', 'r'))

        scores = []
        for knowledge_term in result:
            ckbc_score = knowledge_term['ckbc score']
            comet_score = knowledge_term['comet score']

            scores.append(comet_score)

        y = []
        for i in range(window_width, len(scores)):
            y.append(sum(scores[i - window_width: i]) / window_width)

        plt.plot(np.arange(window_width, len(scores)), y)
        plt.title(relation)

        plt.savefig(f'curves/{relation}.png')
        plt.show()

        # result = []
        # for ent_tuple, weight in weighted_ent_tuples:
        #     ckbc_score = result
        #     comet_score = comet_scorer.score(
        #         h=ent_tuple[0], r=relation, t=ent_tuple[1])
        #     result.append({
        #         'entity tuple': ent_tuple,
        #         'weight': weight,
        #         'ckbc score': ckbc_score,
        #         'comet score': comet_score
        #     })

        json.dump(result, open(
            f'outputs/{relation}/result.json', 'w'), indent=4)


if __name__ == '__main__':
    main()