import os
import fire
import json

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.ckbc import CKBC

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


def main():
    # test_file = "conceptnet_high_quality.txt"
    # # weights = (.25, .25, 1)
    # weights = (0, 1, 0)
    # # test_file = 'test.txt'

    ckbc = CKBC(file='conceptnet_high_quality.txt')
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)

    # save_dir = f'neww_ckbc_{test_file.split(".")[0]}_{weights}_curves'
    save_dir = 'curves_high_quality/'
    os.makedirs(save_dir, exist_ok=True)

    relation_info = json.load(open('data/relation_info.json'))

    # for relation, init_prompts in conceptnet_relation_init_prompts.items():
    #     if relation not in ckbc._ent_tuples:
    #         continue

    for rel, info in relation_info.items():
        if rel not in ckbc._ent_tuples:
            continue
        else:
            ent_tuples = ckbc.get_ent_tuples(rel=rel)

        for setting in ['Init. prompt', 'Init. + GPT3 prompts']:
            prompts = info['init_prompts']
            if 'GPT3' in setting:
                prompts = prompts + info['prompts']

            knowledge_harvester.clear()
            knowledge_harvester.init_prompts(prompts=prompts)
            knowledge_harvester.set_seed_ent_tuples(info['seed_ent_tuples'])
            knowledge_harvester.update_prompts()

            weighted_ent_tuples = knowledge_harvester.get_ent_tuples_weight(
                ent_tuples=ent_tuples)

            y_true, scores = [], []
            for ent_tuple, weight in weighted_ent_tuples:
                label = ckbc.get_label(rel=rel, ent_tuple=ent_tuple)

                y_true.append(label)
                scores.append(weight)

            precision, recall, _ = precision_recall_curve(y_true, scores)
            plt.plot(recall, precision, label=setting)

        # plt.xlim((0, 1))
        # plt.ylim((0, 1))
        plt.legend()
        plt.title(f"{rel}: {len(ent_tuples)} tuples")

        plt.savefig(f"{save_dir}/{rel}.png")
        plt.figure().clear()


if __name__ == '__main__':
    fire.Fire(main)
