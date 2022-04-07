import fire

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.ckbc import CKBC
from data_utils.data_utils import conceptnet_relation_init_prompts

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt
import os

def main():
    test_file = "conceptnet_high_quality.txt"
    # weights = (.25, .25, 1)
    weights = (0, 1, 0)
    # test_file = 'test.txt'
    ckbc = CKBC(test_file)
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)
    save_dir = f'neww_ckbc_{test_file.split(".")[0]}_{weights}_curves'
    os.makedirs(save_dir, exist_ok=True)
    for relation, init_prompts in conceptnet_relation_init_prompts.items():
        if relation not in ckbc._ent_tuples:
            continue

        n_tuples = len(ckbc.get_ent_tuples(rel=relation))
        for n_prompts in [1, 2, 3]:
            knowledge_harvester.clear()
            prompts = init_prompts
            knowledge_harvester.init_prompts(prompts=prompts[:n_prompts])
            weighted_ent_tuples = knowledge_harvester.get_ent_tuples_weight(
                ckbc.get_ent_tuples(rel=relation), weights=weights)

            y_true, scores = [], []
            for ent_tuple, weight in weighted_ent_tuples:
                label = ckbc.get_label(rel=relation, ent_tuple=ent_tuple)

                y_true.append(label)
                scores.append(weight)

            precision, recall, _ = precision_recall_curve(y_true, scores)
            plt.plot(recall, precision, label=f'{n_prompts}prompts')
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.legend()
        plt.title(f"{relation}: {n_tuples} tuples")

        plt.savefig(f"{save_dir}/{relation}.png")
        plt.figure().clear()
        # 
if __name__ == '__main__':
    fire.Fire(main)
