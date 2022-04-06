import fire

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.ckbc import CKBC
from data_utils.data_utils import conceptnet_relation_init_prompts

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


def main():
    ckbc = CKBC()
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)

    for relation, init_prompts in conceptnet_relation_init_prompts.items():
        if relation not in ckbc._ent_tuples:
            continue

        for n_prompts in [1, 2, 3]:
            knowledge_harvester.clear()
            knowledge_harvester.init_prompts(prompts=init_prompts[:n_prompts])

            weighted_ent_tuples = knowledge_harvester.get_ent_tuples_weight(
                ckbc.get_ent_tuples(rel=relation))

            y_true, scores = [], []
            for ent_tuple, weight in weighted_ent_tuples:
                label = ckbc.get_label(rel=relation, ent_tuple=ent_tuple)

                y_true.append(label)
                scores.append(weight)

            precision, recall, _ = precision_recall_curve(y_true, scores)
            plt.plot(recall, precision, label=f'{n_prompts}prompts')

        plt.legend()
        plt.title(relation)

        plt.savefig(f'ckbc_curves/{relation}.png')
        plt.figure().clear()


if __name__ == '__main__':
    fire.Fire(main)
