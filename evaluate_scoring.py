import os
import fire
import json

from models.knowledge_harvester import KnowledgeHarvester
from models.comet_knowledge_scorer import COMETKnowledgeScorer
from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer

from data_utils.ckbc import CKBC

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


def main():
    ckbc = CKBC(file='conceptnet_high_quality.txt')
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)
    # comet_scorer = COMETKnowledgeScorer()
    # ckbc_scorer = CKBCKnowledgeScorer()

    save_dir = 'curves_high_quality/'
    os.makedirs(save_dir, exist_ok=True)

    relation_info = json.load(open('data/relation_info_conceptnet_5seeds.json'))

    for rel, info in relation_info.items():
        if rel not in ckbc._ent_tuples:
            continue
        else:
            ent_tuples = ckbc.get_ent_tuples(rel=rel)

        # for setting in ['ckbc', 'comet', 'init', 1, 5, 20]:
        for setting in ['init', 1, 5, 20]:
            '''
            if setting == 'ckbc':
                weighted_ent_tuples = []
                for ent_tuple in ent_tuples:
                    weighted_ent_tuples.append([ent_tuple, ckbc_scorer.score(
                        h=ent_tuple[0], r=rel, t=ent_tuple[1])])
            elif setting == 'comet':
                weighted_ent_tuples = []
                for ent_tuple in ent_tuples:
                    weighted_ent_tuples.append([ent_tuple, comet_scorer.score(
                        h=ent_tuple[0], r=rel, t=ent_tuple[1])])
            '''
            
            prompts = info['init_prompts'] if setting == 'init' \
                else info['prompts']

            knowledge_harvester.clear()
            if setting != 'init':
                knowledge_harvester._max_n_prompts = setting
                
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

            label = setting if setting in ['ckbc', 'comet'] \
                else f'{setting} prompts'
            plt.plot(recall, precision, label=label)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title(f"{rel}: {len(ent_tuples)} tuples")

        plt.savefig(f"{save_dir}/{rel}.png")
        plt.figure().clear()
        break

if __name__ == '__main__':
    fire.Fire(main)
