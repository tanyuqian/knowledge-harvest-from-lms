import os
import fire
import json
import torch

from models.knowledge_harvester import KnowledgeHarvester
from models.comet_knowledge_scorer import COMETKnowledgeScorer
from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer

from data_utils.ckbc import CKBC
from data_utils.lpaqa import LPAQA

from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


SETTINGS = {
    'conceptnet': ['ckbc', 'comet', 'init', 1, 5, 20],
    'lama': ['cls', 'LPAQA-manual_paraphrase', 'LPAQA-mine', 'LPAQA-paraphrase',
             'init', 1, 5, 20]
}


def main(rel_set='conceptnet'):
    ckbc = CKBC(rel_set=rel_set)
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)

    if rel_set == 'conceptnet':
        comet_scorer = COMETKnowledgeScorer()
        ckbc_scorer = CKBCKnowledgeScorer()
    elif rel_set == 'lama':
        lpaqa = {}
        for s in ['manual_paraphrase', 'mine', 'paraphrase']:
            lpaqa[s] = LPAQA(setting=s)

        lama_scorer = torch.load('roberta-large_lama_1e-05_0.0001_bestmodel.pt')

    save_dir = f'curves/{rel_set}'
    os.makedirs(save_dir, exist_ok=True)

    relation_info = json.load(open(f'data/relation_info_{rel_set}_5seeds.json'))

    curves = {}
    for rel, info in relation_info.items():
        if rel not in ckbc._ent_tuples:
            continue
        else:
            ent_tuples = ckbc.get_ent_tuples(rel=rel)

        if os.path.exists(f'{save_dir}/{rel}.json'):
            print(f'{save_dir}/{rel}.json exists, skipped.')
            continue
        else:
            json.dump([], open(f'{save_dir}/{rel}.json', 'w'))

        for setting in SETTINGS[rel_set]:
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
            elif setting == 'cls':
                weighted_ent_tuples = []
                for ent_tuple in ent_tuples:
                    weighted_ent_tuples.append([ent_tuple, lama_scorer.score(
                        h=ent_tuple[0].title(),
                        r=' '.join(rel.split('_')[1:]),
                        t=ent_tuple[1].title())])
            else:
                knowledge_harvester.clear()
                if type(setting) == int:
                    knowledge_harvester._max_n_prompts = setting

                if type(setting) == str and 'LPAQA' in setting:
                    prompts = lpaqa[setting.split('-')[-1]].get_prompts(rel=rel)
                    for prompt in prompts:
                        knowledge_harvester._weighted_prompts.append(
                            [prompt['prompt'], prompt['weight']])

                    for prompt, weight in knowledge_harvester._weighted_prompts:
                        print(f'{weight:.6f} {prompt}')

                else:
                    prompts = info['init_prompts'] if setting == 'init' \
                        else info['prompts']

                    knowledge_harvester.init_prompts(prompts=prompts)
                    knowledge_harvester.set_seed_ent_tuples(
                        info['seed_ent_tuples'])
                    knowledge_harvester.update_prompts()

                weighted_ent_tuples = knowledge_harvester.get_ent_tuples_weight(
                    ent_tuples=ent_tuples)

            y_true, scores = [], []
            for ent_tuple, weight in weighted_ent_tuples:
                label = ckbc.get_label(rel=rel, ent_tuple=ent_tuple)

                y_true.append(label)
                scores.append(weight)

            precision, recall, _ = precision_recall_curve(y_true, scores)

            label = setting if type(setting) == str and setting != 'init' \
                else f'{setting} prompts'
            plt.plot(recall, precision, label=label)

            curves[label] = {
                'precision': precision.tolist(),
                'recall': recall.tolist()
            }

        json.dump(curves, open(f'{save_dir}/{rel}.json', 'w'), indent=4)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.title(f"{rel}: {len(ent_tuples)} tuples")

        plt.savefig(f"{save_dir}/{rel}.png")
        plt.figure().clear()


if __name__ == '__main__':
    fire.Fire(main)