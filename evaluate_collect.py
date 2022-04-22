# to generate the scores of given entity tuple
# and save them in a temporary file
# then run `aggregate_relation_pr_curve.ipynb` to plot the curve.
import os
import fire
import json

from models.knowledge_harvester import KnowledgeHarvester
from models.comet_knowledge_scorer import COMETKnowledgeScorer
from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer

from data_utils.ckbc import CKBC
# from data_utils import conceptnet_relation_init_prompts
# from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt

def main():
    ckbc = CKBC(file='conceptnet_high_quality.txt')
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large', max_n_ent_tuples=None)
    # comet_scorer = COMETKnowledgeScorer()
    # ckbc_scorer = CKBCKnowledgeScorer()

    save_dir = 'results/curves_ckbc_temp'
    os.makedirs(save_dir, exist_ok=True)

    relation_info = json.load(open('data/relation_info_conceptnet_5seeds.json'))
    # relation_info = conceptnet_relation_init_prompts
    for rel, info in relation_info.items():
        if rel not in ckbc._ent_tuples:
            continue
        ent_tuples = ckbc.get_ent_tuples(rel=rel)
        knowledge_harvester._max_n_prompts = 1

        prompts = info['init_prompts'] + info['prompts']
        for ind, prompt in enumerate(prompts):
            knowledge_harvester.clear()
            knowledge_harvester.init_prompts(prompts=[prompt])
            knowledge_harvester.set_seed_ent_tuples(info['seed_ent_tuples'])
            prompt_weight = knowledge_harvester.validate_prompts(metric_weights=(1/3, 1/3, 1/3))
            weighted_ent_tuples = knowledge_harvester.get_ent_tuples_weight(
                    ent_tuples=ent_tuples, metric_weights=(1/3, 1/3, 1/3))
            json.dump({"prompt": prompt, "weight": prompt_weight, "scores": weighted_ent_tuples}, open(
                save_dir + '{}_{}.json'.format(rel, ind), 'w'))

if __name__ == '__main__':
    fire.Fire(main)
