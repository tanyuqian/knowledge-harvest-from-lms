import fire
import os
import json
from tqdm import tqdm
import torch

from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer


def main(output_dir):
    if 'conceptnet' in output_dir:
        rel_set = 'conceptnet'
        scorer = CKBCKnowledgeScorer()
    else:
        assert 'lama' in output_dir
        rel_set = 'lama'
        scorer = torch.load('roberta-large_lama_1e-05_0.0001_bestmodel.pt')

    relation_info = json.load(open(f'data/relation_info_{rel_set}_5seeds.json'))

    for rel, info in relation_info.items():
        if not os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
            continue

        weighted_ent_tuples = json.load(open(
            f'{output_dir}/{rel}/ent_tuples.json', 'r'))

        if len(weighted_ent_tuples) == 0:
            print(f'{output_dir}/{rel}/ent_tuples.json: empty tuples, skipped.')
            continue

        if os.path.exists(f'{output_dir}/{rel}/scores.json'):
            print(f'file {output_dir}/{rel}/scores.json exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
            json.dump([], open(f'{output_dir}/{rel}/scores.json', 'w'))

        result = []
        for ent_tuple, weight in tqdm(weighted_ent_tuples,
                                      desc=f'evaluating {rel}'):
            r = rel if rel_set == 'conceptnet' \
                else ' '.join(rel.split('_')[1:])

            score = scorer.score(h=ent_tuple[0], r=r, t=ent_tuple[1])
            result.append({
                'entity tuple': ent_tuple,
                'weight': weight,
                'cls score': score
            })

        json.dump(result, open(
            f'{output_dir}/{rel}/scores.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)