import fire
import os
import json
from tqdm import tqdm

from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer
from models.comet_knowledge_scorer import COMETKnowledgeScorer


def main(output_dir):
    ckbc_scorer = CKBCKnowledgeScorer()
    comet_scorer = COMETKnowledgeScorer()

    relation_info = json.load(
        open(f'data/relation_info_conceptnet_5seeds.json'))

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
            ckbc_score = ckbc_scorer.score(
                h=ent_tuple[0], r=rel, t=ent_tuple[1])
            comet_score = comet_scorer.score(
                h=ent_tuple[0], r=rel, t=ent_tuple[1])
            result.append({
                'entity tuple': ent_tuple,
                'weight': weight,
                'ckbc score': ckbc_score,
                'comet score': comet_score
            })

        json.dump(result, open(
            f'{output_dir}/{rel}/scores.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)