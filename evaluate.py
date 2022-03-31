import os
import json
from tqdm import tqdm

from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer
from models.comet_knowledge_scorer import COMETKnowledgeScorer

from data_utils.data_utils import conceptnet_relation_init_prompts


def main():
    ckbc_scorer = CKBCKnowledgeScorer()
    comet_scorer = COMETKnowledgeScorer()

    for relation, _ in conceptnet_relation_init_prompts.items():
        if not os.path.exists(f'outputs/{relation}/weighted_ent_tuples.json'):
            continue

        output_path = f'outputs/{relation}/result.json'
        if os.path.exists(output_path):
            print(f'file {output_path} exists, skipped.')
            continue
        else:
            os.makedirs(f'outputs/{relation}', exist_ok=True)
            json.dump([], open(output_path, 'w'), indent=4)

        weighted_ent_tuples = json.load(open(
            f'outputs/{relation}/weighted_ent_tuples.json', 'r'))

        result = []
        for ent_tuple, weight in tqdm(weighted_ent_tuples,
                                      desc=f'evaluating {relation}'):
            ckbc_score = ckbc_scorer.score(
                h=ent_tuple[0], r=relation, t=ent_tuple[1])
            comet_score = comet_scorer.score(
                h=ent_tuple[0], r=relation, t=ent_tuple[1])
            result.append({
                'entity tuple': ent_tuple,
                'weight': weight,
                'ckbc score': ckbc_score,
                'comet score': comet_score
            })

        json.dump(result, open(
            f'outputs/{relation}/result.json', 'w'), indent=4)


if __name__ == '__main__':
    main()