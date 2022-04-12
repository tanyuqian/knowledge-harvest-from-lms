import fire
import os
import json
from tqdm import tqdm

from models.ckbc_knowledge_scorer import CKBCKnowledgeScorer
from models.comet_knowledge_scorer import COMETKnowledgeScorer

from data_utils.data_utils import conceptnet_relation_init_prompts


def main(output_path):
    ckbc_scorer = CKBCKnowledgeScorer()
    comet_scorer = COMETKnowledgeScorer()

    for relation, _ in conceptnet_relation_init_prompts.items():
        if not os.path.exists(
                f'{output_path}/{relation}/weighted_ent_tuples.json'):
            continue

        weighted_ent_tuples = json.load(open(
            f'{output_path}/{relation}/weighted_ent_tuples.json', 'r'))

        if len(weighted_ent_tuples) == 0:
            print(f'{output_path}/{relation}/weighted_ent_tuples.json: '
                  f'empty tuples, skipped.')
            continue

        output_filename = 'result.json'
        if os.path.exists(f'{output_path}/{relation}/{output_filename}'):
            print(f'file {output_path}/{relation}/{output_filename} exists'
                  f', skipped.')
            continue
        else:
            os.makedirs(f'{output_path}/{relation}', exist_ok=True)
            json.dump([], open(
                f'{output_path}/{relation}/{output_filename}', 'w'))

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
            f'{output_path}/{relation}/{output_filename}', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)