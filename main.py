import os
import json
import fire

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.data_utils import conceptnet_relation_init_prompts


def main(n_tuples=10000, n_prompts=3, max_ent_repeat=5, max_ent_subwords=2):
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large',
        max_n_ent_tuples=n_tuples,
        max_ent_repeat=max_ent_repeat,
        max_ent_subwords=max_ent_subwords)

    for relation, init_prompts in conceptnet_relation_init_prompts.items():
        print(f'Harvesting for relation {relation}...')

        output_path = f'outputs_{n_tuples}tuples_{n_prompts}prompts' \
                      f'_maxsubwords{max_ent_subwords}' \
                      f'_maxrepeat{max_ent_repeat}/{relation}/'
        output_filename = 'weighted_ent_tuples.json'
        if os.path.exists(f'{output_path}/{output_filename}'):
            print(f'file {output_path}/{output_filename} exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_path}', exist_ok=True)
            json.dump([], open(f'{output_path}/{output_filename}', 'w'))

        knowledge_harvester.clear()
        knowledge_harvester.init_prompts(prompts=init_prompts[:n_prompts])
        knowledge_harvester.update_ent_tuples()

        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'{output_path}/{output_filename}', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
