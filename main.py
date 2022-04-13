import os
import json
import fire

from models.knowledge_harvester import KnowledgeHarvester


def main(n_tuples=1000, max_ent_repeat=5, max_ent_subwords=2):
    knowledge_harvester = KnowledgeHarvester(
        model_name='roberta-large',
        max_n_ent_tuples=n_tuples,
        max_ent_repeat=max_ent_repeat,
        max_ent_subwords=max_ent_subwords)

    relation_info = json.load(open('data/relation_info.json'))

    for rel in relation_info:
        print(f'Harvesting for relation {rel}...')

        output_path = f'outputs_{n_tuples}tuples' \
                      f'_maxsubwords{max_ent_subwords}' \
                      f'_maxrepeat{max_ent_repeat}/{rel}/'
        output_filename = 'weighted_ent_tuples.json'
        if os.path.exists(f'{output_path}/{output_filename}'):
            print(f'file {output_path}/{output_filename} exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_path}', exist_ok=True)
            json.dump([], open(f'{output_path}/{output_filename}', 'w'))

        knowledge_harvester.clear()

        knowledge_harvester.init_prompts(
            prompts=relation_info[rel]['init_prompts'] +
                    relation_info[rel]['prompts'])

        knowledge_harvester.set_seed_ent_tuples(
            seed_ent_tuples=relation_info[rel]['seed_ent_tuples'])

        knowledge_harvester.update_prompts()
        knowledge_harvester.update_ent_tuples()

        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'{output_path}/{output_filename}', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
