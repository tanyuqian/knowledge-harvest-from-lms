import os
import json

from models.knowledge_harvester import KnowledgeHarvester

from data_utils.data_utils import conceptnet_relation_init_prompts


def main():
    knowledge_harvester = KnowledgeHarvester(model_name='roberta-large')

    for relation, init_prompts in conceptnet_relation_init_prompts.items():
        print(f'Harvesting for relation {relation}...')

        output_path = f'outputs/{relation}/weighted_ent_tuples.json'
        if os.path.exists(output_path):
            print(f'file {output_path} exists, skipped.')
            continue
        else:
            os.makedirs(f'outputs/{relation}', exist_ok=True)
            json.dump([], open(output_path, 'w'), indent=4)

        knowledge_harvester.clear()
        knowledge_harvester.init_prompts(prompts=init_prompts)
        knowledge_harvester.update_ent_tuples()

        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'outputs/{relation}/weighted_ent_tuples.json', 'w'), indent=4)


if __name__ == '__main__':
    main()