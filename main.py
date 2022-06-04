import os
import json
import fire

from models.knowledge_harvester import KnowledgeHarvester


def main(rel_set='conceptnet',
         model_name='roberta-large',
         max_n_ent_tuples=1000,
         max_n_prompts=20,
         prompt_temp=2.,
         max_ent_repeat=5,
         max_ent_subwords=2,
         use_init_prompts=False):

    knowledge_harvester = KnowledgeHarvester(
        model_name=model_name,
        max_n_ent_tuples=max_n_ent_tuples,
        max_n_prompts=max_n_prompts,
        max_ent_repeat=max_ent_repeat,
        max_ent_subwords=max_ent_subwords,
        prompt_temp=prompt_temp)

    relation_info = json.load(open(f'relation_info/{rel_set}.json'))

    for rel, info in relation_info.items():
        print(f'Harvesting for relation {rel}...')

        setting = f'{n_tuples}tuples'
        if use_init_prompts:
            setting += '_initprompts'
        else:
            setting += f'_top{n_prompts}prompts'

        output_dir = f'results/{rel_set}/{setting}/{model_name}'
        if os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
            print(f'file {output_dir}/{rel}/ent_tuples.json exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
            json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

        knowledge_harvester.clear()

        knowledge_harvester.set_seed_ent_tuples(
            seed_ent_tuples=info['seed_ent_tuples'])
        knowledge_harvester.set_prompts(
            prompts=info['init_prompts'] + info['prompts'])

        knowledge_harvester.update_prompts()
        json.dump(knowledge_harvester.weighted_prompts, open(
            f'{output_dir}/{rel}/prompts.json', 'w'), indent=4)

        for prompt, weight in knowledge_harvester.weighted_prompts:
            print(f'{weight:.4f} {prompt}')

        knowledge_harvester.update_ent_tuples()
        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'{output_dir}/{rel}/ent_tuples.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
