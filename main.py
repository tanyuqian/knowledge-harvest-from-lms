import os
import json
import fire

from models.knowledge_harvester import KnowledgeHarvester


def main(rel_set='conceptnet',
         n_tuples=1000,
         n_prompts=20,
         prompt_temp=1.,
         max_ent_repeat=5,
         max_ent_subwords=2,
         n_seed_tuples=5,
         use_init_prompts=False):
    knowledge_harvester = KnowledgeHarvester(
        model_name='distilbert-base-uncased',
        max_n_ent_tuples=n_tuples,
        max_n_prompts=n_prompts,
        max_ent_repeat=max_ent_repeat,
        max_ent_subwords=max_ent_subwords,
        prompt_temp=prompt_temp)

    relation_info = json.load(open(
        f'data/relation_info_{rel_set}_{n_seed_tuples}seeds.json'))

    for rel, info in relation_info.items():
        print(f'Harvesting for relation {rel}...')

        output_dir = f'outputs/{rel_set}/' \
                     f'{n_tuples}tuples' \
                     f'_{n_prompts}prompts' \
                     f'_{n_seed_tuples}seeds' \
                     f'_maxsubwords{max_ent_subwords}' \
                     f'_maxrepeat{max_ent_repeat}' \
                     f'_temp{prompt_temp}'
        if use_init_prompts:
            output_dir += '_initprompts'
        if os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
            print(f'file {output_dir}/{rel}/ent_tuples.json exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
            json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

        knowledge_harvester.clear()

        knowledge_harvester.set_seed_ent_tuples(
            seed_ent_tuples=info['seed_ent_tuples'])
        prompts = info['init_prompts'] if use_init_prompts \
            else info['init_prompts'] + info['prompts']
        knowledge_harvester.init_prompts(prompts=prompts)

        knowledge_harvester.update_prompts()
        json.dump(knowledge_harvester.weighted_prompts, open(
            f'{output_dir}/{rel}/prompts.json', 'w'), indent=4)

        knowledge_harvester.update_ent_tuples()
        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'{output_dir}/{rel}/ent_tuples.json', 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
