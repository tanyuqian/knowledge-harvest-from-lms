import os
import fire
import json

from search_prompts import search_prompts
from models.knowledge_harvester import KnowledgeHarvester


MAX_N_PROMPTS = 20
MAX_WORD_REPEAT = 5
MAX_ENT_SUBWORDS = 2
PROMPT_TEMP = 2.


def main(init_prompts,
         seed_ent_tuples,
         model_name='roberta-large',
         similarity_threshold=75):

    rel = f'INIT_PROMPTS-{init_prompts}_SEED-{seed_ent_tuples}'

    init_prompts = init_prompts.split('^')
    seed_ent_tuples = [
        ent_tuple.split('~') for ent_tuple in seed_ent_tuples.split('^')]

    print(f'Initial prompts: {init_prompts}')
    print(f'Seed entity tuples: {seed_ent_tuples}')
    print('=' * 50)

    prompts = search_prompts(
        init_prompts=init_prompts,
        seed_ent_tuples=seed_ent_tuples,
        similarity_threshold=similarity_threshold)

    print('Searched-out prompts:')
    print('\n'.join(prompts))
    print('=' * 50)

    knowledge_harvester = KnowledgeHarvester(
        model_name=model_name,
        max_n_ent_tuples=0,
        max_n_prompts=MAX_N_PROMPTS,
        max_word_repeat=MAX_WORD_REPEAT,
        max_ent_subwords=MAX_ENT_SUBWORDS,
        prompt_temp=PROMPT_TEMP)

    for max_n_ent_tuples in [2 ** t for t in range(6, 10)]:
        output_dir = f'results/demo' \
                     f'/sim_{similarity_threshold}' \
                     f'/{model_name}' \
                     f'/{max_n_ent_tuples}tuples/'
        if os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
            print(f'file {output_dir}/{rel}/ent_tuples.json exists, skipped.')
            continue
        else:
            os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
            json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

        print(f'Searching with max_n_ent_tuples={max_n_ent_tuples}...')

        knowledge_harvester.clear()
        knowledge_harvester._max_n_ent_tuples = max_n_ent_tuples

        knowledge_harvester.set_seed_ent_tuples(seed_ent_tuples=seed_ent_tuples)
        knowledge_harvester.set_prompts(prompts=init_prompts + prompts)

        knowledge_harvester.update_prompts()
        json.dump(knowledge_harvester.weighted_prompts, open(
            f'{output_dir}/{rel}/prompts.json', 'w'), indent=4)

        for prompt, weight in knowledge_harvester.weighted_prompts:
            print(f'{weight:.4f} {prompt}')

        knowledge_harvester.update_ent_tuples()
        json.dump(knowledge_harvester.weighted_ent_tuples, open(
            f'{output_dir}/{rel}/ent_tuples.json', 'w'), indent=4)

        for ent_tuple, weight in knowledge_harvester.weighted_ent_tuples[:30]:
            print(ent_tuple, weight)
        print('=' * 50)


if __name__ == '__main__':
    fire.Fire(main)