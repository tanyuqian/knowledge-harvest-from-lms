import os
import string

import fire
import json
from thefuzz import fuzz

from data_utils.data_utils import fix_prompt_style, is_valid_prompt
from search_prompts import get_paraphrase_prompt
from models.gpt3 import GPT3
from models.knowledge_harvester import KnowledgeHarvester


MAX_N_PROMPTS = 20
MAX_WORD_REPEAT = 3
MAX_ENT_SUBWORDS = 2
PROMPT_TEMP = 2.
SIMILARITY_THRESHOLD = 75


def search_prompts(init_prompts, seed_ent_tuples, similarity_threshold,
                   output_path):
    gpt3 = GPT3()

    cache = {}
    prompts = []
    while True:
        new_prompts = []
        for prompt in init_prompts + init_prompts + prompts:
            for ent_tuple in seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                request_str = f'{prompt} ||| {ent_tuple}'
                if request_str not in cache or prompt in init_prompts:
                    cache[request_str] = get_paraphrase_prompt(
                        gpt3=gpt3, prompt=prompt, ent_tuple=ent_tuple)

                para_prompt = cache[request_str]
                # print(f'prompt: {prompt}\tent_tuple: {ent_tuple}'
                #       f'\t-> para_prompt: {para_prompt}')

                if para_prompt is not None and \
                        para_prompt not in init_prompts + prompts:
                    new_prompts.append(para_prompt)

            if len(set(prompts + new_prompts)) >= 20:
                break

        if len(new_prompts) == 0:
            break
        else:
            # prompts.extend(new_prompts)
            flag = False
            for new_prompt in sorted(new_prompts, key=lambda t: len(t)):
                # if len(prompts) != 0:
                #     max_sim = max([fuzz.ratio(new_prompt, prompt)
                #                    for prompt in prompts])
                #     print(f'-- {new_prompt}: {max_sim}')
                if len(prompts) == 0 or \
                        max([fuzz.ratio(new_prompt, prompt)
                             for prompt in prompts]) < similarity_threshold:
                    prompts.append(new_prompt)
                    flag = True

            prompts = list(set(prompts))
            prompts.sort(key=lambda s: len(s))
            prompts = [prompt for prompt in prompts if is_valid_prompt(
                prompt=prompt)]

            json.dump(
                [fix_prompt_style(prompt) for prompt in prompts],
                open(output_path, 'w'),
                indent=4)

            print('=== Searched-out prompts for now ===')
            for prompt in prompts:
                print(prompt)
            print('=' * 50)

            if len(prompts) >= 10 or flag == False:
                break

    for i in range(len(prompts)):
        prompts[i] = fix_prompt_style(prompts[i])

    return prompts


def search_ent_tuples(
        init_prompts,
        seed_ent_tuples,
        prompts,
        model_name,
        max_n_ent_tuples,
        result_dir):

    knowledge_harvester = KnowledgeHarvester(
        model_name=model_name,
        max_n_ent_tuples=0,
        max_n_prompts=MAX_N_PROMPTS,
        max_word_repeat=MAX_WORD_REPEAT,
        max_ent_subwords=MAX_ENT_SUBWORDS,
        prompt_temp=PROMPT_TEMP)

    print(f'Searching with max_n_ent_tuples={max_n_ent_tuples}...')

    knowledge_harvester.clear()
    knowledge_harvester._max_n_ent_tuples = max_n_ent_tuples

    knowledge_harvester.set_seed_ent_tuples(seed_ent_tuples=seed_ent_tuples)
    knowledge_harvester.set_prompts(prompts=init_prompts + prompts)

    knowledge_harvester.update_prompts()
    json.dump(knowledge_harvester.weighted_prompts, open(
        f'{result_dir}/prompts.json', 'w'), indent=4)

    for prompt, weight in knowledge_harvester.weighted_prompts:
        print(f'{weight:.4f} {prompt}')

    knowledge_harvester.update_ent_tuples()
    json.dump(knowledge_harvester.weighted_ent_tuples, open(
        f'{result_dir}/ent_tuples.json', 'w'), indent=4)

    return knowledge_harvester.weighted_ent_tuples


def get_rel(init_prompts_str, seed_ent_tuples_str):
    init_prompts = init_prompts_str.split('^')
    seed_ent_tuples = [ent_tuple_str.split('~')
                       for ent_tuple_str in seed_ent_tuples_str.split('^')]

    seed_ent_tuples = sorted(
        [ent.lower().strip() for ent in ent_tuple]
        for ent_tuple in seed_ent_tuples)

    init_prompts_str = '^'.join(init_prompts)
    seed_ent_tuples_str = '^'.join(
        ['~'.join(ent_tuple) for ent_tuple in seed_ent_tuples])

    rel = f'INIT_PROMPTS-{init_prompts_str}_SEED-{seed_ent_tuples_str}'

    return init_prompts, seed_ent_tuples, rel


def main(init_prompts_str,
         seed_ent_tuples_str,
         model_name='roberta-large',
         max_n_ent_tuples=100):

    init_prompts, seed_ent_tuples, rel = get_rel(
        init_prompts_str=init_prompts_str,
        seed_ent_tuples_str=seed_ent_tuples_str)

    for i, prompt in enumerate(init_prompts):
        for ent_idx, ch in enumerate(string.ascii_uppercase):
            prompt = prompt.replace(ch, f'<ent{ent_idx}>')
        prompt = prompt.replace('_', ' ').replace('<ent', '<ENT')
        init_prompts[i] = fix_prompt_style(prompt=prompt)

    print(f'Initial prompts: {init_prompts}')
    print(f'Seed entity tuples: {seed_ent_tuples}')
    print('=' * 50)

    prompts_output_dir = f'results/demo/prompts/'
    prompts_output_path = f'{prompts_output_dir}/{rel}.json'
    if os.path.exists(prompts_output_path):
        prompts = json.load(open(prompts_output_path))
    else:
        os.makedirs(prompts_output_dir, exist_ok=True)
        json.dump([], open(prompts_output_path, 'w'))

        prompts = search_prompts(
            init_prompts=init_prompts,
            seed_ent_tuples=seed_ent_tuples,
            similarity_threshold=SIMILARITY_THRESHOLD,
            output_path=prompts_output_path)

    print('Searched-out prompts:')
    print('\n'.join(prompts))
    print('=' * 50)

    if max_n_ent_tuples == 0:
        return

    output_dir = f'results/demo/{max_n_ent_tuples}tuples/{model_name}'
    if os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
        print('Results found in cache.')
        for ent_tuple, weight in json.load(
                open(f'{output_dir}/{rel}/ent_tuples.json'))[:30]:
            print(ent_tuple, weight)
        print('=' * 50)
        return
    else:
        os.makedirs(f'{output_dir}/{rel}', exist_ok=True)
        json.dump([], open(f'{output_dir}/{rel}/ent_tuples.json', 'w'))

    weighted_ent_tuples = search_ent_tuples(
        init_prompts=init_prompts,
        seed_ent_tuples=seed_ent_tuples,
        prompts=prompts,
        model_name=model_name,
        max_n_ent_tuples=max_n_ent_tuples,
        result_dir=f'{output_dir}/{rel}/')

    for ent_tuple, weight in weighted_ent_tuples[:30]:
        print(ent_tuple, weight)
    print('=' * 50)


if __name__ == '__main__':
    fire.Fire(main)