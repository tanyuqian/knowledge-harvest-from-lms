import fire
import json
import random

from models.gpt3 import GPT3

from data_utils.data_utils import conceptnet_relation_init_prompts
from data_utils.concept_net import ConceptNet
from data_utils.lama import LAMA

from thefuzz import fuzz


SEED = 11111


def search_prompts(init_prompts, seed_ent_tuples):
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
                    cache[request_str] = gpt3.get_paraphrase_prompt(
                        prompt=prompt, ent_tuple=ent_tuple)

                para_prompt = cache[request_str]
                print(f'prompt: {prompt}\tent_tuple: {ent_tuple}'
                      f'\t-> para_prompt: {para_prompt}')

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
                if len(prompts) != 0:
                    max_sim = max([fuzz.ratio(new_prompt, prompt)
                                   for prompt in prompts])
                    print(f'-- {new_prompt}: {max_sim}')
                if len(prompts) == 0 or max([fuzz.ratio(
                        new_prompt, prompt) for prompt in prompts]) < 75:
                    prompts.append(new_prompt)
                    flag = True

            prompts = list(set(prompts))
            prompts.sort(key=lambda s: len(s))

            if len(prompts) >= 10 or flag == False:
                break

    return prompts


def main(rel_set='conceptnet', n_seed_tuples=5):
    random.seed(SEED)

    relation_info = {}
    if rel_set == 'conceptnet':
        conceptnet = ConceptNet()
        for rel, init_prompts in conceptnet_relation_init_prompts.items():
            ent_tuples = conceptnet.get_ent_tuples(rel=rel)
            seed_ent_tuples = random.sample(
                ent_tuples, k=min(n_seed_tuples, len(ent_tuples)))

            relation_info[rel] = {
                'init_prompts': init_prompts[:1],
                'seed_ent_tuples': seed_ent_tuples
            }
    elif rel_set == 'lama':
        lama = LAMA()
        for rel, info in lama.info.items():
            print(rel, len(info['ent_tuples']))
            seed_ent_tuples = random.sample(info['ent_tuples'], k=n_seed_tuples)

            relation_info[rel] = {
                'init_prompts': info['init_prompts'],
                'seed_ent_tuples': seed_ent_tuples
            }
    elif rel_set == 'human':
        relation_info = json.load(open('data/relations_human_5seeds.json'))
    else:
        raise ValueError

    for rel, info in relation_info.items():
        if info['prompts'] != []:
            continue

        info['prompts'] = search_prompts(
            init_prompts=info['init_prompts'],
            seed_ent_tuples=info['seed_ent_tuples'])

        for key, value in info.items():
            print(f'{key}: {value}')
        for prompt in info['prompts']:
            print(f'- {prompt}')
        print('=' * 50)

        output_path = f'data/relation_info_{rel_set}_{n_seed_tuples}seeds.json'
        json.dump(relation_info, open(output_path, 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
