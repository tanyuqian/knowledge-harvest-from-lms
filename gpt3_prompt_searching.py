import json

from models.gpt3 import GPT3

from data_utils.data_utils import conceptnet_relation_init_prompts
from data_utils.concept_net import ConceptNet


def main(n_seed_tuples=5):
    gpt3 = GPT3()
    conceptnet = ConceptNet()

    cache = {}

    relation_info = {}
    for rel, init_prompts in conceptnet_relation_init_prompts.items():
        init_prompts = init_prompts[:1]
        seed_ent_tuples = conceptnet.get_ent_tuples(rel=rel)[:n_seed_tuples]

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
                prompts.extend(new_prompts)
                prompts = list(set(prompts))
                prompts.sort(key=lambda s: len(s))

                if len(prompts) >= 20:
                    break

        relation_info[rel] = {
            'init_prompts': init_prompts,
            'seed_ent_tuples': seed_ent_tuples,
            'prompts': prompts
        }

        print(f'Relation: {rel}')
        print(f'Init Prompt: {init_prompts}')
        print(f'Seed Entity Tuples: {seed_ent_tuples}')
        for prompt in prompts:
            print(f'- {prompt}')
        print('=' * 50)

        json.dump(relation_info, open(
            f'data/relation_info_{n_seed_tuples}seeds.json', 'w'), indent=4)


if __name__ == '__main__':
    main()
