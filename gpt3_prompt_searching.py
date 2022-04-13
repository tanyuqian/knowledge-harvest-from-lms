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
        for ent_tuple in seed_ent_tuples + seed_ent_tuples:
            new_prompts = []
            for prompt in init_prompts + prompts:
                print(f'prompt: {prompt}; ent_tuple: {ent_tuple}')
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                request_str = f'{prompt} ||| {ent_tuple}'
                if request_str not in cache or cache[request_str] is None:
                    cache[request_str] = gpt3.get_paraphrase_prompt(
                        prompt=prompt, ent_tuple=ent_tuple)

                para_prompt = cache[request_str]

                if para_prompt is not None:
                    new_prompts.append(para_prompt)

            prompts.extend(new_prompts)
            prompts = list(set(prompts))

            if len(prompts) >= 10:
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

        json.dump(relation_info, open('data/relation_info.json', 'w'), indent=4)


if __name__ == '__main__':
    main()

# gpt3 = GPT3()
#
# prompt = '<ENT0> makes someone want <ENT1>'
# ent_tuple = ['warm weather', 'play baseball']
#
# print(gpt3.get_paraphrase_prompt(prompt=prompt, ent_tuple=ent_tuple))
