import fire
import json
from nltk import sent_tokenize
from thefuzz import fuzz

from models.gpt3 import GPT3

from data_utils.data_utils import get_n_ents, get_sent, fix_prompt_style


TRANSFORMATIONS_SENT = [['', ''], ['a ', ''], ['the ', '']]
TRANSFORMATIONS_ENT = [
    ['', ''], ['being', 'is'], ['being', 'are'], ['ing', ''], ['ing', 'e']]


def get_paraphrase_prompt(gpt3, prompt, ent_tuple):
    assert get_n_ents(prompt) == len(ent_tuple)

    ent_tuple = [ent.lower() for ent in ent_tuple]
    sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)

    for _ in range(5):
        raw_response = gpt3.call(prompt=f'paraphrase:\n{sent}\n')

        para_sent = raw_response['choices'][0]['text']
        para_sent = sent_tokenize(para_sent)[0]
        para_sent = para_sent.strip().strip('.').lower()

        print('para_sent:', para_sent)

        prompt = para_sent
        valid = True
        for idx, ent in enumerate(ent_tuple):
            # prompt = prompt.replace(ent, f'<ENT{idx}>')
            for trans_sent in TRANSFORMATIONS_SENT:
                for trans_ent in TRANSFORMATIONS_ENT:
                    if prompt.count(f'<ENT{idx}>') == 0:
                        transed_prompt = prompt.replace(*trans_sent)
                        transed_ent = ent.replace(*trans_ent)
                        if transed_prompt.count(transed_ent) == 1:
                            prompt = transed_prompt.replace(
                                transed_ent, f'<ENT{idx}>')

            if prompt.count(f'<ENT{idx}>') != 1:
                valid = False
                break

        if valid:
            return prompt

    return None


def search_prompts(init_prompts, seed_ent_tuples, similarity_threshold):
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
                if len(prompts) == 0 or \
                        max([fuzz.ratio(new_prompt, prompt)
                             for prompt in prompts]) < similarity_threshold:
                    prompts.append(new_prompt)
                    flag = True

            prompts = list(set(prompts))
            prompts.sort(key=lambda s: len(s))

            if len(prompts) >= 10 or flag == False:
                break

    for i in range(len(prompts)):
        prompts[i] = fix_prompt_style(prompts[i])

    return prompts


def main(rel_set='conceptnet', similarity_threshold=75):
    relation_info = json.load(open(f'relation_info/{rel_set}.json'))

    for rel, info in relation_info.items():
        info['init_prompts'] = [
            fix_prompt_style(prompt) for prompt in info['init_prompts']]

        if 'prompts' not in info or len(info['prompts']) == 0:
            info['prompts'] = search_prompts(
                init_prompts=info['init_prompts'],
                seed_ent_tuples=info['seed_ent_tuples'],
                similarity_threshold=similarity_threshold)

            for key, value in info.items():
                print(f'{key}: {value}')
            for prompt in info['prompts']:
                print(f'- {prompt}')
            print('=' * 50)

        output_path = f'relation_info/{rel_set}.json'
        json.dump(relation_info, open(output_path, 'w'), indent=4)


if __name__ == '__main__':
    fire.Fire(main)
