import re


def fix_ent_tuples(raw_ent_tuples):
    ent_tuples = []
    for ent_tuple in sorted(
            raw_ent_tuples, key=lambda t: t['sentence'].lower()):
        if len(ent_tuples) == 0:
            ent_tuples.append(ent_tuple)
        elif ent_tuple['sentence'].lower() != \
                ent_tuples[-1]['sentence'].lower():
            ent_tuples.append(ent_tuple)
        else:
            ent_tuples[-1]['logprob'] = max(
                ent_tuples[-1]['logprob'], ent_tuple['logprob'])

    ent_tuples.sort(key=lambda t: t['logprob'], reverse=True)

    return ent_tuples


def get_gpt3_prompt_mask_filling(prompt):
    prompt = re.sub(r'<ENT[0-9]+>', '[blank]', prompt)
    return f'fill in blanks:\n{prompt}\n'


def get_ent_tuple_from_sentence(sent, prompt):
    def _convert_re_prompt(matching):
        return f'(?P{matching.group()}.*)'

    re_prompt = re.sub(r'<ENT[0-9]+>', _convert_re_prompt, prompt)
    re_prompt = re.compile(re_prompt)
    matching = re.match(re_prompt, sent)

    # print('sent:', sent)
    # print('re_prompt:', re_prompt)

    if matching is None:
        return None

    ent_tuple = {'sentence': sent, 'prompt': prompt, 'ents': {}}
    for ent_idx in matching.re.groupindex:
        ent_tuple['ents'][ent_idx] = matching.group(ent_idx).lower()

    return ent_tuple
