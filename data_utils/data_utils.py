import re
from nltk.corpus import stopwords


stopwords = stopwords.words('english')
stopwords.extend([
    'everything', 'everybody', 'everyone',
    'anything', 'anybody', 'anyone',
    'something', 'somebody', 'someone',
    'nothing', 'nobody',
    'one', 'neither', 'either'])


def get_index_in_prompt(ent_idx, prompt):
    return re.findall(f'<ENT[0-9]+>', prompt).index(f'<ENT{ent_idx}>')


def get_n_ents(prompt):
    n = 0
    while f'<ENT{n}>' in prompt:
        n += 1
    return n


def get_sent(prompt, ent_tuple):
    sent = prompt
    for idx, ent in enumerate(ent_tuple):
        sent = sent.replace(f'<ENT{idx}>', ent)

    return sent


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


def get_gpt3_prompt_paraphrase(sent):
    return f'paraphrase:\n{sent}\n'


def get_gpt3_prompt_mask_filling(prompt):
    prompt = re.sub(r'<ENT[0-9]+>', '[blank]', prompt)
    return f'fill in blanks:\n{prompt}\n'


def get_ent_tuple_from_sentence(sent, prompt):
    def _convert_re_prompt(matching):
        return f'(?P{matching.group()}.*)'

    re_prompt = re.sub(r'<ENT[0-9]+>', _convert_re_prompt, prompt)
    re_prompt = re.compile(re_prompt)
    matching = re.match(re_prompt, sent)

    if matching is None:
        return None

    ent_tuple = {'sentence': sent, 'prompt': prompt, 'ents': {}}
    for ent_idx in matching.re.groupindex:
        ent_tuple['ents'][ent_idx] = matching.group(ent_idx).lower()

    return ent_tuple
