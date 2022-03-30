import re
from nltk.corpus import stopwords


stopwords = stopwords.words('english')
stopwords.extend([
    'everything', 'everybody', 'everyone',
    'anything', 'anybody', 'anyone',
    'something', 'somebody', 'someone',
    'nothing', 'nobody',
    'one', 'neither', 'either',
    'us'])


conceptnet_relation_init_prompts = {
    'AtLocation': [
        '<ENT0> is a typical location for <ENT1>',
        '<ENT0> is the inherent location of <ENT1>'
    ],
    'CapableOf': [
        'Something that <ENT0> can typically do is <ENT1>'
    ],
    'Causes': [
        'It is typical for <ENT0> to cause <ENT1>'
    ],
    'CausesDesire': [
        '<ENT0> makes someone want <ENT1>'
    ],
    'CreatedBy': [
        '<ENT1> is a process or agent that creates <ENT0>'
    ],
    'DefinedAs': [
        '<ENT0> and <ENT1> overlap considerably in meaning',
        '<ENT1> is a more explanatory version of <ENT0>'
    ],
    'Desires': [
        '<ENT0> is a conscious entity that typically wants <ENT1>'
    ],
    'HasA': [
        '<ENT1> belongs to <ENT0> as an inherent part',
        '<ENT1> belongs to <ENT0> due to a social construct of possession'
    ],
    'HasFirstSubevent': [
        '<ENT0> is an event that begins with subevent <ENT1>'
    ],
    'HasLastSubevent': [
        '<ENT0> is an event that concludes with subevent <ENT1>'
    ],
    'HasPrerequisite': [
        'In order for <ENT0> to happen, <ENT1> needs to happen',
        '<ENT1> is a dependency of <ENT0>'
    ],
    'HasProperty': [
        '<ENT0> has <ENT1> as a property',
        '<ENT0> can be described as <ENT1>'
    ],
    'HasSubEvent': [
        '<ENT1> happens as a subevent of <ENT0>'
    ],
    'IsA': [
        '<ENT0> is a subtype or a specific instance of <ENT1>',
        'every <ENT0> is a <ENT1>'
    ],
    'MadeOf': [
        '<ENT0> is made of <ENT1>'
    ],
    'MotivatedByGoal': [
        'Someone does <ENT0> because they want result <ENT1>',
        '<ENT0> is a step toward accomplishing the goal <ENT1>'
    ],
    'PartOf': [
        '<ENT0> is a part of <ENT1>'
    ],
    'ReceivesAction': [
        '<ENT1> can be done to <ENT0>'
    ],
    'SymbolOf': [
        '<ENT0> symbolically represents <ENT1>'
    ],
    'UsedFor': [
        '<ENT0> is used for <ENT1>',
        'the purpose of <ENT0> is <ENT1>'
    ]
}


def get_mask_index_in_prompt(ent_idx, n_masks, prompt):
    mask_idx = 0
    for t in re.findall(r'<ENT[0-9]+>', prompt):
        t_idx = int(t[len('<ENT'):-1])
        if t_idx != ent_idx:
            mask_idx += n_masks[t_idx]
        else:
            break

    return mask_idx


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
