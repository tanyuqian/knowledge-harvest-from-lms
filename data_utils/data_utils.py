import re
import json
from nltk.corpus import stopwords


stopwords = stopwords.words('english')
stopwords.extend([
    'everything', 'everybody', 'everyone',
    'anything', 'anybody', 'anyone',
    'something', 'somebody', 'someone',
    'nothing', 'nobody',
    'one', 'neither', 'either',
    'us', 'first', 'second', 'next',
    'following', 'last', 'new', 'main'])


conceptnet_relation_init_prompts = {
    'AtLocation': [
        '<ENT1> is the location for <ENT0> .',
        'the <ENT1> is where the <ENT0> is kept .',
        '<ENT0> is located in <ENT1> .'
    ],
    'CapableOf': [
        'Something that <ENT0> can typically do is <ENT1> .',
        # 'A <ENT0> can typically <ENT1> .',
        # '<ENT1> could be done by <ENT0> .',
        "<ENT0> will <ENT1> .",
        'A <ENT0> is capable to <ENT1> .'
    ],
    'Causes': [
        'It is typical for <ENT0> to cause <ENT1> .',
        # 'many people <ENT1> when they <ENT0>',
        'People would expect <ENT1> after <ENT0> .',
        '<ENT0> typically causes <ENT1> .'
    ],
    'CausesDesire': [
        '<ENT0> makes someone want <ENT1> .',
        '<ENT1> is a desire caused by <ENT0> .',
        # '<ENT0> causes <ENT1>'
        '<ENT0> causes people to <ENT1> .'
    ],
    'CreatedBy': [
        '<ENT1> is a process or agent that creates <ENT0> .',
        'a <ENT1> is someone or something that creates <ENT0> .',
        'the process or agent that creates <ENT0> is <ENT1> .'
    ],
    'DefinedAs': [
        '<ENT1> is a more explanatory version of <ENT0> .',
        'the concept of <ENT0> is often described as the <ENT1> .',
        '<ENT1> is a more accurate explanation of what <ENT0> is .'
    ],
    'Desires': [
        # '<ENT0> is a conscious entity that typically wants <ENT1>',
        '<ENT1> is something that every <ENT0> desires .',
        '<ENT0> desires <ENT1> .',
        # '<ENT0> usually wants <ENT1>'
        '<ENT0> usually wants to <ENT1>. '
    ],
    'HasA': [
        'Usually, we would expect <ENT0> to have <ENT1> .',
        'the <ENT1> is part of the <ENT0> .',
        '<ENT0> can not function without <ENT1>. '
    ],
    'HasFirstSubevent': [
        '<ENT0> is an event that begins with subevent <ENT1>',
        '<ENT0> begins with <ENT1>',
        '<ENT0> typically starts with <ENT1>',
    ],
    'HasLastSubevent': [
        '<ENT0> is an event that concludes with subevent <ENT1>',
        'After you finish <ENT0>, you have to <ENT1>',
        '<ENT0> usually includes <ENT1> afterwards'
    ],
    'HasPrerequisite': [
        # 'In order for <ENT0> to happen, <ENT1> needs to happen',
        'If someone wants to <ENT0>, they need to <ENT1> first .',
        # '<ENT1> is a dependency of <ENT0>',
        "It is impossible to <ENT0> if you don't <ENT1> .",
        # '<ENT1> requires <ENT0>'
        "You need to <ENT1> before <ENT0> ."
    ],
    'HasProperty': [
        # '<ENT0> has <ENT1> as a property',
        '<ENT0> are <ENT1> .',
        '<ENT0> can be described as <ENT1> .',
        '<ENT0> is known to be <ENT1> .'
    ],
    'HasSubEvent': [
        '<ENT1> happens as a subevent of <ENT0> .',
        '<ENT0> entails <ENT1> .',
        'when <ENT0>, one often has to <ENT1> .'
    ],
    'IsA': [
        '<ENT0> is a subtype or a specific instance of <ENT1> .',
        'every <ENT0> is a <ENT1> .',
        'every <ENT0> is a type of <ENT1> .',
    ],
    'MadeOf': [
        '<ENT0> is made of <ENT1> .',
        'the <ENT0> is made from <ENT1> .',
        'the <ENT0> is composed of <ENT1> .'
    ],
    'MotivatedByGoal': [
        # 'Someone does <ENT0> because they want result <ENT1>',
        # '<ENT0> is a step toward accomplishing the goal <ENT1>',
        # '<ENT0> is a necessary step in order to <ENT1>'
        'Someone does <ENT0> because they want to <ENT1> .',
        'What drives people to <ENT0> is the goal to <ENT1> .',
        '<ENT1> is the goal for people to <ENT0> .'
    ],
    'PartOf': [
        '<ENT0> is a part of <ENT1>',
        'the <ENT0> is located in the <ENT1>',
        '<ENT1> includes <ENT0> as one of its many components'
    ],
    'ReceivesAction': [
        '<ENT1> can be done to <ENT0>',
        'you can <ENT1> a <ENT0>',
        'you can <ENT1> your own <ENT0>'
    ],
    'SymbolOf': [
        '<ENT0> symbolically represents <ENT1>',
        '<ENT0> is often seen as a symbol of <ENT1>',
        '<ENT0> is often used to represent <ENT1>'
    ],
    'UsedFor': [
        '<ENT0> is used for <ENT1> .',
        'the purpose of <ENT0> is <ENT1> .',
        '<ENT0> is for <ENT1> .'
    ]
}


def get_relations(rel_set):
    assert rel_set in ['conceptnet', 'lama']

    return list(json.load(open(
        f'data/relation_info_{rel_set}_5seeds.json')).keys())


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
