import re
from nltk.corpus import stopwords


stopwords = stopwords.words('english')
stopwords.extend([
    'everything', 'everybody', 'everyone',
    'anything', 'anybody', 'anyone',
    'something', 'somebody', 'someone',
    'nothing', 'nobody',
    'one', 'neither', 'either', 'many',
    'us', 'first', 'second', 'next',
    'following', 'last', 'new', 'main', 'also'])


def is_valid_prompt(prompt):
    for i in range(1, len(prompt)):
        if prompt[i:].startswith('<ENT') and prompt[i - 1] not in [' ', '\"']:
            return False

    return True


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


def get_mask_place(ent_idx, n_masks, prompt):
    mask_idx = 0
    for t in re.findall(r'<ENT[0-9]+>', prompt):
        t_idx = int(t[len('<ENT'):-1])
        if t_idx != ent_idx:
            mask_idx += n_masks[t_idx]
        else:
            break

    return mask_idx


def get_n_masks(t, n_ents, max_ent_subwords):
    n_masks = []
    for i in range(n_ents):
        n_masks.append(t % max_ent_subwords + 1)
        t //= max_ent_subwords

    return n_masks


def get_masked_prompt(prompt, n_masks, mask_token):
    input_text = prompt
    for ent_idx, n_mask in enumerate(n_masks):
        input_text = input_text.replace(f'<ENT{ent_idx}>', mask_token * n_mask)

    return input_text


def fix_prompt_style(prompt):
    prompt = prompt.strip(' .')
    if prompt[0].isalpha():
        prompt = prompt[0].upper() + prompt[1:]

    return prompt + ' .'


def find_sublist(a, b):
    for l in range(len(a)):
        if a[l:l+len(b)] == b:
            return l

    return None
