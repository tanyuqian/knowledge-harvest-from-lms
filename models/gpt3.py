import os
import openai
from nltk import sent_tokenize

from data_utils.data_utils import get_ent_tuple_from_sentence,\
    get_gpt3_prompt_mask_filling, get_gpt3_prompt_paraphrase,\
    fix_ent_tuples, get_n_ents, get_sent

TRANSFORMATIONS_SENT = [['', ''], ['a ', ''], ['the ', '']]
TRANSFORMATIONS_ENT = [
    ['', ''], ['being', 'is'], ['being', 'are'], ['ing', ''], ['ing', 'e']]


class GPT3:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def get_raw_response(self,
                         prompt,
                         engine="text-davinci-002",
                         temperature=1.,
                         max_tokens=30,
                         top_p=1.,
                         frequency_penalty=0,
                         presence_penalty=0,
                         logprobs=0,
                         n=1):
        return openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            n=n)

    def get_ent_tuples(self, prompt, n):
        gpt3_prompt = get_gpt3_prompt_mask_filling(prompt=prompt)

        choices = []
        while n > 0:
            raw_response = self.get_raw_response(
                prompt=gpt3_prompt, n=min(n, 128))
            choices.extend(raw_response['choices'])
            n -= min(n, 128)

        ent_tuples = []
        for choice in choices:
            try:
                sent = choice['text'].strip().strip('.')
                assert '\n' not in sent
                assert len(sent_tokenize(sent)) == 1

                tokens = choice['logprobs']['tokens']
                token_logprobs = choice['logprobs']['token_logprobs']

                sent_logprob = 0.
                for token, token_logprob in zip(tokens, token_logprobs):
                    if token == '<|endoftext|>':
                        break
                    elif token not in ['\n', '.']:
                        assert token in sent
                        sent_logprob += token_logprob

                ent_tuple = get_ent_tuple_from_sentence(
                    sent=sent, prompt=prompt)
                ent_tuple['logprob'] = sent_logprob

                ent_tuples.append(ent_tuple)

            except:
                print('an error.')
                print('choice:', choice)

        ent_tuples = fix_ent_tuples(raw_ent_tuples=ent_tuples)

        return ent_tuples

    def get_paraphrase_prompt(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)

        for _ in range(5):
            raw_response = self.get_raw_response(
                prompt=get_gpt3_prompt_paraphrase(sent))

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

            # words = word_tokenize(para_sent)
            # if any([(ent not in words) for ent in ent_tuple]):
            #     continue
            # for idx, ent in enumerate(ent_tuple):
            #     words[words.index(ent)] = f'<ENT{idx}>'
            #
            # return ' '.join(words)

        return None
