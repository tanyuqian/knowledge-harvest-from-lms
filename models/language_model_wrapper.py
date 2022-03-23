import re

from transformers import RobertaTokenizer, RobertaForMaskedLM

from data_utils.data_utils import stopwords, get_n_ents, get_sent


class LanguageModelWrapper:
    def __init__(self, model_name):
        self._model_name = model_name

        if model_name == 'roberta-large':
            self._tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self._model = RobertaForMaskedLM.from_pretrained(model_name)
            self._encoder = self._model.roberta
            self._lm_head = self._model.lm_head
        else:
            raise NotImplementedError

        self._model.eval()
        self._model.to('cuda')

        self._banned_ids = None
        self._get_banned_ids()

    def _get_banned_ids(self):
        self._banned_ids = self._tokenizer.all_special_ids
        for idx in range(self._tokenizer.vocab_size):
            if self._tokenizer.decode(idx).lower().strip() in stopwords:
                self._banned_ids.append(idx)

    def get_masked_input_text(self, prompt):
        if self._model_name == 'roberta-large':
            return re.sub(r'<ENT[0-9]+>', '<mask>', prompt)

    def get_mask_spans(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        input_ids = self._tokenizer(sent)['input_ids']

        mask_spans = []
        for ent_idx, ent in enumerate(ent_tuple):
            prefix = prompt[:prompt.find(f'<ENT{ent_idx}>')].strip()
            for i in range(len(ent_tuple)):
                prefix = prefix.replace(f'<ENT{i}>', ent_tuple[i])

            if self._model_name == 'roberta-large':
                l = 1
                while not self._tokenizer.decode(input_ids[1:l]) == prefix:
                    l += 1

                r = l + 1
                while not self._tokenizer.decode(input_ids[1:r]).endswith(ent):
                    r += 1
            else:
                raise NotImplementedError

            mask_spans.append([l, r])

        return mask_spans

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def encoder(self):
        return self._encoder

    @property
    def lm_head(self):
        return self._lm_head

    @property
    def mask_token_id(self):
        return self._tokenizer.mask_token_id

    @property
    def banned_ids(self):
        return self._banned_ids


