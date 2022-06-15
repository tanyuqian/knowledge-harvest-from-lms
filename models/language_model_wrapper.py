import string
import torch
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForMaskedLM

from data_utils.data_utils import stopwords, get_n_ents, get_sent, find_sublist


class LanguageModelWrapper:
    def __init__(self, model_name):
        self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

        self._model.eval()
        self._model.to('cuda')

        self._banned_ids = None
        self._get_banned_ids()

    def _get_banned_ids(self):
        self._banned_ids = self._tokenizer.all_special_ids
        for idx in range(self._tokenizer.vocab_size):
            if self._tokenizer.decode(idx).lower().strip() in stopwords:
                self._banned_ids.append(idx)

    def get_mask_logits(self, input_text):
        with torch.no_grad():
            inputs = self.tokenizer(input_text, return_tensors="pt").to('cuda')
            outputs = self.model(**inputs)

        return outputs.logits[
            inputs['input_ids'] == self.tokenizer.mask_token_id]

    def get_mask_filling_logprobs(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        ent_tuple = deepcopy(ent_tuple)
        for ent_idx, ent in enumerate(ent_tuple):
            if prompt.startswith(f'<ENT{ent_idx}>'):
                ent_tuple[ent_idx] = ent[0].upper() + ent[1:]

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        mask_spans = self.get_mask_spans(prompt=prompt, ent_tuple=ent_tuple)

        mask_positions = []
        for mask_span in mask_spans:
            mask_positions.extend([pos for pos in range(*mask_span)])

        masked_inputs = self.tokenizer(
            [sent] * len(mask_positions), return_tensors='pt').to('cuda')
        label_token_ids = []
        for i, pos in enumerate(mask_positions):
            label_token_ids.append(masked_inputs['input_ids'][i][pos].item())
            masked_inputs['input_ids'][i][mask_positions[i:]] = \
                self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(**masked_inputs).logits
            logprobs = torch.log_softmax(logits, dim=-1)

        mask_logprobs = logprobs[
            torch.arange(len(mask_positions)), mask_positions,
            label_token_ids].tolist()

        torch.cuda.empty_cache()

        return {
            'input_ids': self.tokenizer.encode(sent),
            'mask_spans': mask_spans,
            'mask_positions': mask_positions,
            'mask_logprobs': mask_logprobs
        }

    def get_mask_spans(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        input_ids = self._tokenizer.encode(sent)

        mask_spans = []
        for ent_idx, ent in enumerate(ent_tuple):
            prefix = prompt[:prompt.find(f'<ENT{ent_idx}>')].strip()
            for i in range(len(ent_tuple)):
                prefix = prefix.replace(f'<ENT{i}>', ent_tuple[i])
            prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

            # processing -ing, -s, etc.
            ent_in_sent = prompt[prompt.find(f'<ENT{ent_idx}>'):].split()[0]
            for punc in string.punctuation:
                ent_in_sent = ent_in_sent.split(punc)[0]
            ent_in_sent = ent_in_sent.replace(f'<ENT{ent_idx}>', ent)

            ent_token_ids = self.tokenizer.encode(
                f' {ent_in_sent}' if sent[len(prefix)] == ' ' else ent_in_sent,
                add_special_tokens=False)

            if len(prefix_ids) > 0:
                l = find_sublist(input_ids, prefix_ids) + len(prefix_ids)
            else:
                l = find_sublist(input_ids, ent_token_ids)
            r = l + len(ent_token_ids)

            assert input_ids[l:r] == ent_token_ids
            mask_spans.append([l, r])

        return mask_spans

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def banned_ids(self):
        return self._banned_ids

