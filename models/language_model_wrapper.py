import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from data_utils.data_utils import stopwords, get_n_ents, get_sent, find_sublist


class LanguageModelWrapper:
    def __init__(self, model_name):
        self._model_name = model_name
        self._max_batch_size = 64

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)
        self._encoder = getattr(self._model, model_name.split('-')[0])

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

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        mask_spans = self.get_mask_spans(prompt=prompt, ent_tuple=ent_tuple)

        mask_positions = []
        for mask_span in mask_spans:
            for i in range(mask_span[0], mask_span[1]):
                mask_positions.append(i)

        masked_inputs = self.tokenizer(
            [sent] * len(mask_positions), return_tensors='pt').to('cuda')
        label_token_ids = []
        for i in range(len(mask_positions)):
            label_token_ids.append(
                masked_inputs['input_ids'][i][mask_positions[i]])
            for pos in mask_positions[i:]:
                masked_inputs['input_ids'][i][pos] = \
                    self.tokenizer.mask_token_id

        with torch.no_grad():
            logits = self.model(**masked_inputs).logits
            logprobs = torch.log_softmax(logits, dim=-1)

        mask_logprobs = logprobs[
            torch.arange(0, len(mask_positions)), mask_positions,
            label_token_ids].tolist()

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

            ent = sent[len(prefix):].strip().split(' ')[0]  # -ing, -s, etc.
            ent_token_ids = self.tokenizer.encode(
                f' {ent}' if sent[len(prefix)] == ' ' else ent,
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

