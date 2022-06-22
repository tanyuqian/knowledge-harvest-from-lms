import string
import torch
from copy import deepcopy

from transformers import AutoTokenizer, AutoModelForMaskedLM

from data_utils.data_utils import stopwords, get_n_ents, get_sent, find_sublist
from collections import defaultdict
import copy
import re

MODELS = {
    "roberta": ["roberta-large", "roberta-base"],
    "bert": ["bert-large-cased", "bert-base-cased", "bert-base-uncased", "bert-large-uncased"],
    "albert": ["albert-base-v2"],
    "distilbert": ["distilbert-base-uncased", "distilbert-base-cased"],
}


class LanguageModelWrapper:
    def __init__(self, model_name):
        self._model_name = model_name

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(model_name)

        self._model.eval()
        # self._model.to('cuda')

        self._banned_ids = None
        self._get_banned_ids()

    def _get_banned_ids(self):
        self._banned_ids = self._tokenizer.all_special_ids
        alphabeta = re.compile("(##)?[a-zA-Z0-9].*")
        for idx in range(self._tokenizer.vocab_size):
            token = self._tokenizer.decode(idx).strip().lower()
            obj = re.search(alphabeta, token)
            if not (obj and obj[0] == token and len(token) > 1 and token not in stopwords): # not a word or wordpiece
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
        # [[l1, l1], [l2, r2], ...]

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
            ent = sent[len(prefix):].strip().split()[0]
            for punc in string.punctuation:
                ent = ent.split(punc)[0]

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

    def _tokenize_prompt_with_slots(self, prompt, n_masks, batch_size):
        mask_token = self.tokenizer.mask_token
        n_entities = len(n_masks)
        order = [(ent_idx, prompt.find(f'<ENT{ent_idx}>')) for ent_idx in range(n_entities)]
        order = sorted(order, key=lambda x: x[-1])  # rank the entity idx by the position
        for ent_idx in range(n_entities):
            prompt = prompt.replace(f"<ENT{ent_idx}>", " ".join([mask_token] * n_masks[ent_idx]))
        batch_prompts = self.tokenizer([prompt] * batch_size, return_tensors="pt").to('cuda')
        pos_mask = [ind for ind, token_id in enumerate(batch_prompts['input_ids'][0]) \
            if self.tokenizer.mask_token_id == token_id.item()]
        pos_entities = [0] * n_entities
        ptr = 0
        for rank in range(n_entities):
            ent_idx = order[rank][0]
            pos_entities[ent_idx] = pos_mask[ptr]
            ptr += n_masks[ent_idx]
        return batch_prompts, pos_entities

    def tokenize_tuples_by_len(self, tuples, not_at_beginning):
        # note: This code doesn't work for prompt like "...<ENT0>s ..."
        # but it should be fine if the prompt ranking also uses this code because it will filter out prompts like that.
        ids = []
        n_entities = len(tuples[0])
        tuples_fit = copy.deepcopy(tuples)
        for i in range(n_entities): 

            if not not_at_beginning[i]:
                for tuple_idx in range(len(tuples_fit)):
                    ent = tuples_fit[tuple_idx][i]
                    tuples_fit[tuple_idx][i] = ent[0].upper() + ent[1:]
            else:
                for tuple_idx in range(len(tuples_fit)):
                    tuples_fit[tuple_idx][i] = " " + tuples_fit[tuple_idx][i]
            if self._model_name in MODELS["roberta"]:
                # need to handle prefix space for roberta
                ids.append(
                    self.tokenizer([tuple_[i] for tuple_ in tuples_fit], \
                    # add_special_tokens=False, add_prefix_space=not_at_beginning[i])['input_ids']
                    add_special_tokens=False)['input_ids']
                )
            elif self._model_name in MODELS["bert"] + MODELS["distilbert"] + MODELS["albert"]:
                ids.append(
                    self.tokenizer([tuple_[i] for tuple_ in tuples_fit], \
                        add_special_tokens=False)['input_ids']
                )
            else:
                raise NotImplementedError
        tuple_dict = defaultdict(list)
        for tuple_ids, tuple_texts in zip(zip(*ids), tuples):
            tuple_dict[tuple([len(entity) for entity in tuple_ids])].append((tuple_ids, tuple_texts))
        return tuple_dict

    def _get_batch_prediction(self, input_ids, token_type_ids, attention_mask, pos):
        with torch.no_grad():
            if self._model_name in MODELS["roberta"] + MODELS["distilbert"]:
                outputs = self.model(input_ids=input_ids,
                    attention_mask=attention_mask, labels=None)
            elif self._model_name in MODELS["bert"] + MODELS["albert"]:
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids,\
                    attention_mask=attention_mask, labels=None)
            else:
                raise NotImplementedError
            prediction = torch.log_softmax(outputs.logits[:, pos, :], dim=-1)
        return prediction

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def banned_ids(self):
        return self._banned_ids

