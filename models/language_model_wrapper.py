import re

from transformers import RobertaTokenizer, RobertaForMaskedLM

from data_utils.data_utils import stopwords, get_n_ents, get_sent
from collections import defaultdict
import torch
import copy
from tqdm import *

class LanguageModelWrapper:
    def __init__(self, model_name):
        self._model_name = model_name
        self._max_batch_size = 64
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

    def get_masked_input_text(self, prompt, n_masks):
        if self._model_name == 'roberta-large':
            input_text = prompt
            for ent_idx, n_mask in enumerate(n_masks):
                input_text = input_text.replace(
                    f'<ENT{ent_idx}>', '<mask>' * n_mask)
        else:
            raise NotImplementedError
        
        return input_text

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
        if self._model_name == 'roberta-large':
            ids = []
            n_entities = len(tuples[0])
            for i in range(n_entities): 
                if not not_at_beginning[i]:
                    tuples = [tuple_[0].upper() + tuple_[1:] for tuple_ in tuples]
                ids.append(
                    self.tokenizer([tuple_[i] for tuple_ in tuples], \
                    add_special_tokens=False, add_prefix_space=not_at_beginning[i])['input_ids']
                )
        tuple_dict = defaultdict(list)
        for tuple_ids, tuple_texts in zip(zip(*ids), tuples):
            tuple_dict[tuple([len(entity) for entity in tuple_ids])].append((tuple_ids, tuple_texts))
        return tuple_dict

    def _get_batch_prediction(self, input_ids, token_type_ids, attention_mask, pos):
        with torch.no_grad():
            if self._model_name == "roberta-large":
                outputs = self.model(input_ids=input_ids,
                    attention_mask=attention_mask, labels=None)
            else:
                outputs = self.model(input_ids=input_ids, token_type_ids=token_type_ids,\
                    attention_mask=attention_mask, labels=None)
            prediction = torch.log_softmax(outputs.logits[:, pos, :], dim=-1)
        return prediction

    def score_tuples(self, tuples, prompt, n_masks):
        batch_size = min(self._max_batch_size, len(tuples))
        n_ents = len(n_masks)
        batch_prompt, pos_entities = self._tokenize_prompt_with_slots(prompt, n_masks, batch_size)
        return_scores = []
        for i in range((len(tuples) - 1)//batch_size + 1):                
            tuples_batch = tuples[i * batch_size: (i + 1) * batch_size]
            cur_batch_size = len(tuples_batch)
            batch_ids = copy.deepcopy(batch_prompt["input_ids"])[:cur_batch_size]
            batch_scores = [[] for _ in range(cur_batch_size)]
            for ent_idx in range(n_ents):
                for token in range(n_masks[ent_idx]):
                    target_pos = pos_entities[ent_idx] + token
                    target_ids = [tuples_batch[case_idx][0][ent_idx][token] for case_idx in range(cur_batch_size)]
                    
                    cur_scores = self._get_batch_prediction(batch_ids[:cur_batch_size], \
                        batch_prompt.get("token_type_ids", [])[:cur_batch_size], \
                        batch_prompt["attention_mask"][:cur_batch_size], target_pos)  
                        # roberta doesn't have "token_type_ids"

                    for case_idx in range(cur_batch_size):
                        batch_scores[case_idx].append(cur_scores[case_idx, target_ids[case_idx]])
                        batch_ids[case_idx][target_pos] = target_ids[case_idx]  
                        # fill in the blank for next token prediction

            return_scores += batch_scores
        return return_scores

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

    @property
    def all_special_ids(self):
        return self._tokenizer.all_special_ids


