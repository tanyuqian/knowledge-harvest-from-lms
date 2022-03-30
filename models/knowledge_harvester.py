import torch

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import get_n_ents, get_sent


class KnowledgeHarvester:
    def __init__(self, model_name, max_n_prompts=20, max_n_ent_tuples=1000):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples

        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []

    def init_prompts(self, prompts):
        for prompt in prompts:
            self._weighted_prompts.append([prompt, 1.])

    def get_ent_tuple_weight(self, ent_tuple):
        score = 0.
        for prompt, weight in self._weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)
        return score

    def score(self, prompt, ent_tuple):
        assert get_n_ents(prompt) == len(ent_tuple)

        sent = get_sent(prompt=prompt, ent_tuple=ent_tuple)
        mask_spans = self._model.get_mask_spans(
            prompt=prompt, ent_tuple=ent_tuple)

        masked_inputs = self._model.tokenizer(
            sent, return_tensors='pt').to('cuda')
        truth_input_ids = self._model.tokenizer(sent)['input_ids']
        for mask_span in mask_spans:
            for i in range(mask_span[0], mask_span[1]):
                masked_inputs['input_ids'][0][i] = self._model.mask_token_id

        score = 1.
        for mask_span in mask_spans:
            for mask_pos in range(mask_span[0], mask_span[1]):
                logits = self._model.model(**masked_inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]
                score *= probs[mask_pos][truth_input_ids[mask_pos]].item()

                masked_inputs['input_ids'][0][mask_pos] = \
                    truth_input_ids[mask_pos]

        return score

    def update_ent_tuples(self):
        collected_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts, n=self._max_n_ent_tuples)

        ent_tuples = sorted(
            [t[0] for t in self._weighted_ent_tuples] + collected_tuples)
        ent_tuples = [ent_tuples[i] for i in range(len(ent_tuples))
                      if i == 0 or ent_tuples[i] != ent_tuples[i - 1]]

        ent_tuple_weights = [
            self.get_ent_tuple_weight(ent_tuple) for ent_tuple in ent_tuples]

        self._weighted_ent_tuples = [
            [ent_tuple, weight]
            for ent_tuple, weight in zip(ent_tuples, ent_tuple_weights)]

        self._weighted_ent_tuples.sort(key=lambda t: t[1], reverse=True)
        self._weighted_ent_tuples = \
            self._weighted_ent_tuples[:self._max_n_ent_tuples]

        for ent_tuple, weight in self._weighted_ent_tuples:
            print(ent_tuple, weight)

    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts