import torch

from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import get_n_ents, get_sent


class KnowledgeHarvester:
    def __init__(self,
                 model_name,
                 max_n_prompts=20,
                 max_n_ent_tuples=10000,
                 max_ent_repeat=10,
                 max_ent_subwords=1,
                 prompt_temp=1.):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_ent_repeat = max_ent_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp

        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

        self._seed_ent_tuples = None

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []

    def init_prompts(self, prompts):
        for prompt in prompts:
            prompt = prompt.strip(' .').lower().replace('<ent', '<ENT')
            self._weighted_prompts.append([prompt, 1.])

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    # def get_ent_tuple_weight(self, ent_tuple):
    #     score = 0.
    #     for prompt, weight in self._weighted_prompts:
    #         score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)
    #     return score

    # need to be batchified
    def get_ent_tuples_weight(self, ent_tuples, metric_weights=(.25, .25, .5)):
        scores, tuples = self._score_tuples_prompts(ent_tuples)
        # now the score has the shape (n_prompts, n_tuples, 3)
        metric_weights = torch.tensor(metric_weights)
        scores = torch.sum(scores * metric_weights.reshape(1, 1, *metric_weights.shape)\
            .expand(*scores.shape), dim=-1)
        # aggregate all the metrics. Now (n_prompts, n_tuples)

        weights = torch.tensor([weight for prompt, weight in self._weighted_prompts])
        scores = torch.sum(scores * weights.reshape(*weights.shape, 1).expand(*scores.shape), dim=0)
        scores_with_tuples = [(ent_tuple, score.item()) for score, ent_tuple in zip(scores, tuples)]
        return scores_with_tuples
        
    def _score_tuples_prompts(self, ent_tuples):
        result_list = []
        tuples_list = []
        n_ents = get_n_ents(self._weighted_prompts[0][0])
        tuples_list = sorted(ent_tuples) # must be in the same order... (alphabet)
        for prompt, weight in self._weighted_prompts:
            prompt_result_list = []
            not_at_beginning = [prompt.strip().find(f'<ENT{i}>') != 0 for i in range(n_ents)]
            tokenized_entity_pairs = self._model.tokenize_tuples_by_len(ent_tuples, not_at_beginning)
            # {(len_ent_1, len_ent_2, ...): [(ent_1_ids, ent_2_ids, ...), (ent_1_text, ent_2_text, ...), ...]}
            print(f"scoring with prompts: '{prompt}' ...")
            for n_masks, tuples in tokenized_entity_pairs.items():
                # print(f"scoring tuples of length {n_masks}... {len(tuples)} in total.")
                scores = self._model.score_tuples(tuples, prompt, n_masks=n_masks)
                for score, ent_tuple in zip(scores, tuples):
                    # print(f"score and tuple: {score}, {ent_tuple}")
                    prompt_result_list.append((ent_tuple[1], (sum(score).item()/sum(n_masks), sum(score).item()/len(n_masks), min(score).item()))) # += [(ent_tuple[1], score) ]
            prompt_result_list = sorted(prompt_result_list, key=lambda x:x[0])
            result_list.append([result[1] for result in prompt_result_list])
        return torch.tensor(result_list), tuples_list  # (n_prompt, n_tuples, 3), (n_tuples,)

    def score_single(self, prompt, ent_tuple, weights=(0.25, 0.25, 0.5)):
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

        scores = []
        for mask_span in mask_spans:
            for mask_pos in range(mask_span[0], mask_span[1]):
                logits = self._model.model(**masked_inputs).logits
                logprobs = torch.log_softmax(logits, dim=-1)[0]
                scores.append(
                    logprobs[mask_pos][truth_input_ids[mask_pos]].item())
                masked_inputs['input_ids'][0][mask_pos] = \
                    truth_input_ids[mask_pos]

        sum_score = sum(scores) / len(ent_tuple)
        mean_score = sum(scores) / len(scores)
        min_score = min(scores)
        return weights[0] * sum_score + weights[1] * mean_score + weights[2] * min_score

    def update_ent_tuples(self):
        collected_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts,
            n=self._max_n_ent_tuples,
            max_ent_repeat=self._max_ent_repeat,
            max_ent_subwords=self._max_ent_subwords)

        ent_tuples = sorted(
            [t[0] for t in self._weighted_ent_tuples] + collected_tuples)

        ent_tuples = [ent_tuples[i] for i in range(len(ent_tuples))
                      if i == 0 or ent_tuples[i] != ent_tuples[i - 1]]

        self._weighted_ent_tuples = self.get_ent_tuples_weight(ent_tuples)
        self._weighted_ent_tuples = sorted(
            self._weighted_ent_tuples,
            key=lambda t: t[1], reverse=True)[:self._max_n_ent_tuples]

    def validate_prompts(self, metric_weights):
        weights = []
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            scores = []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                scores.append(self.score_single(
                    prompt=prompt, ent_tuple=ent_tuple, weights=metric_weights))
            weights.append(sum(scores) / len(scores))
        return weights

    def update_prompts(self, metric_weights):
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            scores = []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                scores.append(self.score_single(
                    prompt=prompt, ent_tuple=ent_tuple), weights=metric_weights)

            self._weighted_prompts[i][1] = \
                sum(scores) / len(scores) / self._prompt_temp

        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight

        for prompt, weight in self._weighted_prompts:
            print(f'{weight:.4f} {prompt}')

    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts