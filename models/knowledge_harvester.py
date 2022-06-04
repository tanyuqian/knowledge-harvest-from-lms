import torch

from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import get_n_ents, fix_prompt_style


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
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    def set_prompts(self, prompts):
        for prompt in prompts:
            self._weighted_prompts.append([fix_prompt_style(prompt), 1.])

    def update_prompts(self):
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            scores = []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]
                scores.append(self.score(prompt=prompt, ent_tuple=ent_tuple))

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

    def score(self, prompt, ent_tuple):


    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts