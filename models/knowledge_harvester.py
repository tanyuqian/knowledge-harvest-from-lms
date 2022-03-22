from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher


class KnowledgeHarvester:
    def __init__(self, model_name):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._n_ents = None

        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

    def init_prompts(self, prompts):
        for prompt in prompts:
            self._weighted_prompts.append([prompt, 1.])

    def harvest(self):
        collected_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts, n=100)

        for weight, ent_tuple in collected_tuples:
            print(ent_tuple, weight)