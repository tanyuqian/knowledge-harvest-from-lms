import os
import json
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class COMETKnowledgeScorer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = './comet-atomic_2020_BART'
        self._model_path = model_path

        if not os.path.exists(model_path):
            os.system('bash data_utils/download_comet.sh')

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    @property
    def relations(self):
        added_tokens = json.load(open(f'{self._model_path}/added_tokens.json'))
        relations = [key for key in added_tokens.keys() if key != '[GEN]']
        return relations

    def score(self, h, r, t):
        query = f'{h} {r} [GEN]'
        inputs = self.tokenizer(query, return_tensors='pt').to(self.device)
        t_token_ids = self.tokenizer(
            t, return_tensors='pt')['input_ids'][:, 1:-1].to(self.device)

        outputs = self.model(**inputs, labels=t_token_ids)

        return -outputs.loss.item()


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        print(f"using task specific params for {task}: {pars}")
        model.config.update(pars)
