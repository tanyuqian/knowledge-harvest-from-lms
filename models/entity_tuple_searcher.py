import torch

from data_utils.data_utils import get_n_ents, get_index_in_prompt


class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model

    def search(self, weighted_prompts, top_k):
        n_ents = get_n_ents(weighted_prompts[0][0])

        collected_tuples = []
        self.dfs(
            weighted_prompts=weighted_prompts,
            n_ents=n_ents,
            cur_ent_idx=0,
            cur_ent_tuple=[],
            cur_weight=1.,
            collected_tuples=collected_tuples,
            top_k=top_k)

        collected_tuples.sort(key=lambda t: t[1], reverse=True)

        return collected_tuples

    def dfs(self,
            weighted_prompts,
            n_ents,
            cur_ent_idx,
            cur_ent_tuple,
            cur_weight,
            collected_tuples,
            top_k):
        if cur_ent_idx == n_ents:
            if len(set([e.strip().lower() for e in cur_ent_tuple])) == \
                    len(cur_ent_tuple):
                collected_tuples.append([cur_ent_tuple, cur_weight])
            return

        mask_state = None
        for prompt, weight in weighted_prompts:
            input_text = self._model.get_masked_input_text(prompt=prompt)
            inputs = self._model.tokenizer(input_text, return_tensors="pt")

            outputs = self._model.encoder(**inputs)
            sequence_output = outputs.last_hidden_state[
                inputs['input_ids'] == self._model.mask_token_id]

            index_in_prompt = get_index_in_prompt(
                ent_idx=cur_ent_idx, prompt=prompt)
            if mask_state is None:
                mask_state = torch.zeros_like(sequence_output[index_in_prompt])
            mask_state = mask_state + sequence_output[index_in_prompt] * weight

        mask_state = mask_state / sum(weight for _, weight in weighted_prompts)

        logits = self._model.lm_head(mask_state.reshape(1, -1))
        logits[::, self._model.banned_ids] = -float('inf')
        probs = torch.softmax(logits, dim=-1)[0]
        probs, pred_ids = torch.sort(probs, descending=True)

        for prob, pred_id in zip(probs[:top_k], pred_ids[:top_k]):
            pred_ent = self._model.tokenizer.decode(pred_id)

            weighted_prompts_upd = []
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append(
                    [prompt.replace(f'<ENT{cur_ent_idx}>', pred_ent), weight])

                self.dfs(
                    weighted_prompts=weighted_prompts_upd,
                    n_ents=n_ents,
                    cur_ent_idx=cur_ent_idx + 1,
                    cur_ent_tuple=cur_ent_tuple + [pred_ent],
                    cur_weight=cur_weight * prob.item(),
                    collected_tuples=collected_tuples,
                    top_k=top_k)
