import torch
import heapq

from data_utils.data_utils import get_n_ents, get_index_in_prompt


class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model

    def search(self, weighted_prompts, n):
        n_ents = get_n_ents(weighted_prompts[0][0])

        collected_tuples_heap = []
        self.dfs(
            weighted_prompts=weighted_prompts,
            n_ents=n_ents,
            cur_ent_idx=0,
            cur_ent_tuple=[],
            cur_weight=1.,
            collected_tuples_heap=collected_tuples_heap,
            n=n)

        collected_tuples = sorted(
            collected_tuples_heap, key=lambda t: t[0], reverse=True)

        for weight, ent_tuple in collected_tuples:
            print(ent_tuple, weight)
        print('=' * 50)

        return [t[1] for t in collected_tuples]

    def dfs(self,
            weighted_prompts,
            n_ents,
            cur_ent_idx,
            cur_ent_tuple,
            cur_weight,
            collected_tuples_heap,
            n):
        if cur_ent_idx == n_ents:
            if len(collected_tuples_heap) < n:
                heapq.heappush(
                    collected_tuples_heap, [cur_weight, cur_ent_tuple])
            else:
                heapq.heappushpop(
                    collected_tuples_heap, [cur_weight, cur_ent_tuple])
            return

        mask_state = None
        for prompt, weight in weighted_prompts:
            input_text = self._model.get_masked_input_text(prompt=prompt)
            inputs = self._model.tokenizer(
                input_text, return_tensors="pt").to('cuda')

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

        for prob, pred_id in zip(probs, pred_ids):
            if len(collected_tuples_heap) == n and \
                    cur_weight * prob.item() < collected_tuples_heap[0][0]:
                break

            pred_ent = self._model.tokenizer.decode(pred_id)

            pred_ent = pred_ent.strip().lower()
            if pred_ent in cur_ent_tuple:
                continue

            if not any([ch.isalpha() for ch in pred_ent]):
                continue

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
                collected_tuples_heap=collected_tuples_heap,
                n=n)
