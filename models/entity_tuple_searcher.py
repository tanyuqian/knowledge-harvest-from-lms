import torch
import heapq

from data_utils.data_utils import get_n_ents, get_mask_index_in_prompt, \
    stopwords


class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model

    def search(self, weighted_prompts, n):
        n_ents = get_n_ents(weighted_prompts[0][0])

        collected_tuples_heap = []

        for t in range(1 << n_ents):
            bin_t = f'{t:b}'
            bin_t = '0' * (n_ents - len(bin_t)) + bin_t

            n_masks = [int(ch) + 1 for ch in bin_t]

            self.dfs(
                weighted_prompts=weighted_prompts,
                n_ents=n_ents,
                n_masks=n_masks,
                cur_ent_tuple=[],
                cur_weight=1.,
                collected_tuples_heap=collected_tuples_heap,
                n=n)

        collected_tuples = sorted(collected_tuples_heap, reverse=True)

        for weight, ent_tuple in collected_tuples:
            print(ent_tuple, weight)
        print('=' * 50)

        return [t[1] for t in collected_tuples]

    def dfs(self,
            weighted_prompts,
            n_ents,
            n_masks,
            cur_ent_tuple,
            cur_weight,
            collected_tuples_heap,
            n):
        cur_ent_idx = len(cur_ent_tuple)

        if cur_ent_idx == n_ents:
            pred_ent_tuple = [cur_weight, cur_ent_tuple]
            if len(collected_tuples_heap) < n:
                heapq.heappush(collected_tuples_heap, pred_ent_tuple)
            else:
                heapq.heappushpop(
                    collected_tuples_heap, pred_ent_tuple)
            return

        collected_ents = []
        ent_weight_threshold = 0. if len(collected_tuples_heap) < n else \
            collected_tuples_heap[0][0] / cur_weight
        self.dfs_ent(
            cur_ent_tuple=cur_ent_tuple,
            n_masks=n_masks,
            weighted_prompts=weighted_prompts,
            cur_token_ids=[],
            cur_weight=1.,
            collected_ent_heap=collected_ents,
            weight_threashold=ent_weight_threshold,
            n=n)

        collected_ents.sort(reverse=True)

        for prob, pred_ent in collected_ents:
            if len(collected_tuples_heap) == n and \
                    cur_weight * prob < collected_tuples_heap[0][0]:
                break

            weighted_prompts_upd = []
            for prompt, weight in weighted_prompts:
                weighted_prompts_upd.append(
                    [prompt.replace(f'<ENT{cur_ent_idx}>', pred_ent), weight])

            self.dfs(
                weighted_prompts=weighted_prompts_upd,
                n_ents=n_ents,
                n_masks=n_masks,
                cur_ent_tuple=cur_ent_tuple + [pred_ent],
                cur_weight=cur_weight * prob.item(),
                collected_tuples_heap=collected_tuples_heap,
                n=n)

    def dfs_ent(self,
                cur_ent_tuple,
                n_masks,
                weighted_prompts,
                cur_token_ids,
                cur_weight,
                collected_ent_heap,
                weight_threashold,
                n):
        ent_idx = len(cur_ent_tuple)

        if len(cur_token_ids) == n_masks[ent_idx]:
            pred_ent = self._model.tokenizer.decode(cur_token_ids)

            pred_ent = pred_ent.strip().lower()
            if pred_ent in cur_ent_tuple or pred_ent in stopwords:
                return

            if not any([ch.isalpha() for ch in pred_ent]):
                return

            if len(collected_ent_heap) < n:
                heapq.heappush(collected_ent_heap, [cur_weight, pred_ent])
            else:
                heapq.heappushpop(collected_ent_heap, [cur_weight, pred_ent])

            return

        mask_state = None
        for prompt, weight in weighted_prompts:
            prompt = prompt.replace(
                f'<ENT{ent_idx}>',
                self._model.tokenizer.decode(cur_token_ids) +
                '<mask>' * (n_masks[ent_idx] - len(cur_token_ids)))

            input_text = self._model.get_masked_input_text(
                prompt=prompt, n_masks=n_masks)
            inputs = self._model.tokenizer(
                input_text, return_tensors="pt").to('cuda')

            with torch.no_grad():
                outputs = self._model.encoder(**inputs)

            sequence_output = outputs.last_hidden_state[
                inputs['input_ids'] == self._model.mask_token_id]

            mask_idx_in_prompt = get_mask_index_in_prompt(
                ent_idx=ent_idx, n_masks=n_masks, prompt=prompt)
            if mask_state is None:
                mask_state = torch.zeros_like(
                    sequence_output[mask_idx_in_prompt])
            mask_state = \
                mask_state + sequence_output[mask_idx_in_prompt] * weight

        mask_state = mask_state / sum(weight for _, weight in weighted_prompts)

        logits = self._model.lm_head(mask_state.reshape(1, -1))
        # logits[::, self._model.banned_ids] = -float('inf')
        probs = torch.softmax(logits, dim=-1)[0]
        probs, pred_ids = torch.sort(probs, descending=True)

        for prob, pred_id in zip(probs, pred_ids):
            if len(collected_ent_heap) == n and \
                    cur_weight * prob.item() < collected_ent_heap[0][0]:
                break

            if cur_weight * prob.item() < weight_threashold:
                break

            self.dfs_ent(
                cur_ent_tuple=cur_ent_tuple,
                n_masks=n_masks,
                weighted_prompts=weighted_prompts,
                cur_token_ids=cur_token_ids + [pred_id],
                cur_weight=cur_weight * prob.item(),
                collected_ent_heap=collected_ent_heap,
                weight_threashold=weight_threashold,
                n=n)