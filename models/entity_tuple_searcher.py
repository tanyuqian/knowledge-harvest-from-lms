import torch
import heapq
import time

from data_utils.data_utils import get_n_ents, get_mask_index_in_prompt, \
    stopwords


class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model

    def search(self, weighted_prompts, n):
        n_ents = get_n_ents(weighted_prompts[0][0])

        start = time.time()

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
                cur_logprobs=[],
                collected_tuples_heap=collected_tuples_heap,
                n=n)

        collected_tuples = sorted(collected_tuples_heap, reverse=True)

        print(f"searched for entity tuples in {time.time() - start} s", )

        # for weight, ent_tuple in collected_tuples:
        #     print(ent_tuple, weight)
        # print('=' * 50)

        return [t[1] for t in collected_tuples]

    def dfs(self,
            weighted_prompts,
            n_ents,
            n_masks,
            cur_ent_tuple,
            cur_logprobs,
            collected_tuples_heap,
            n):
        cur_ent_idx = len(cur_ent_tuple)

        if cur_ent_idx == n_ents:
            pred_ent_tuple = [min(cur_logprobs), cur_ent_tuple]
            if len(collected_tuples_heap) < n:
                heapq.heappush(collected_tuples_heap, pred_ent_tuple)
                # this is the initial case?

            else:
                heapq.heappushpop(
                    collected_tuples_heap, pred_ent_tuple)
            return

        collected_ents = []
        logprob_threshold = float('-inf') if len(collected_tuples_heap) < n \
            else collected_tuples_heap[0][0]

        self.dfs_ent(
            cur_ent_tuple=cur_ent_tuple,
            n_masks=n_masks,
            weighted_prompts=weighted_prompts,
            cur_token_ids=[],
            cur_logprobs=[],
            collected_ent_heap=collected_ents,
            logprob_threashold=logprob_threshold,
            n=n)

        collected_ents.sort(reverse=True)

        # for prob, pred_ent in collected_ents:
        #     print(pred_ent, prob)

        for ent_min_logprob, _, pred_ent in collected_ents:
            min_upd = min(cur_logprobs + [ent_min_logprob])
            if len(collected_tuples_heap) == n and \
                    min_upd < collected_tuples_heap[0][0]:
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
                cur_logprobs=cur_logprobs + [ent_min_logprob],
                collected_tuples_heap=collected_tuples_heap,
                n=n)

    def dfs_ent(self,
                cur_ent_tuple,
                n_masks,
                weighted_prompts,
                cur_token_ids,
                cur_logprobs,
                collected_ent_heap,
                logprob_threashold,
                n):
        ent_idx = len(cur_ent_tuple)

        if len(cur_token_ids) == n_masks[ent_idx]:
            pred_ent = self._model.tokenizer.decode(cur_token_ids)

            pred_ent = pred_ent.strip().lower()
            if pred_ent in cur_ent_tuple or pred_ent in stopwords:
                return

            if len(collected_ent_heap) < n:
                heapq.heappush(collected_ent_heap, [
                    min(cur_logprobs), cur_logprobs, pred_ent])
            else:
                heapq.heappushpop(collected_ent_heap, [
                    min(cur_logprobs), cur_logprobs, pred_ent])

            return

        mask_state = None
        for raw_prompt, weight in weighted_prompts:
            prompt = raw_prompt.replace(
                f'<ENT{ent_idx}>',
                self._model.tokenizer.decode(cur_token_ids) +
                '<mask>' * (n_masks[ent_idx] - len(cur_token_ids)))
            # a small bug?: 'It is typical for <mask> to cause <ENT1>'

            input_text = self._model.get_masked_input_text(
                prompt=prompt, n_masks=n_masks)
            
            inputs = self._model.tokenizer(
                input_text, return_tensors="pt").to('cuda')  # single sentence

            with torch.no_grad():
                outputs = self._model.encoder(**inputs)

            sequence_output = outputs.last_hidden_state[
                inputs['input_ids'] == self._model.mask_token_id]
            # n_mask (2) * embedding_dim (1024)
            mask_idx_in_prompt = get_mask_index_in_prompt(
                ent_idx=ent_idx, n_masks=n_masks, prompt=raw_prompt)
            if mask_state is None:
                mask_state = torch.zeros_like(
                    sequence_output[mask_idx_in_prompt])
            mask_state = \
                mask_state + sequence_output[mask_idx_in_prompt] * weight

        mask_state = mask_state / sum(weight for _, weight in weighted_prompts)

        logits = self._model.lm_head(mask_state.reshape(1, -1))
        logits[::, self._model.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(logits, dim=-1)[0]
        logprobs, pred_ids = torch.sort(logprobs, descending=True)

        for logprob, pred_id in zip(logprobs, pred_ids):
            min_upd = min(cur_logprobs + [logprob.item()])
            if len(collected_ent_heap) == n and \
                    min_upd < collected_ent_heap[0][0]:
                break

            if min_upd < logprob_threashold:
                break

            if not any([ch.isalpha() for ch in
                        self._model.tokenizer.decode(pred_id)]):
                continue

            self.dfs_ent(
                cur_ent_tuple=cur_ent_tuple,
                n_masks=n_masks,
                weighted_prompts=weighted_prompts,
                cur_token_ids=cur_token_ids + [pred_id],
                cur_logprobs=cur_logprobs + [logprob.item()],
                collected_ent_heap=collected_ent_heap,
                logprob_threashold=logprob_threashold,
                n=n)