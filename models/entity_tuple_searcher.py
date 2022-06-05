import string
import torch
import heapq

from data_utils.data_utils import get_n_ents, get_mask_place, \
    get_masked_prompt, get_n_masks, stopwords


class EntityTupleSearcher:
    def __init__(self, model):
        self._model = model

    def search(self, weighted_prompts, max_word_repeat, max_ent_subwords, n):
        n_ents = get_n_ents(weighted_prompts[0][0])

        collected_tuples_heap = []
        repeat_cnt = {}

        for t in range(max_ent_subwords ** n_ents):
            n_masks = get_n_masks(
                t=t, n_ents=n_ents, max_ent_subwords=max_ent_subwords)
            print(f'searching with n_masks={n_masks}')

            self.dfs(
                weighted_prompts=weighted_prompts,
                n_ents=n_ents,
                n_masks=n_masks,
                cur_ent_tuple=[],
                cur_logprobs=[],
                collected_tuples_heap=collected_tuples_heap,
                repeat_cnt=repeat_cnt,
                max_word_repeat=max_word_repeat,
                n=n)

        ent_tuples = sorted([t[1] for t in collected_tuples_heap])

        ent_tuples = [ent_tuples[i] for i in range(len(ent_tuples))
                      if i == 0 or ent_tuples[i] != ent_tuples[i - 1]]

        return ent_tuples

    def dfs(self,
            weighted_prompts,
            n_ents,
            n_masks,
            cur_ent_tuple,
            cur_logprobs,
            collected_tuples_heap,
            repeat_cnt,
            max_word_repeat,
            n):
        cur_ent_idx = len(cur_ent_tuple)

        if cur_ent_idx == n_ents:
            pred = [min(cur_logprobs), cur_ent_tuple]

            for ent in cur_ent_tuple:
                for word in ent.split():
                    if repeat_cnt.get(word, 0) + 1 > max_word_repeat:
                        return

            heapq.heappush(collected_tuples_heap, pred)
            for ent in cur_ent_tuple:
                for word in ent.split():
                    repeat_cnt[word] = repeat_cnt.get(word, 0) + 1

            while len(collected_tuples_heap) > n:
                heap_top = heapq.heappop(collected_tuples_heap)
                for ent in heap_top[1]:
                    for word in ent.split():
                        repeat_cnt[word] = repeat_cnt[word] - 1

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
            n=n if len(cur_ent_tuple) == 0 else max_word_repeat)

        collected_ents.sort(reverse=True)

        for ent_min_logprob, pred_ent in collected_ents:
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
                repeat_cnt=repeat_cnt,
                max_word_repeat=max_word_repeat,
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
            # filter "the xxx", "new xxx", etc.
            if any([word in stopwords for word in pred_ent.split()]):
                return

            # filter entity with less than 3 characters
            if len(pred_ent.replace(' ', '')) <= 2:
                return

            # filter entity with single-character words
            if min([len(t) for t in pred_ent.split()]) <= 1:
                return

            for ent in cur_ent_tuple:
                # filter repeating entity in the entity tuple
                if pred_ent.replace(' ', '') == ent.replace(' ', ''):
                    return
                # filter repeating entity in the entity tuple
                if ent in pred_ent or pred_ent in ent:
                    return

            # filter entity appearing in the prompt
            for raw_prompt, _ in weighted_prompts:
                if pred_ent in raw_prompt:
                    return

            heapq.heappush(collected_ent_heap, [min(cur_logprobs), pred_ent])
            while len(collected_ent_heap) > n:
                heapq.heappop(collected_ent_heap)

            return

        mask_logits_total = None
        for raw_prompt, weight in weighted_prompts:
            prompt = raw_prompt.replace(
                f'<ENT{ent_idx}>',
                self._model.tokenizer.decode(cur_token_ids) +
                self._model.tokenizer.mask_token * (
                        n_masks[ent_idx] - len(cur_token_ids)))

            input_text = get_masked_prompt(
                prompt=prompt, n_masks=n_masks,
                mask_token=self._model.tokenizer.mask_token)
            mask_logits = self._model.get_mask_logits(input_text=input_text)

            mask_idx_in_prompt = get_mask_place(
                ent_idx=ent_idx, n_masks=n_masks, prompt=raw_prompt)
            mask_logits = mask_logits[mask_idx_in_prompt]

            if mask_logits_total is None:
                mask_logits_total = torch.zeros_like(mask_logits)
            mask_logits_total = mask_logits_total + mask_logits * weight

        mask_logits_total = mask_logits_total / sum(
            weight for _, weight in weighted_prompts)

        mask_logits_total[self._model.banned_ids] = -float('inf')
        logprobs = torch.log_softmax(mask_logits_total, dim=-1)
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

            if any([punc in self._model.tokenizer.decode(pred_id)
                    for punc in string.punctuation]):
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
