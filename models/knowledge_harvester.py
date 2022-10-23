from collections import defaultdict
from tqdm import tqdm
from scipy.special import softmax

from models.language_model_wrapper import LanguageModelWrapper
from models.entity_tuple_searcher import EntityTupleSearcher

from data_utils.data_utils import fix_prompt_style, is_valid_prompt, get_n_ents
import copy
import timeit
import torch
import random
from tqdm import *

class KnowledgeHarvester:
    def __init__(self,
                 model_name,
                 max_n_prompts=20,
                 max_n_ent_tuples=10000,
                 max_word_repeat=5,
                 max_ent_subwords=1,
                 prompt_temp=1.):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._max_n_prompts = max_n_prompts
        self._max_n_ent_tuples = max_n_ent_tuples
        self._max_word_repeat = max_word_repeat
        self._max_ent_subwords = max_ent_subwords
        self._prompt_temp = prompt_temp
        self._model = LanguageModelWrapper(model_name=model_name)
        self._ent_tuple_searcher = EntityTupleSearcher(model=self._model)

        self._seed_ent_tuples = None
        self._max_batch_size = 128
        self._beam_size = max(128, max_n_ent_tuples * 2)

    def clear(self):
        self._weighted_prompts = []
        self._weighted_ent_tuples = []
        self._seed_ent_tuples = None

    def set_seed_ent_tuples(self, seed_ent_tuples):
        self._seed_ent_tuples = seed_ent_tuples

    def set_prompts(self, prompts):
        for prompt in prompts:
            if is_valid_prompt(prompt=prompt):
                self._weighted_prompts.append([fix_prompt_style(prompt), 1.])

    '''
    def update_prompts(self):
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            pos_scores, neg_scores = [], []
            for ent_tuple in self._seed_ent_tuples:
                ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

                pos_scores.append(self.score(
                    prompt=prompt, ent_tuple=ent_tuple))

                for ent_idx in range(len(ent_tuple)):
                    for ent_tuple1 in self._seed_ent_tuples:
                        if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                            continue

                        ent_tuple_neg = \
                            ent_tuple[:ent_idx] + \
                            [ent_tuple1[ent_idx]] + \
                            ent_tuple[ent_idx + 1:]

                        neg_scores.append(self.score(
                            prompt=prompt, ent_tuple=ent_tuple_neg))

            pos_score = sum(pos_scores) / len(pos_scores)
            neg_score = sum(neg_scores) / len(neg_scores)

            self._weighted_prompts[i][1] = \
                (pos_score - 0.5 * neg_score) / self._prompt_temp

        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.
        norm_weights /= norm_weights.sum()

        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight
        self._weighted_prompts = [
            t for t in self._weighted_prompts if t[1] > 1e-4]
    '''

    def update_prompts(self):
        pos_tuples, neg_tuples = [], []

        for ent_tuple in self._seed_ent_tuples:
            ent_tuple = [ent.replace('_', ' ') for ent in ent_tuple]

            pos_tuples.append(ent_tuple)

            for ent_idx in range(len(ent_tuple)):
                for ent_tuple1 in self._seed_ent_tuples:
                    if ent_tuple1[ent_idx] == ent_tuple[ent_idx]:
                        continue

                    neg_tuples.append(
                        ent_tuple[:ent_idx] + \
                        [ent_tuple1[ent_idx]] + \
                        ent_tuple[ent_idx + 1:]
                    )
        neg_tuples = random.sample(neg_tuples, 2 * len(pos_tuples))
        # reduce the number of neg samples
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            # print(i)
            pos_scores_with_tuples = self.get_ent_tuples_weight(pos_tuples, given_prompts=[[prompt, 1.]])
            neg_scores_with_tuples = self.get_ent_tuples_weight(neg_tuples, given_prompts=[[prompt, 1.]])
            pos_score = sum([score for tup, score in pos_scores_with_tuples]) / len(pos_scores_with_tuples)
            neg_score = sum([score for tup, score in neg_scores_with_tuples]) / len(neg_scores_with_tuples)

            self._weighted_prompts[i][1] = \
                (pos_score - 0.5 * neg_score) / self._prompt_temp
        '''
        pos_scores = self.get_ent_tuples_weight(pos_tuples, return_details=True)
        # print(pos_scores.shape)
        neg_scores = self.get_ent_tuples_weight(neg_tuples, return_details=True)
        # print(neg_scores.shape)
        for i, (prompt, _) in enumerate(self._weighted_prompts):
            self._weighted_prompts[i][1] = \
                    (pos_scores.mean(-1)[i] - 0.5 * neg_scores.mean(-1)[i]) / self._prompt_temp
            self._weighted_prompts[i][1] = float(self._weighted_prompts[i][1])
        '''
        self._weighted_prompts = sorted(
            self._weighted_prompts,
            key=lambda t: t[1], reverse=True)[:self._max_n_prompts]

        norm_weights = softmax([weight for _, weight in self._weighted_prompts])
        norm_weights[norm_weights < 0.05] = 0.
        norm_weights /= norm_weights.sum()

        for i, norm_weight in enumerate(norm_weights):
            self._weighted_prompts[i][1] = norm_weight
        self._weighted_prompts = [
            t for t in self._weighted_prompts if t[1] > 1e-4]

        print(self._weighted_prompts)

    def update_ent_tuples(self):
        start = timeit.default_timer()
        
        '''
        ent_tuples = self._ent_tuple_searcher.search(
            weighted_prompts=self._weighted_prompts,
            n=self._max_n_ent_tuples,
            max_word_repeat=self._max_word_repeat,
            max_ent_subwords=self._max_ent_subwords)
        '''
        
        n_ents = get_n_ents(self._weighted_prompts[0][0])
        raw_tuples = []
        ents = [[] for i in range(n_ents)]
        for t in range(1 << n_ents):
            bin_code = f'{t:b}'
            bin_code = '0' * (n_ents - len(bin_code)) + bin_code
            num_tokens = [int(i) + 1 for i in bin_code]
            print(num_tokens)
            raw_tuples = self.beam_search_entity_pairs(num_tokens) # 1 or 2 tokens
            # print(raw_tuples)
            cnt = 0
            for ent_idx in range(n_ents):
                batch_decodings = self._model.tokenizer.batch_decode([tup[cnt: cnt + num_tokens[ent_idx]] for tup, score in raw_tuples])
                ents[ent_idx] += [d.strip() for d in batch_decodings]
                cnt += num_tokens[ent_idx]
                # print(ent_idx, ents[-5:])
            # print(ents)
            # exit(0)
        # raw_tuples = self.beam_search_entity_pairs((1, 1, 1))
        

        # remove repetitve tuples
        ent_tuples = []
        for tup in zip(*ents):
            if tup[0] == tup[1] or \
                (n_ents == 3 and (tup[0] == tup[2] or tup[1] == tup[2])):
                continue
            # ent_tuples.append([ent.strip() for ent in tup])
            ent_tuples.append([ent for ent in tup])
        # print(res)
        print(ent_tuples)
        cur = timeit.default_timer()
        print(f"searched entity pairs ({self._max_n_ent_tuples}): {cur-start}s")
        start = cur

        self._weighted_ent_tuples = []
        '''
        for ent_tuple in tqdm(ent_tuples, desc='re-scoring ent_tuples'):
            best_ent_tuple = None
            best_score = float('-inf')
            for t in range(1 << len(ent_tuple)):
                bin_code = f'{t:b}'
                bin_code = '0' * (len(ent_tuple) - len(bin_code)) + bin_code

                coded_ent_tuple = []
                for b, ent in zip(bin_code, ent_tuple):
                    coded_ent_tuple.append(ent.title() if b == '1' else ent)

                score = self.score_ent_tuple(ent_tuple=coded_ent_tuple)
                if score > best_score:
                    best_score = score
                    best_ent_tuple = coded_ent_tuple

            self._weighted_ent_tuples.append([best_ent_tuple, best_score])
        '''
        self._weighted_ent_tuples = self.get_ent_tuples_weight(ent_tuples)
        cur = timeit.default_timer()
        print(f"ranked entity pairs ({self._max_n_ent_tuples}): {cur - start}s")
        start = cur

        self._weighted_ent_tuples = sorted(
            self._weighted_ent_tuples, key=lambda t: t[1], reverse=True)[:self._max_n_ent_tuples * (1 << n_ents)]

        norm_weights = softmax(
            [weight for _, weight in self._weighted_ent_tuples])
        for i, norm_weight in enumerate(norm_weights):
            self._weighted_ent_tuples[i][1] = norm_weight
        
    def score_ent_tuple(self, ent_tuple):
        score = 0.
        for prompt, weight in self.weighted_prompts:
            score += weight * self.score(prompt=prompt, ent_tuple=ent_tuple)

        return score

    def score(self, prompt, ent_tuple):
        logprobs = self._model.get_mask_filling_logprobs(
            prompt=prompt, ent_tuple=ent_tuple)['mask_logprobs']

        token_wise_score = sum(logprobs) / len(logprobs)
        ent_wise_score = sum(logprobs) / len(ent_tuple)
        min_score = min(logprobs)

        return (token_wise_score + ent_wise_score + min_score) / 3.


    def get_ent_tuples_weight(self, ent_tuples, metric_weights=(1/3, 1/3, 1/3), given_prompts=None, return_details=False):
        begin = timeit.timeit()
        scores, tuples = self._score_tuples_prompts(ent_tuples, given_prompts=given_prompts)
        # now the score has the shape (n_prompts, n_tuples, 3)
        metric_weights = torch.tensor(metric_weights)
        scores = torch.sum(scores * metric_weights.reshape(1, 1, *metric_weights.shape)\
            .expand(*scores.shape), dim=-1)
        # aggregate all the metrics. Now (n_prompts, n_tuples)
        if return_details:
            return scores
        weighted_prompts = given_prompts if given_prompts is not None else self._weighted_prompts
        weights = torch.tensor([weight for prompt, weight in weighted_prompts])
        scores = torch.sum(scores * weights.reshape(*weights.shape, 1).expand(*scores.shape), dim=0)
        scores_with_tuples = [[ent_tuple, score.item()] for score, ent_tuple in zip(scores, tuples)]
        return scores_with_tuples

    def _score_tuples_prompts(self, ent_tuples, given_prompts=None):
        result_list = []
        # tuples_list = []
        n_ents = get_n_ents(self._weighted_prompts[0][0])
        # tuples_list = sorted(ent_tuples) # must be in the same order... (alphabet)
        weighted_prompts = self._weighted_prompts if given_prompts == None else given_prompts
        for prompt, weight in weighted_prompts:
            prompt_result_list = []
            not_at_beginning = [prompt.strip().find(f'<ENT{i}>') != 0 for i in range(n_ents)]
            tokenized_entity_pairs = self._model.tokenize_tuples_by_len(ent_tuples, not_at_beginning)
            # {(len_ent_1, len_ent_2, ...): [(ent_1_ids, ent_2_ids, ...), (ent_1_text, ent_2_text, ...), ...]}
            for n_masks, tuples in tokenized_entity_pairs.items():
                # print(f"scoring tuples of length {n_masks}... {len(tuples)} in total.")
                scores = self.score_tuples(tuples, prompt, n_masks=n_masks)
                torch.cuda.empty_cache()

                for score, ent_tuple in zip(scores, tuples):
                    # print(f"score and tuple: {score}, {ent_tuple}")
                    prompt_result_list.append((ent_tuple[1], (sum(score)/sum(n_masks), sum(score)/len(n_masks), min(score))))
                     # += [(ent_tuple[1], score) ]
            prompt_result_list = sorted(prompt_result_list, key=lambda x:x[0])
            result_list.append([result[1] for result in prompt_result_list])

        tuples_list = [result[0] for result in prompt_result_list]
        return torch.tensor(result_list), tuples_list




    def score_tuples(self, tuples, prompt, n_masks):
        batch_size = min(self._max_batch_size, len(tuples))
        n_ents = len(n_masks)
        batch_prompt, pos_entities = self._model._tokenize_prompt_with_slots(prompt, n_masks, batch_size)
        return_scores = []
        print(f"scoring with prompt {prompt}")
        for i in trange((len(tuples) - 1)//batch_size + 1):
            tuples_batch = tuples[i * batch_size: (i + 1) * batch_size]
            cur_batch_size = len(tuples_batch)
            batch_ids = copy.deepcopy(batch_prompt["input_ids"])[:cur_batch_size]
            batch_scores = [[] for _ in range(cur_batch_size)]
            for ent_idx in range(n_ents):
                for token in range(n_masks[ent_idx]):
                    target_pos = pos_entities[ent_idx] + token
                    target_ids = [tuples_batch[case_idx][0][ent_idx][token] for case_idx in range(cur_batch_size)]

                    cur_scores = self._model._get_batch_prediction(batch_ids[:cur_batch_size], \
                        batch_prompt.get("token_type_ids", [])[:cur_batch_size], \
                        batch_prompt["attention_mask"][:cur_batch_size], target_pos)  
                        # roberta doesn't have "token_type_ids"

                    for case_idx in range(cur_batch_size):
                        batch_scores[case_idx].append(cur_scores[case_idx, target_ids[case_idx]].item())
                        batch_ids[case_idx][target_pos] = target_ids[case_idx]  
                        # fill in the blank for next token prediction

            return_scores += batch_scores
            # print("hey", return_scores)
            # exit(0)
        return return_scores


    def beam_search_entity_pairs(self, n_masks):
        # print(f"Searching for entity pairs in {n_masks}")
        prompts_tokens = []
        for prompt, weight in self._weighted_prompts:
            #TODO: set a hyperparameter
            batch_prompts, pos_entities = self._model._tokenize_prompt_with_slots(prompt,\
                n_masks, self._max_batch_size)
            # print(batch_prompts, pos_entities)
            # assert len(pos_h) == H and len(pos_t) == T, (prompt, inputs, pos_h, pos_t)
            prompts_tokens.append((batch_prompts, pos_entities))

        paths_with_scores = [[[], 0]]
        new_paths_with_scores = []
        
        for step in range(sum(n_masks)):
            
            print("step: ", step)
            for i in trange((len(paths_with_scores) - 1)//self._max_batch_size + 1):
                # print("batch: ", i)
                batch = paths_with_scores[i * self._max_batch_size: (i + 1) * self._max_batch_size]
                # print(batch)
                # print(prompts_tokens)
                res = self.batch_beam_step([p for p, s in batch], \
                    torch.tensor([s for p, s in batch]).to("cuda"), step, prompts_tokens, n_masks)
                #  print(res)
                new_paths_with_scores += res
            new_paths_with_scores = sorted(new_paths_with_scores, key=lambda x: -x[1])[:self._beam_size]
            
            paths_with_scores = new_paths_with_scores
            new_paths_with_scores = []


        '''
        head_to_eval, tail_to_eval = [], []
        for pair, scores in triples.items():
            if scores[1] == 0:
                head_to_eval.append(pair[0])
            if scores[0] == 0:
                tail_to_eval.append(pair[1])
        '''

        '''
        for p, s in paths_with_scores:
            tail_scores[tuple(p[H:])] = s - head_scores[tuple(p[:H])]
        '''

        return [(p, s) for p, s in paths_with_scores]

    def batch_beam_step(self, batch_paths, batch_prev_scores, step, prompts_tokens, n_masks, reverse=False):
        # print(batch_paths)
        cur_bs = len(batch_paths)
        batch_prompt_score = []
        # last_step = sum(prompts_tokens[0][1]) == step + 1  # to filter repetitive tokens in the future.
        # print("cur", prompts_tokens)
        for batch_prompts, pos_entities in prompts_tokens:
            # expand the pos_entities.
            # cnt = 0
            expanded_pos = []
            for i in range(len(n_masks)):
                cnt = 0
                for j in range(n_masks[i]):
                    expanded_pos.append(pos_entities[i] + j)
            
            batch_ids = copy.deepcopy(batch_prompts["input_ids"])[:cur_bs]
            for k in range(cur_bs):
                for l in range(step):
                    if reverse:
                        batch_ids[k][(expanded_pos)[l]] = batch_paths[k][l]
                    else:
                        batch_ids[k][(expanded_pos)[l]] = batch_paths[k][l]
                # print(tokenizer.decode(batch_ids[k]))
            log_probs = self._model._get_batch_prediction(batch_ids, \
                batch_prompts.get('token_type_ids', [])[:cur_bs], batch_prompts['attention_mask'][:cur_bs],\
                     expanded_pos[step])
            # print(log_probs)
            # print(log_probs[:, :10])  # bs, vocab
            batch_prompt_score.append(log_probs.view(1, cur_bs, -1))
        # print(batch_prev_scores.shape) # bs
        vocab = log_probs.shape[-1]
        prev_scores = batch_prev_scores.view(-1, 1).expand(-1, vocab) # bs, vocab
        batch_prompt_score = torch.vstack(batch_prompt_score) # n_prompt * bs * vocab
        batch_prompt_score[:, :, self._model._banned_ids] -= 100
        batch_prompt_score = torch.sum(batch_prompt_score, 0)  / len(prompts_tokens)# bs * vocab
        new_batch_scores = prev_scores + batch_prompt_score
        sorted_args = torch.argsort(new_batch_scores.reshape(-1), descending=True)[:self._beam_size]  # bs * vocab | cut off beam_size
        x = torch.div(sorted_args, vocab, rounding_mode='trunc')
        y = torch.remainder(sorted_args, vocab)
        # print(x, y)
        return [(batch_paths[i] + [j.item()], new_batch_scores[i, j].item()) for i, j in zip(x, y)]

    @property
    def weighted_ent_tuples(self):
        return self._weighted_ent_tuples

    @property
    def weighted_prompts(self):
        return self._weighted_prompts