
import torch
import copy
from transformers import RobertaTokenizer, RobertaForMaskedLM
import random
import json
from models.knowledge_harvester import KnowledgeHarvester

def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]', '[Y]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
    # tokenizer.lama_x = '[X]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        # module.register_backward_hook(self.hook)
        module.register_full_backward_hook(self.hook)
    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]
    def get(self):
        return self._stored_gradient
class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model
    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        # logits, *_ = self._model(**model_inputs)
        logits = self._model(**model_inputs).logits
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits  # 42, 50276
def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out
# tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
# model = RobertaForMaskedLM.from_pretrained("roberta-large")

def get_n_ents(prompt):
    n = 0
    while f'<ENT{n}>' in prompt:
        n += 1
    return n

def get_sent(prompt, ent_tuple):
    sent = prompt
    for idx, ent in enumerate(ent_tuple):
        sent = sent.replace(f'<ENT{idx}>', ent)
    return sent

def get_loss_gradients(harvester, embedding_gradient, prompt, ent_with_lengths, grad=True, weights=(1, 1, 1)):
    gradients = 0
    losses = 0
    n_prompt = len(prompt)
    for n_ents, ents in ent_with_lengths.items():
        encodings = harvester._model._tokenizer('<mask> ' * n_ents[0] + " ".join(["[T]"] * n_prompt) + ' <mask>' * n_ents[1], return_tensors='pt').to("cuda")
        span = (encodings["input_ids"] == harvester._model._tokenizer.encode("[T]", add_special_tokens=False)[0]).to('cuda')
        encodings["input_ids"][span] = prompt # replace with the current prompt...
        
        for j in range(len(ents)):
            # print(ents[j][1])
            harvester._model._model.zero_grad()
            cur_masked_inputs = copy.deepcopy(encodings)
            scores = []
            for step in range(n_ents[0]):
                logits = harvester._model.model(**cur_masked_inputs).logits
                logprobs = torch.log_softmax(logits, dim=-1)[0]
                scores.append(
                    logprobs[1 + step][ents[j][0][0][step]])
                # print(scores[-1])
                new_masked_inputs = copy.deepcopy(cur_masked_inputs)
                cur_masked_inputs = new_masked_inputs
                cur_masked_inputs["input_ids"][0][1 + step] = ents[j][0][0][step]
                
            for step in range(n_ents[1]):
                logits = harvester._model.model(**cur_masked_inputs).logits
                logprobs = torch.log_softmax(logits, dim=-1)[0]
                scores.append(
                    logprobs[-n_ents[1] + step - 1][ents[j][0][1][step]])
                # print(scores[-1])
                new_masked_inputs = copy.deepcopy(cur_masked_inputs)
                cur_masked_inputs = new_masked_inputs
                cur_masked_inputs["input_ids"][0][-n_ents[1] + step - 1] = ents[j][0][1][step]
                
            sum_score = sum(scores) / 2
            mean_score = sum(scores) / len(scores)
            min_score = min(scores)
            loss = - (sum_score * weights[0] + mean_score * weights[1] + min_score * weights[2])
            losses += loss.item()
            if grad:
                loss.backward()
                g = embedding_gradient.get()
                cur_gradient = g.masked_select(span.unsqueeze(-1).expand((-1, -1, 1024))).reshape(-1, 1024)
                gradients += cur_gradient
    harvester._model._model.zero_grad()
    return losses, gradients
    
def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)
    return top_k_ids

def main():

    n_prompt = 5  # "<s> <ENT0> <T> <T> <T> <T> <T> <ENT1> </s>""
    patience = 20
    num_candidates = 20
    seed_file = "data/relation_info_conceptnet_5seeds.json"
    rel_info = json.load(open(seed_file, 'r'))

    for rel, info in rel_info.items():
        tuples = info["seed_ent_tuples"]
        tuples = [list(t) for t in tuples]
        harvester = KnowledgeHarvester("roberta-large")
        harvester._model._model.eval()
        harvester._model._model.zero_grad()
        embeddings = harvester._model._model.roberta.embeddings.word_embeddings
        add_task_specific_tokens(harvester._model._tokenizer)
        embedding_gradient = GradientStorage(embeddings)
        ent_with_lengths = harvester._model.tokenize_tuples_by_len(tuples, not_at_beginning=[False, True])
        harvester._model._model = harvester._model._model.to("cuda")
        prompt = harvester._model._tokenizer.encode("<mask>" * n_prompt, add_special_tokens=False, return_tensors='pt')[0].to("cuda")
        no_improvement = 0
        for _ in range(200):
            if no_improvement > patience:
                break
            token_to_flip = random.randrange(n_prompt)
            current_score, gradients = get_loss_gradients(harvester, embedding_gradient, prompt, ent_with_lengths)
            if prompt.eq(harvester._model._tokenizer.mask_token_id).any():
                current_score = float('inf')
            print("current prompt: ", harvester._model._tokenizer.decode(prompt), current_score)
            candidates = hotflip_attack(gradients[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=num_candidates,
                                    filter=None)
            candidate_scores = torch.zeros(num_candidates,device='cuda')
            cand_prompt = copy.deepcopy(prompt)
            for i, c in enumerate(candidates):
                cand_prompt[token_to_flip] = c
                with torch.no_grad():
                    candidate_scores[i], grad = get_loss_gradients(harvester, embedding_gradient, cand_prompt, ent_with_lengths, grad=False)
            print(candidate_scores)
            if (candidate_scores < current_score).any():
                no_improvement = 0
                best_candidate_score = candidate_scores.min()
                best_candidate_idx = candidate_scores.argmin()
                prompt[token_to_flip] = candidates[best_candidate_idx.item()]
                print('Better trigger detected.', harvester._model._tokenizer.decode(prompt), candidate_scores[best_candidate_idx.item()])
            else:
                no_improvement += 1
                print("Not found...")
                
        with open("data/autoprompt_concept.txt", 'a') as result_file:
            result_file.write(json.dumps({rel: "<ENT0>"+harvester._model._tokenizer.decode(prompt)+" <ENT1>"}) + "\n")

main()