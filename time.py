from demo import search_ent_tuples, get_rel
import os
import string
from data_utils.data_utils import fix_prompt_style, is_valid_prompt


init_prompts_str = "B is the location for A"
seed_ent_tuples_str = "flotation_device~boat^water~soft_drink^gear~car^giraffes~africa^trousers~suitcase"
init_prompts, seed_ent_tuples, rel = get_rel(
    init_prompts_str=init_prompts_str,
    seed_ent_tuples_str=seed_ent_tuples_str)

for i, prompt in enumerate(init_prompts):
    for ent_idx, ch in enumerate(string.ascii_uppercase):
        prompt = prompt.replace(ch, f'<ent{ent_idx}>')
    prompt = prompt.replace('_', ' ').replace('<ent', '<ENT')
    init_prompts[i] = fix_prompt_style(prompt=prompt)

model_name = "roberta-large"
max_n_ent_tuples = 100
output_dir = f'results/time/{max_n_ent_tuples}tuples/{model_name}'
os.makedirs(f'{output_dir}/{rel}', exist_ok=True)

prompts = [
        "The <ENT1> is where i put my <ENT0> .",
        "The <ENT1> is the best place to store your <ENT0> when you're traveling .",
        "The <ENT1> is where you would typically find <ENT0> .",
        "A <ENT0> is typically found on a <ENT1> .",
        "<ENT1> is the natural habitat for <ENT0> .",
        "<ENT0> is an ingredient in <ENT1>s .",
        "The <ENT1> is where the <ENT0> is located .",
        "<ENT0> are found in <ENT1> .",
        "A <ENT1> is a great location to keep a <ENT0> in case you need it .",
        "There may be <ENT0> in the <ENT1> .",
        "<ENT1> is the location for <ENT0> ."
    ]
weighted_ent_tuples = search_ent_tuples(
                init_prompts=init_prompts,
                seed_ent_tuples=seed_ent_tuples,
                prompts=prompts,
                model_name=model_name,
                max_n_ent_tuples=max_n_ent_tuples,
                result_dir=f'{output_dir}/{rel}/')
                
print(weighted_ent_tuples)