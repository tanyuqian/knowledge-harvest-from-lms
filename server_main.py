import os
import json

from flask import Flask, jsonify

from backend import MAX_N_ENT_TUPLES_LIST
from demo import get_rel


app = Flask(__name__)


def search(model_name, init_prompts_str, seed_ent_tuples_str):
    os.system(f'python backend.py'
              f' --model_name {model_name}'
              f' --init_prompts_str {init_prompts_str}'
              f' --seed_ent_tuples_str {seed_ent_tuples_str}'
              f' &')


def retrieve_results(model_name, init_prompts_str, seed_ent_tuples_str):
    prompts, ent_tuples = [], []

    init_prompts, seed_ent_tuples, rel = get_rel(
        init_prompts_str=init_prompts_str,
        seed_ent_tuples_str=seed_ent_tuples_str)

    if os.path.exists(f'results/demo/prompts/{rel}.json'):
        prompts = [[prompt, ''] for prompt in json.load(open(
            f'results/demo/prompts/{rel}.json'))]

    for max_n_ent_tuples in reversed(MAX_N_ENT_TUPLES_LIST):
        result_path = f'results/demo/{max_n_ent_tuples}tuples/{model_name}' \
                      f'/{rel}/prompts.json'
        if os.path.exists(result_path):
            prompts = json.load(open(result_path))
            for i in range(len(prompts)):
                prompts[i][1] = f'{prompts[i][1]:.2f}'
            break

    for max_n_ent_tuples in reversed(MAX_N_ENT_TUPLES_LIST):
        result_path = f'results/demo/{max_n_ent_tuples}tuples/{model_name}' \
                      f'/{rel}/ent_tuples.json'
        if os.path.exists(result_path) and json.load(open(result_path)) != []:
            ent_tuples = json.load(open(result_path))[:20]

            sum_weights = sum(t[1] for t in ent_tuples)
            for i in range(len(ent_tuples)):
                ent_tuples[i][1] = f'{ent_tuples[i][1] / sum_weights:.6f}'

            break

    return {'prompts': prompts, 'ent_tuples': ent_tuples}


@app.route('/predict/<model_name>/<init_prompts_str>/<seed_ent_tuples_str>')
def predict(model_name, init_prompts_str, seed_ent_tuples_str):
    search(
        model_name=model_name,
        init_prompts_str=init_prompts_str,
        seed_ent_tuples_str=seed_ent_tuples_str)

    return jsonify(retrieve_results(
        model_name=model_name,
        init_prompts_str=init_prompts_str,
        seed_ent_tuples_str=seed_ent_tuples_str))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1111, debug=True)