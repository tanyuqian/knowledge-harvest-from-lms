import os
import fire
import json
import random
from prettytable import PrettyTable

from data_utils.concept_net import ConceptNet
from data_utils.lama import LAMA


def get_samples(rel_set, rel, conceptnet, lama, n_samples):
    if rel_set == 'conceptnet':
        ent_tuples = conceptnet.get_ent_tuples(rel=rel)
    else:
        ent_tuples = lama.info[rel]['ent_tuples']

    return random.sample(ent_tuples, min(len(ent_tuples), n_samples))


def main(output_dir, n_present=20):
    rel_set = output_dir.split('/')[1]

    # conceptnet = ConceptNet() if rel_set == 'conceptnet' else None
    # lama = LAMA() if rel_set == 'lama' else None

    relation_info = json.load(open(f'data/relation_info_{rel_set}_5seeds.json'))

    output_file = open(f'{output_dir}/summary.txt', 'w')

    for rel, info in relation_info.items():
        columns = {}

        columns[f'Seed samples'] = info['seed_ent_tuples']

        if not os.path.exists(f'{output_dir}/{rel}/ent_tuples.json'):
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue

        weighted_prompts = json.load(open(f'{output_dir}/{rel}/prompts.json'))
        weighted_ent_tuples = json.load(open(
            f'{output_dir}/{rel}/ent_tuples.json'))[:500]

        if weighted_ent_tuples == []:
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue


        columns[f'Ours (Top {n_present})'] = [
            str(ent_tuple) for ent_tuple, _ in weighted_ent_tuples[:n_present]]

        columns[f'Ours (Random Sample)'] = [
            str(ent_tuple) for ent_tuple, _ in random.sample(
                weighted_ent_tuples, n_present)]

        columns[f'Ours (Tail {n_present})'] = [
            str(ent_tuple) for ent_tuple, _ in weighted_ent_tuples[-n_present:]]

        table = PrettyTable()
        for key, col in columns.items():
            if len(col) < n_present:
                col.extend(['\\'] * (n_present - len(col)))
            table.add_column(key, col)

        print(f'Relation: {rel}', file=output_file)
        print('Prompts:', file=output_file)
        for prompt, weight in weighted_prompts:
            print(f'- {weight:.4f} {prompt}', file=output_file)
        print('Harvested Tuples:', file=output_file)
        print(table, file=output_file)
        print('=' * 50, file=output_file, flush=True)


if __name__ == '__main__':
    fire.Fire(main)