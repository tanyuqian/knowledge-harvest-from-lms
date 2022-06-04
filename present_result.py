import os
import sys
import fire
import json
import random
from prettytable import PrettyTable


def main(result_dir, n_present=20):
    rel_set = result_dir.split('/')[1]
    relation_info = json.load(open(f'data/relation_info_{rel_set}_5seeds.json'))

    summary_file = open(f'{result_dir}/summary.txt', 'w')

    for rel, info in relation_info.items():
        columns = {'Seed samples': info['seed_ent_tuples']}

        if not os.path.exists(f'{result_dir}/{rel}/ent_tuples.json'):
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue

        weighted_prompts = json.load(open(f'{result_dir}/{rel}/prompts.json'))
        weighted_ent_tuples = json.load(open(
            f'{result_dir}/{rel}/ent_tuples.json'))

        if len(weighted_ent_tuples):
            print(f'outputs of relation \"{rel}\" not found. skipped.')
            continue
        weighted_ent_tuples = \
            weighted_ent_tuples[:len(weighted_ent_tuples) // 2]

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

        def _print_results(output_file):
            print(f'Relation: {rel}', file=output_file)
            print('Prompts:', file=output_file)
            for prompt, weight in weighted_prompts:
                print(f'- {weight:.4f} {prompt}', file=output_file)
            print('Harvested Tuples:', file=output_file)
            print(table, file=output_file)
            print('=' * 50, file=output_file, flush=True)

        _print_results(output_file=summary_file)
        _print_results(output_file=sys.stdout)

    print(f'This summary has been saved into {summary_file.name}.')


if __name__ == '__main__':
    fire.Fire(main)
