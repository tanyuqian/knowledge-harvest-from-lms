import fire
import json
import random
from prettytable import PrettyTable

from data_utils.concept_net import ConceptNet


def main(output_dir, n_present=20):
    conceptnet = ConceptNet()

    relation_info = json.load(open('data/relation_info_5seeds.json'))

    output_file = open(f'{output_dir}/summary.txt', 'w')

    for rel, info in relation_info.items():
        columns = {}

        columns['ConceptNet Samples'] = [
            str(ent_tuple) for ent_tuple in conceptnet.get_ent_tuples(
                rel=rel)[:n_present]]

        weighted_prompts = json.load(open(f'{output_dir}/{rel}/prompts.json'))

        weighted_ent_tuples = json.load(open(
            f'{output_dir}/{rel}/ent_tuples.json'))[:500]

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