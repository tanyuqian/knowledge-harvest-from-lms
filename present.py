import fire
import json
import random
from prettytable import PrettyTable

from data_utils.concept_net import ConceptNet


def main(output_dir, n_present=20):
    conceptnet = ConceptNet()

    relation_info = json.load(open('data/relation_info.json'))

    for rel, info in relation_info.items():
        columns = {}

        columns['ConceptNet Samples'] = [
            str(ent_tuple) for ent_tuple in conceptnet.get_ent_tuples(
                rel=rel)[:n_present]]

        weighted_ent_tuples = json.load(open(
            f'{output_dir}/{rel}/weighted_ent_tuples.json'))[:500]

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

        print(f'Relation: {rel}')
        print('Init Prompts:')
        for prompt in info['init_prompts']:
            print('-', prompt)
        # print('Extracted Prompts:')
        # for prompt in info['prompts']:
        #     print('-', prompt)
        print('Harvested Tuples:')
        print(table)
        print('=' * 50)


if __name__ == '__main__':
    fire.Fire(main)