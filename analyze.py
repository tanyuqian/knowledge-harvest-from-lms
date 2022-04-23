import fire
import os
import json
from glob import glob


def main():
    relation_info = json.load(open(
        f'data/relation_info_conceptnet_5seeds.json'))

    for output_dir in glob('results/outputs_*'):
        print(f'output_dir: {output_dir}:')

        for metric in ['ckbc', 'comet']:
            scores = []
            for rel in relation_info:
                # print(f'relation: {rel}')

                if not os.path.exists(f'{output_dir}/{rel}/scores.json'):
                    # print(f'{output_dir} doesn\'t have realtion {rel}.')
                    continue

                result = json.load(open(
                    f'{output_dir}/{rel}/scores.json'))[:100]
                if result == []:
                    # print(f'{output_dir} doesn\'t have realtion {rel}.')
                    continue

                for knowledge_term in result:
                    scores.append(knowledge_term[f'{metric} score'])

            if len(scores) == 0:
                continue

            print(f'{metric}: {sum(scores) / len(scores)}')
            if metric == 'ckbc':
                print(f'{metric} acc: '
                      f'{sum([int(t > 0.5) for t in scores]) / len(scores)}')

        print('=' * 50)


if __name__ == '__main__':
    fire.Fire(main)