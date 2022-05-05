import fire
import os
import json
from glob import glob

from prettytable import PrettyTable


N_TOP = 100
# RESULT_DIRS = {
#     'init prompts': '1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0_initprompts',
#     'best 1 prompt': '1000tuples_1prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0',
#     'all prompts (temp=1.)': '1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0',
#     'all prompts (temp=2.)': '1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp2.0',
#     'all prompts (temp=4.)': '1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp4.0'
# }


def main(rel_set='conceptnet'):
    relation_info = json.load(open(
        f'data/relation_info_{rel_set}_5seeds.json'))

    for setting_dir in glob(f'outputs/{rel_set}/*'):
        setting = setting_dir.split('/')[-1]

        print(f'Setting: {setting}')

        result = PrettyTable()
        result.add_column('Relations', list(relation_info.keys()) + ['Total'])

        for output_dir in glob(f'{setting_dir}/*'):
            model_name = output_dir.split('/')[-1]
            print(f'output_dir: {output_dir}:')

            col = []
            col_scores = []
            for rel in relation_info:
                if not os.path.exists(f'{output_dir}/{rel}/scores.json'):
                    col.append('//')
                    continue

                scores = json.load(open(f'{output_dir}/{rel}/scores.json'))[:N_TOP]
                if scores == []:
                    col.append('//')
                    continue

                scores = [t['cls score'] for t in scores]
                col.append(f'{sum(scores) / len(scores):.4f}')
                col_scores.append(sum(scores) / len(scores))

            if len(col_scores) > 0:
                col.append(f'{sum(col_scores) / len(col_scores):.4f}')
            else:
                col.append('//')
            result.add_column(model_name, col)

        print(result)


if __name__ == '__main__':
    fire.Fire(main)