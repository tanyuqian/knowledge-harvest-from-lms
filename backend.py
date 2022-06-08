import fire
import os


MAX_N_ENT_TUPLES_LIST = [20, 40]


def main(model_name, init_prompts_str, seed_ent_tuples_str):
    for max_n_ent_tuples in [0] + MAX_N_ENT_TUPLES_LIST:
        cmd = f'python demo.py '\
              f'--init_prompts_str {init_prompts_str} ' \
              f'--seed_ent_tuples_str {seed_ent_tuples_str} ' \
              f'--model_name {model_name} ' \
              f'--max_n_ent_tuples {max_n_ent_tuples}'

        if max_n_ent_tuples > 0:
            cmd += ' &'

        os.system(cmd)


if __name__ == '__main__':
    fire.Fire(main)