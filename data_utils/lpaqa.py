import json
from glob import glob


class LPAQA:
    def __init__(self, setting):
        self._prompts = {}

        for prompt_path in glob(f'data/lpaqa/{setting}/*'):
            rel_idx = prompt_path.split('/')[-1].split('.')[0]
            self._prompts[rel_idx] = []

            if prompt_path.endswith('txt'):
                for line in open(prompt_path).readlines():
                    prompt = line.strip().replace(
                        '[X]', '<ENT0>').replace('[Y]', '<ENT1>')
                    self._prompts[rel_idx].append({'prompt': prompt})

                for prompt in self._prompts[rel_idx]:
                    prompt['weight'] = 1. / len(self._prompts[rel_idx])

            elif prompt_path.endswith('jsonl'):
                for line in open(prompt_path).readlines():
                    prompt = json.loads(line)
                    self._prompts[rel_idx].append({
                        'prompt': prompt['template'].replace(
                            '[X]', '<ENT0>').replace('[Y]', '<ENT1>'),
                        'weight': prompt['weight']
                    })

    def get_prompts(self, rel):
        rel_idx = rel.split('_')[0]
        return self._prompts[rel_idx]


# lpaqa = LPAQA(setting='paraphrase')
# print(sum([value['weight'] for value in lpaqa.get_prompts('P1001_xxx')]))

