import os
import json


class LAMA:
    def __init__(self):
        if not os.path.exists('data/lama'):
            os.system('bash data_utils/download_lama.sh')

        info = {}
        for line in open('data/lama/relations.jsonl').readlines():
            rel = json.loads(line)
            rel_name = rel['relation'] + '_' + \
                       rel['label'].replace('/', '').replace(' ', '_')
            info[rel_name] = {
                'init_prompts': [rel['template'].replace(
                    '[X]', '<ENT0>').replace('[Y]', '<ENT1>')],
                'description': rel['description']
            }

        for rel in info:
            rel_id = rel.split('_')[0]
            rel_ent_tuple_path = f'data/lama/TREx/{rel_id}.jsonl'
            if os.path.exists(rel_ent_tuple_path) == False:
                continue

            info[rel]['ent_tuples'] = []
            for line in open(rel_ent_tuple_path).readlines():
                term = json.loads(line)
                info[rel]['ent_tuples'].append(
                    [term['sub_label'], term['obj_label']])

        self._info = {}
        for rel in info:
            if 'ent_tuples' in info[rel]:
                self._info[rel] = info[rel]

    @property
    def info(self):
        return self._info


# lama = LAMA()
# for rel in lama.info:
#     print(lama.info[rel])
#     break