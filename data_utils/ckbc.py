import os
import json


class CKBC:
    def __init__(self, file="test.txt"):
        if not os.path.exists('data/ckbc'):
            os.system('bash data_utils/download_ckbc.sh')

        self._ent_tuples = {}
        self._label = {}
        for line in open(f'data/ckbc/{file}').readlines():
            rel, ent0, ent1, label = line.split('\t')

            if rel not in self._ent_tuples:
                self._ent_tuples[rel] = []
                self._label[rel] = {}

            self._ent_tuples[rel].append([ent0, ent1])
            self._label[rel][f'{ent0} ||| {ent1}'] = int(label)

    def get_ent_tuples(self, rel):
        return self._ent_tuples[rel]

    def get_label(self, rel, ent_tuple):
        return self._label[rel][' ||| '.join(ent_tuple)]


# ckbc = CKBC()
# for t in ckbc.get_ent_tuples(rel='AtLocation')[:10]:
#     print(t)
#
# print(ckbc.get_label(rel='AtLocation', ent_tuple=['fish', 'water']))