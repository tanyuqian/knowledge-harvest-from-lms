import os

from data_utils.data_utils import get_relations


class CKBC:
    def __init__(self, rel_set='conceptnet'):
        test_filename = f'test_{rel_set}.txt'

        if not os.path.exists(f'data/ckbc/{test_filename}'):
            os.system('bash data_utils/download_ckbc.sh')

        self._ent_tuples = {}
        self._label = {}
        for line in open(f'data/ckbc/{test_filename}').readlines():
            rel, ent0, ent1, label = line.split('\t')

            if rel_set == 'lama':
                rel = get_lama_relation(rel=rel)

            # ent0, ent1 = ent0.lower(), ent1.lower()

            if rel not in self._ent_tuples:
                self._ent_tuples[rel] = []
                self._label[rel] = {}

            self._ent_tuples[rel].append([ent0, ent1])
            self._label[rel][f'{ent0} ||| {ent1}'] = int(label)

    def get_ent_tuples(self, rel):
        return self._ent_tuples[rel]

    def get_label(self, rel, ent_tuple):
        return self._label[rel][' ||| '.join(ent_tuple)]


def get_lama_relation(rel):
    lama_rels = get_relations(rel_set='lama')

    for lama_rel in lama_rels:
        if lama_rel.endswith(rel.replace(' ', '_')):
            return lama_rel

    return None


# ckbc = CKBC()
# for t in ckbc.get_ent_tuples(rel='AtLocation')[:10]:
#     print(t)
#
# print(ckbc.get_label(rel='AtLocation', ent_tuple=['fish', 'water']))