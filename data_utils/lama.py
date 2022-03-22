import os
import json


class LAMA:
    def __init__(self):
        if not os.path.exists('data/lama'):
            os.system('bash data_utils/download_lama.sh')

    @property
    def relations(self):
        relations = []
        for line in open('data/lama/relations.jsonl').readlines():
            relations.append(json.loads(line))

        return relations