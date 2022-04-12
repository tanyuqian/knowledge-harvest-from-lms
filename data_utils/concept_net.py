import os
import json
import random
import gdown


CONCEPT_NET_GDRIVE_ID = '1WM34NKTyAm4o0DX3B74XcMG7h7sKPrp9'


class ConceptNet:
    def __init__(self):
        if not os.path.exists('data/concept_net/'):
            os.makedirs('data/concept_net/', exist_ok=True)
            gdown.download(
                id=CONCEPT_NET_GDRIVE_ID,
                output='data/concept_net/concept_net.json')

        knowledge = json.load(open('data/concept_net/concept_net.json'))
        self._ent_tuples = {}
        for relation in knowledge:
            print(f'processing relation {relation}...')

            ent_tuples = []
            for key in knowledge[relation]:
                for ent_tuple in knowledge[relation][key]:
                    if ent_tuple[0] == key:
                        ent_tuples.append(ent_tuple[:2])

            random.shuffle(ent_tuples)
            self._ent_tuples[relation] = ent_tuples

    def get_ent_tuples(self, rel):
        return self._ent_tuples[rel]