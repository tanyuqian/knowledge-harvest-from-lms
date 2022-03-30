import os
import pickle
import math
import numpy as np


DEL_RELS = ['HasPainIntensity', 'HasPainCharacter', 'LocationOfAction',
            'LocatedNear',
            'DesireOf', 'NotMadeOf', 'InheritsFrom', 'InstanceOf',
            'RelatedTo', 'NotDesires',
            'NotHasA', 'NotIsA', 'NotHasProperty', 'NotCapableOf']


class CKBCKnowledgeScorer:
    def __init__(self):
        if not os.path.exists('ckbc-demo/'):
            os.system('wget https://home.ttic.edu/~kgimpel/comsense_resources/'
                      'ckbc-demo.tar.gz')
            os.system('tar -zxvf ckbc-demo.tar.gz')

        model = pickle.load(open(
            "ckbc-demo/Bilinear_cetrainSize300frac1.0dSize200relSize150acti0."
            "001.1e-05.800.RAND.tanh.txt19.pickle",
            "rb"), encoding='latin1')

        self._Rel = model['rel']
        self._We = model['embeddings']
        self._Weight = model['weight']
        self._Offset = model['bias']
        self._words = model['words_name']
        self._rel = model['rel_name']

        for del_rel in DEL_RELS:
            self._rel.pop(del_rel.lower())

    @property
    def relations(self):
        return list(self._rel.keys())

    def score(self, h, r, t):
        h = h.replace(' ', '_')
        r = r.lower()
        t = t.replace(' ', '_')

        result = score(
            h, t, self._words, self._We, self._rel, self._Rel,
            self._Weight, self._Offset)

        return result[r]


def getVec(We, words, t):
    t = t.strip()
    array = t.split('_')
    if array[0] in words:
        vec = We[words[array[0]], :]
    else:
        vec = We[words['UUUNKKK'], :]
        print('can not find corresponding vector:', array[0].lower())
    for i in range(len(array) - 1):
        if array[i + 1] in words:
            vec = vec + We[words[array[i + 1]], :]
        else:
            print('can not find corresponding vector:', array[i + 1].lower())
            vec = vec + We[words['UUUNKKK'], :]
    vec = vec / len(array)
    return vec


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def score(term1, term2, words, We, rel, Rel, Weight, Offset):
    v1 = getVec(We, words, term1)
    v2 = getVec(We, words, term2)
    result = {}

    for k, v in rel.items():
        v_r = Rel[rel[k], :]
        gv1 = np.tanh(np.dot(v1, Weight) + Offset)
        gv2 = np.tanh(np.dot(v2, Weight) + Offset)

        temp1 = np.dot(gv1, v_r)
        score = np.inner(temp1, gv2)
        result[k] = (sigmoid(score))

    return result