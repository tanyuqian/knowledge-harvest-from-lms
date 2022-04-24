import os
import json
import random
from collections import defaultdict
from itertools import chain

data_dir = "data/lama"

relations = []
with open(os.path.join(data_dir, "LAMA_relations.jsonl"), 'r') as f:
    for line in f.readlines():
        relations.append(json.loads(line))
try:
    with open(os.path.join(data_dir, "lama_facts.json"), 'r') as f:
        results = json.load(f)
except:
    results = {}
    for relation in relations:
        dir_name = relation["relation"]
        name = relation["label"]
        # if dir_name != 'P1376':
        #     continue
        root = os.path.join(data_dir, "cmp_lms_data", dir_name)
        data = []
        if not os.path.exists(root):
            continue
        for file in os.listdir(root):
            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                data += f.readlines()
        res = []
        for l in data:
            dic = json.loads(l)
            sub, obj = dic["sub_label"], dic["obj_label"]
            # if sub in tok.vocab and obj in tok.vocab:
            res.append([sub, obj])
        results[name] = res
        print(f"{dir_name}: {len(res)} pairs.")


    with open(os.path.join(data_dir, "lama_facts.json"), 'w') as f:
        json.dump(results, f)
    # break
tuple_list = {k: [(ins[0], ins[1], 2) for ins in v] for k, v in results.items()}
def get_dataset(rel="Desires", quality="high", max_num_truth=1000):
    if quality == 'high':
        bank = {k: sorted([l for l in v if l[-1] > 1], key=lambda x: -x[-1])
                [:max_num_truth] for k, v in tuple_list.items()}
    else:
        bank = tuple_list
    # print(bank)
    if len(bank.get(rel, [])) == 0:
        return []
    true_pairs = ["\t".join([rel, h, t, "1"]) for h, t, w in bank[rel]]
    false_rel_pools = list(
        chain([(k, *l) for k, v in bank.items() if k != rel for l in v]))
    false_rel_pairs = ["\t".join([rel, h, t, "0"]) for _, h, t, w in random.sample(
        false_rel_pools, len(bank[rel])//3)]
    facts = [(h, t) for h, t, w in bank[rel]]
    false_h_pairs = []
    false_t_pairs = []
    replaced_tuples = random.sample(
        bank[rel], len(bank[rel]) - len(bank[rel])//3)
    for h, t, w in replaced_tuples[:len(bank[rel])//3]:
        h_ = h
        while (h_, t) in facts:
            h_, _ = random.sample(facts, 1)[0]
        false_h_pairs.append("\t".join([rel, h_, t, "0"]))
    for h, t, w in replaced_tuples[len(bank[rel])//3:]:
        t_ = t
        while (h, t_) in facts:
            _, t_ = random.sample(facts, 1)[0]
        false_h_pairs.append("\t".join([rel, h, t_, "0"]))
        # there can be some repetitive pairs
    return true_pairs + false_rel_pairs + false_h_pairs + false_t_pairs

dataset = []
# print(tuple_list)
for rel in tuple_list.keys():
    # print(rel)
    dataset += get_dataset(rel=rel, quality="high", max_num_truth=1000)
with open("data/lama/lama_test.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(dataset))

'''
P19: 1779 pairs.
P20: 1817 pairs.
P279: 1900 pairs.
P37: 1170 pairs.
P413: 1952 pairs.
P449: 1801 pairs.
P47: 1138 pairs.
P138: 1116 pairs.
P364: 1756 pairs.
P463: 302 pairs.
P101: 1238 pairs.
P106: 1821 pairs.
P527: 1922 pairs.
P530: 958 pairs.
P176: 1925 pairs.
P27: 1958 pairs.
P407: 1857 pairs.
P30: 1959 pairs.
P178: 1167 pairs.
P1376: 212 pairs.
P131: 1775 pairs.
P1412: 1924 pairs.
P108: 750 pairs.
P136: 1859 pairs.
P17: 1912 pairs.
P39: 1485 pairs.
P264: 1053 pairs.
P276: 1764 pairs.
P937: 1853 pairs.
P140: 860 pairs.
P1303: 1513 pairs.
P127: 1114 pairs.
P103: 1919 pairs.
P190: 769 pairs.
P1001: 1664 pairs.
P31: 1879 pairs.
P495: 1905 pairs.
P159: 1801 pairs.
P36: 876 pairs.
P740: 1843 pairs.
P361: 1131 pairs
'''