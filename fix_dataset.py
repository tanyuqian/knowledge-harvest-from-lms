
import json
import random

# making prompts better by adding "." and capitalized the first letter.
# re-sampled the seed entity pairs for LAMA (the previous pairs are all lowercased)

# conceptnet
with open("data/(old)relation_info_conceptnet_5seeds.json") as f:
    data = json.load(f)
for rel, info in data.items():
    for ind in range(len(info["prompts"])):
        p_ = info["prompts"][ind].strip()
        if not p_.endswith("."):
            p_ = p_ + ' .'
        if not p_.startswith("<"):
            p_ = p_[0].upper() + p_[1:]
        info["prompts"][ind] = p_
    for ind in range(len(info["init_prompts"])):
        p_ = info["init_prompts"][ind].strip()
        if not p_.endswith("."):
            p_ = p_ + ' .'
        if not p_.startswith("<"):
            p_ = p_[0].upper() + p_[1:]
        info["init_prompts"][ind] = p_
with open("data/relation_info_conceptnet_5seeds.json", "w") as f:
    json.dump(data, f)

# lama
with open("data/(old)relation_info_lama_5seeds.json") as f:
    data = json.load(f)


for rel, info in data.items():
    rel_id = rel.split("_")[0]
    dev_file = f"data/lama/cmp_lms_data/{rel_id}/test.jsonl"
    x = open(dev_file).readlines()
    x = [json.loads(i) for i in random.sample(x, 5)]
    info["seed_ent_tuples"] = [(i["sub_label"], i['obj_label']) for i in x]
    for ind in range(len(info["prompts"])):
        p_ = info["prompts"][ind].strip()
        if not p_.endswith("."):
            p_ = p_ + ' .'
        if not p_.startswith("<"):
            p_ = p_[0].upper() + p_[1:]
        info["prompts"][ind] = p_
    for ind in range(len(info["init_prompts"])):
        p_ = info["init_prompts"][ind].strip()
        if not p_.endswith("."):
            p_ = p_ + ' .'
        if not p_.startswith("<"):
            p_ = p_[0].upper() + p_[1:]
        info["init_prompts"][ind] = p_
with open("data/relation_info_lama_5seeds.json", "w") as f:
    json.dump(data, f)