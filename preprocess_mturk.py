import os
import json
import csv
import random
eval_rels = {
    "business": "business",
    "help": "help",
    "ingredient_for": "ingredient_for",
    "place_for": "place_for",
    "prevent": "prevent",
    "processed_from": "source of",  # the name was a bit misleading
    "representative_figure": "representative_figure",
    "separated_by_the_ocean": "separated_by_the_ocean",
    "synonym": "antonym"  # used to be a typo
 } 
 # 9 relations in total. Will process 3-ary relation separately.
 # fixed some erros. 

models = [
    # "bert-base-cased",
    "roberta-large",
    "bert-large-cased",
    "roberta-base",
    "distilbert-base-cased"
]

rel_info = json.load(open("data/relation_info_human_5seeds.json"))
rel_info = {eval_rels[i]: j for i, j in rel_info.items() if i in eval_rels}

settings = [
    "1000tuples_1prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0",
    "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0",
    # "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0_autoprompts",
    "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0_initprompts"
]

setting_short = {
    "1000tuples_1prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0": "1p",
    "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0": "20p",
    "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0_autoprompts": "autoprompts",
    "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0_initprompts": "initprompts"
}


default_setting = "1000tuples_20prompts_5seeds_maxsubwords2_maxrepeat5_temp1.0"
default_models = "roberta-large"

settings_with_model = \
    [(setting, default_models) for setting in settings] + \
    [(default_setting, model) for model in models]


settings_with_model = list(set(settings_with_model))
for i in settings_with_model:
    print(i)

merged = []
for setting_with_model in settings_with_model:
    path = os.path.join("outputs", "human", setting_with_model[0], setting_with_model[1])
    setting, model = setting_with_model
    setting = setting_short[setting]
    for relation in eval_rels:
        file = os.path.join(path, relation, "ent_tuples.json")
        results = json.load(open(file))
        tuples = [(*i[0], eval_rels[relation], ind, setting, model) for ind, i in enumerate(results[:100])]
        merged += tuples

json.dump(merged, open("merged.json", "w"))
random.shuffle(merged)
lengths = len(merged)
print(lengths)
"""
n_batch = 5
batchs = [merged[i:i+n_batch] for i in range(0, lengths, n_batch)]
print("batch size: ", [len(i) for i in batchs])
for idx, samples in enumerate(batchs):
    # ent_1_1,relation_1,ent_1_2,relation_1_def,example_1_1_head,example_1_1_tail,example_1_2_head,example_1_2_tail,example_1_3_head,example_1_3_tail,ent_2_1,relation_2,ent_2_2,relation_2_def,example_2_1_head,example_2_1_tail,example_2_2_head,example_2_2_tail,example_2_3_head,example_2_3_tail,ent_3_1,relation_3,ent_3_2,relation_3_def,example_3_1_head,example_3_1_tail,example_3_2_head,example_3_2_tail,example_3_3_head,example_3_3_tail
"""

with open('human_mturk_2_ary_relations.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(
        ["ent_1_1","relation_1","ent_1_2","setting_1", "model_1", "rank_1", "relation_1_def",
        "example_1_1_head", "example_1_1_tail", "example_1_2_head", "example_1_2_tail", "example_1_3_head" , "example_1_3_tail",
        "ent_2_1", "relation_2", "ent_2_2", "setting_2", "model_2", "rank_2", "relation_2_def",
        "example_2_1_head", "example_2_1_tail", "example_2_2_head", "example_2_2_tail", "example_2_3_head", "example_2_3_tail",
        "ent_3_1", "relation_3", "ent_3_2", "setting_3", "model_3", "rank_3", "relation_3_def",
        "example_3_1_head", "example_3_1_tail", "example_3_2_head", "example_3_2_tail", "example_3_3_head", "example_3_3_tail"])
    for start_idx in range(0, lengths, 3):
        content = []
        for idx in range(start_idx, start_idx + 3):
            rel = merged[idx][2]
            pairs = rel_info[rel]["seed_ent_tuples"]
            content += [
                merged[idx][0], rel, merged[idx][1], merged[idx][4], merged[idx][5], merged[idx][3],
                rel_info[rel]["init_prompts"][0].replace("<ENT0>", "A").replace("<ENT1>", "B"),
                pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1], pairs[2][0], pairs[2][1]
            ]
        spamwriter.writerow(content)
    # spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
