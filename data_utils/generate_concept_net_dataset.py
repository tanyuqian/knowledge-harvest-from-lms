#!/usr/bin/env python
# coding: utf-8
import csv
from collections import defaultdict
from itertools import chain
import json
import random
# copied from the code.
conceptnet_relation_init_prompts = {
    'AtLocation': [
        '<ENT0> is the location for <ENT1>',
        'the <ENT0> is where the <ENT1> is kept',
        '<ENT1> is located in <ENT0>'
    ],
    'CapableOf': [
        'Something that <ENT0> can typically do is <ENT1>',
        'a <ENT0> can typically <ENT1> things',
        'a <ENT0> can typically be used to <ENT1> something'
    ],
    'Causes': [
        'It is typical for <ENT0> to cause <ENT1>',
        'many people <ENT1> when they <ENT0>',
        '<ENT0> typically causes <ENT1>'
    ],
    'CausesDesire': [
        '<ENT0> makes someone want <ENT1>',
        '<ENT1> is a desire caused by <ENT0>',
        '<ENT0> causes <ENT1>'
    ],
    'CreatedBy': [
        '<ENT1> is a process or agent that creates <ENT0>',
        'a <ENT1> is someone or something that creates <ENT0>',
        'the process or agent that creates <ENT0> is <ENT1>'
    ],
    'DefinedAs': [
        '<ENT1> is a more explanatory version of <ENT0>',
        'the concept of <ENT0> is often described as the <ENT1>',
        '<ENT1> is a more accurate explanation of what <ENT0> is'
    ],
    'Desires': [
        '<ENT0> is a conscious entity that typically wants <ENT1>',
        '<ENT0> desires <ENT1>',
        '<ENT0> usually wants <ENT1>'
    ],
    'HasA': [
        '<ENT1> belongs to <ENT0>',
        'the <ENT1> is part of the <ENT0>',
        '<ENT0> can not function without <ENT1>'
    ],
    'HasFirstSubevent': [
        '<ENT0> is an event that begins with subevent <ENT1>',
        '<ENT0> begins with <ENT1>',
        '<ENT0> typically starts with <ENT1>',
    ],
    'HasLastSubevent': [
        '<ENT0> is an event that concludes with subevent <ENT1>',
        'after you finish <ENT0>, you have to <ENT1>',
        '<ENT0> usually includes <ENT1> afterwards'
    ],
    'HasPrerequisite': [
        'In order for <ENT0> to happen, <ENT1> needs to happen',
        '<ENT1> is a dependency of <ENT0>',
        '<ENT1> requires <ENT0>'
    ],
    'HasProperty': [
        '<ENT0> has <ENT1> as a property',
        '<ENT0> can be described as <ENT1>',
        '<ENT0> is known to be <ENT1>'
    ],
    'HasSubEvent': [
        '<ENT1> happens as a subevent of <ENT0>',
        '<ENT0> entails <ENT1>',
        'when <ENT0>, one often has to <ENT1>'
    ],
    'IsA': [
        '<ENT0> is a subtype or a specific instance of <ENT1>',
        'every <ENT0> is a <ENT1>',
        'every <ENT0> is a type of <ENT1>',
    ],
    'MadeOf': [
        '<ENT0> is made of <ENT1>',
        'the <ENT0> is made from <ENT1>',
        'the <ENT0> is composed of <ENT1>'
    ],
    'MotivatedByGoal': [
        'Someone does <ENT0> because they want result <ENT1>',
        '<ENT0> is a step toward accomplishing the goal <ENT1>',
        '<ENT0> is a necessary step in order to <ENT1>'
    ],
    'PartOf': [
        '<ENT0> is a part of <ENT1>',
        'the <ENT0> is located in the <ENT1>',
        '<ENT1> includes <ENT0> as one of its many components'
    ],
    'ReceivesAction': [
        '<ENT1> can be done to <ENT0>',
        'you can <ENT1> a <ENT0>',
        'you can <ENT1> your own <ENT0>'
    ],
    'SymbolOf': [
        '<ENT0> symbolically represents <ENT1>',
        '<ENT0> is often seen as a symbol of <ENT1>',
        '<ENT0> is often used to represent <ENT1>'
    ],
    'UsedFor': [
        '<ENT0> is used for <ENT1>',
        'the purpose of <ENT0> is <ENT1>',
        '<ENT0> is where you <ENT1>'
    ]
}
target_rel = list(conceptnet_relation_init_prompts.keys())
i = 0
tuple_list = defaultdict(list)
with open("conceptnet-assertions-5.7.0.csv", 'r', encoding='utf-8') as f:
    line = 1
    while line:
        line = f.readline()
        i += 1
        if i % 1000000 == 0:
            print(i)
        if len(line.strip()) > 0:
            parsed_line = line.strip().split("\t")
            if parsed_line[1].split('/')[-1] in target_rel:
                dic = json.loads(parsed_line[-1])
                head_ = parsed_line[2].split("/")
                tail_ = parsed_line[3].split("/")
                if head_[-2] != 'en' or tail_[-2] != 'en':
                    continue
                head = head_[-1].replace("_", " ")
                tail = tail_[-1].replace("_", " ")
                # if "surfaceStart" in dic and "surfaceEnd" in dic:
                #     assert dic["surfaceStart"].replace("-", " ") == head and dic["surfaceEnd"].replace("-", " ") == tail, (dic["surfaceStart"].replace("-", " ") , head , dic["surfaceEnd"].replace("-", " ") , tail)
                tuple_list[parsed_line[1].split(
                    '/')[-1]].append((head, tail, dic["weight"]))
stats = [(r, len(ls)) for r, ls in tuple_list.items()]
print("all: ", sorted(stats, key=lambda x: x[-1]))
stats = [(r, len([i for i in ls if i[-1] > 1]))
         for r, ls in tuple_list.items()]
print("confidence > 1: ", sorted(stats, key=lambda x: x[-1]))
random.sample([1, 2], 2)
def get_dataset(rel="Desires", quality="high", max_num_truth=1000):
    if quality == 'high':
        bank = {k: sorted([l for l in v if l[-1] > 1], key=lambda x: -x[-1])
                [:max_num_truth] for k, v in tuple_list.items()}
    else:
        bank = tuple_list
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
    return true_pairs + false_rel_pairs + false_h_pairs + false_t_pairs
temp = get_dataset()
dataset = []
for rel in target_rel:
    dataset += get_dataset(rel=rel, quality="high", max_num_truth=1000)
with open("data/ckbc/conceptnet_high_quality.txt", 'w', encoding='utf-8') as f:
    f.write("\n".join(dataset))
