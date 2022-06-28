## BertNet: Harvesting Knowledge Graphs from PLMs

This repo contains preliminary code of the following paper:

BertNet: Harvesting Knowledge Graphs from Pretrained Language Models \
Shibo Hao*, Bowen Tan*, Kaiwen Tang*, Hengzhe Zhang, Eric P. Xing, Zhiting Hu \
(* Equal contribution) 

### Getting Started
* Symbolic knowledge graphs (KGs) have been constructed either by expensive human crowdsourcing or with domain-specific complex information extraction pipelines.
* In this work, we aim at harvesting symbolic KGs from the LMs, a new framework for automatic KG construction empowered by the neural LMs' flexibility and scalability. 
* Compared to prior works that often rely on large human annotated data or existing massive KGs, our approach requires only the minimal definition of relations as inputs, and hence is suitable for extracting knowledge of rich **new** relations not available before. 
* As a result, we derive from diverse LMs a family of new KGs (e.g., *BertNet* and *RoBERTaNet*) 
  * contain a richer set of commonsense relations, including complex ones (e.g., *"A is capable of but not good at B"*), than the human-annotated KGs (e.g., *ConceptNet*). 
* Besides, the resulting KGs also serve as a vehicle to interpret the respective source LMs, leading to new insights into the varying knowledge capability of different LMs.


![](framework.png)

### Requirements
We use `python 3.8` and all the required packages can be installed by pip:
```
pip install -r requirements.txt
```
Our code runs on a single GTX 1080Ti GPU.

### Harvesting KGs from LMs

#### Automatic Creation of Diverse Prompts
This step corresponds to [paper]() Section 3.1. 
```
python search_prompts.py --rel_set conceptnet
```
* `--rel_set`: can be one of conceptnet/lama/human, as shown in the [relation_info/](relation_info/) folder.

The prompt searching reads the relation definitions from `"init_prompts"` and `"seed_ent_tuples"` values from `relation_info/{rel_set}.json` and save the searched-out prompts into `"prompts"` values.\
(Files in the [relation_info/](relation_info/) folder have contained the results of this step.)

#### Efficient Search for Knowledge Tuples
This step corresponds to [paper]() Section 3.2.
```
python main.py --rel_set conceptnet --model_name roberta-large --n_ent_tuples 1000 --n_prompts 20
```
* `--rel_set`: The set of relations, can be one of `conceptnet`/`lama`/`human`, as shown in the [relation_info/](relation_info/) folder.
* `--model_name`: The LM as the source of knowledge, can be one of `roberta-large`/`roberta-base`/`bert-large-cased`/`bert-base-cased`/`disdilbert-base-cased`.
* `--max_n_ent_tuples`: Target size of knowledge tuples to search for every relation.
* `--max_n_prompts`: Number of prompts to be used in the searching of knowledge tuples. 

The searched-out knowledge tuples will be saved into [results/](results/), e.g., [results/conceptnet/1000tuples_top20prompts/roberta-large/](results/conceptnet/1000tuples_top20prompts/roberta-large/).


### Results
This command can present results in a formatted manner:
```
python present_result.py --result_dir results/conceptnet/1000tuples_top20prompts/roberta-large/
```
The results will be saved into `summary.txt` in `{result_dir}`, e.g., [results/conceptnet/1000tuples_top20prompts/roberta-large/summary.txt](results/conceptnet/1000tuples_top20prompts/roberta-large/summary.txt).



#### Example -- "A can B but not good at" (RoBERTaNet)
```
+------------------------------------+--------------------------------+-------------------------------------------+
|            Seed samples            |         Ours (Top 20)          | Ours (Random samples over top 200 tuples) |
+------------------------------------+--------------------------------+-------------------------------------------+
|         ['chicken', 'fly']         |        ['dog', 'fetch']        |             ['Cough', 'lung']             |
|          ['dog', 'swim']           |  ['human stomach', 'ferment']  |       ['young football', 'kickers']       |
| ['long-distance runner', 'sprint'] |      ['human', 'invent']       |            ['Bisexual', 'woo']            |
|         ['skater', 'ski']          |   ['hiker', 'cross country']   |            ['Cpl', 'firefight']           |
|      ['researcher', 'teach']       |      ['brawler', 'fight']      |            ['barrow', 'wheel']            |
|                 \                  |        ['dog', 'snoop']        |          ['png', 'image preview']         |
|                 \                  |        ['dog', 'snuff']        |              ['ant', 'snoop']             |
|                 \                  |   ['young children', 'sit']    |           ['baseball', 'field']           |
|                 \                  |     ['young child', 'sit']     |          ['individual', 'bluff']          |
|                 \                  |     ['Cpl', 'fire truck']      |             ['raven', 'roost']            |
|                 \                  |        ['frog', 'swim']        |             ['frog', 'sprint']            |
|                 \                  |        ['fern', 'grow']        |        ['hollywood', 'make movie']        |
|                 \                  |      ['mother', 'parent']      |           ['brug', 'water well']          |
|                 \                  |    ['young maid', 'gossip']    |              ['ant', 'sniff']             |
|                 \                  |       ['clown', 'sing']        |             ['bough', 'twirl']            |
|                 \                  |  ['Human Medium', 'manifest']  |             ['troll', 'tweet']            |
|                 \                  |  ['human medium', 'channel']   |             ['jack', 'shovel']            |
|                 \                  |       ['frog', 'sprint']       |             ['bull', 'bluff']             |
|                 \                  |    ['individual', 'invent']    |             ['maid', 'shower']            |
|                 \                  | ['Psychopath', 'Social Trick'] |            ['mother', 'father']           |
+------------------------------------+--------------------------------+-------------------------------------------+
```

#### Example -- "A needs B to C" (BertNet)
```
+------------------------------------------------+--------------------------------------------------------+--------------------------------------------------+
|                  Seed samples                  |                     Ours (Top 20)                      |    Ours (Random samples over top 200 tuples)     |
+------------------------------------------------+--------------------------------------------------------+--------------------------------------------------+
|       ['developer', 'computer', 'code']        |        ['good folk', 'good manners', 'behave']         |    ['Chinese Cuisine', 'ingredients', 'cook']    |
|     ['people', 'social media', 'connect']      |  ['social movements', 'social networks', 'organize']   |     ['union', 'workers', 'organize workers']     |
|            ['pig', 'food', 'grow']             | ['human cells', 'protein synthesis', 'differentiate']  |            ['girls', 'boys', 'play']             |
|          ['people', 'money', 'live']           |      ['small boats', 'small waves', 'float away']      |      ['certain rules', 'changes', 'comply']      |
| ['intern', 'good performance', 'return offer'] |   ['Social Scientists', 'Research Methods', 'teach']   |         ['movie', 'film', 'make movies']         |
|                       \                        |     ['small boats', 'small waves', 'float along']      |        ['students', 'resources', 'learn']        |
|                       \                        |       ['Chinese Cuisine', 'ingredients', 'cook']       |   ['small boats', 'small waves', 'float away']   |
|                       \                        |     ['female gender', 'gender norms', 'identify']      |        ['mankind', 'evolution', 'evolve']        |
|                       \                        |      ['young couples', 'special skills', 'cope']       |             ['men', 'blood', 'kill']             |
|                       \                        |     ['small boats', 'small waves', 'float safely']     | ['Christian Couples', 'counseling', 'reconcile'] |
|                       \                        |    ['single gender', 'gender identity', 'identify']    |       ['bar', 'alcohol', 'drink alcohol']        |
|                       \                        | ['Social Organizations', 'Social Workers', 'organize'] |     ['passengers', 'special trains', 'meet']     |
|                       \                        | ['human cells', 'protein molecules', 'differentiate']  |           ['males', 'females', 'feed']           |
|                       \                        |     ['university', 'Student Loans', 'pay tuition']     |   ['female may', 'male protection', 'escape']    |
|                       \                        |  ['strong society', 'strong leaders', 'lead society']  | ['human civilization', 'technology', 'advance']  |
|                       \                        |         ['law', 'legal means', 'enforce laws']         |           ['parents', 'help', 'cope']            |
|                       \                        |    ['human civilization', 'technology', 'advance']     |   ['Christian Faith', 'conversion', 'convert']   |
|                       \                        |        ['strong arms', 'strong legs', 'stand']         |            ['men', 'blood', 'fight']             |
|                       \                        |        ['good company', 'good friends', 'keep']        |           ['humans', 'blood', 'feed']            |
|                       \                        |    ['French Wines', 'Wine Glasses', 'taste better']    |  ['male cells', 'sperm cells', 'differentiate']  |
+------------------------------------------------+--------------------------------------------------------+--------------------------------------------------+```