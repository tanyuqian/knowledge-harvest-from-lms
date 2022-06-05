## BertNet: Harvesting Knowledge Graphs from PLMs

![](framework.png)

### Environment
We use `python 3.8` and all the required packages can be installed by pip:
```
pip install -r requirements.txt
```

### Prompt Creation
```
python search_prompts.py --rel_set conceptnet (lama/human)
```

### Entity Pair Search
```
python main.py --rel_set conceptnet --model_name roberta-large
```

### Present results
```
python present_result.py --result_dir your_result_dir
```