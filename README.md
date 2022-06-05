## BertNet: Harvesting Knowledge Graphs from PLMs

![](framework.png)

### Preperation
```
pip install -r requirements.txt
```

### Search prompts
```
python search_prompts.py --rel_set conceptnet (lama/human)
```

### Search entity tuples
```
python main.py --rel_set conceptnet --model_name roberta-large
```

### Present results
```
python present_result.py --result_dir your_result_dir
```