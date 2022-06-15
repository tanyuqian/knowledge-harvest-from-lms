## BertNet: Harvesting Knowledge Graphs from PLMs

![](framework.png)

### Environment
We use `python 3.8` and all the required packages can be installed by pip:
```
pip install -r requirements.txt
```

### Start server
```
FLASK_ENV=development python server_main.py
```
in your browser
````
0.0.0.0:1111/predict/roberta-large/B_is_the_location_for_A/flotation_device~boat^water~soft_drink^gear~car^giraffes~africa^trousers~suitcase
````