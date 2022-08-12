## BertNet: Harvesting Knowledge Graphs from PLMs

![](framework.png)

### Environment

We use `python 3.8` and all the required packages can be installed by pip:

```
pip install -r requirements.txt
```

### Running Steps

1. Start Backend server
2. Start Front-end Server

### Start Backend server

```
FLASK_ENV=development python server_main.py
```

in your browser

````
0.0.0.0:1111/predict/roberta-large/B_is_the_location_for_A/flotation_device~boat^water~soft_drink^gear~car^giraffes~africa^trousers~suitcase
````

Note: On some servers where the location of python is not available in the environmental variable, we need to add the
location to the environment variable by modifying the "server_main.py".

### Start Front-end Server

```bash
# It is recommended to run the front-end server at the "front_end_server" folder.
cd front_end_server; python Server.py
```

Note: When querying a new item, it may take a few seconds to get the result from the model. Once the result has been
retrieved, our server will store the result in the local disk. In the next time of querying the same item, the result
will be shown in less than one second.