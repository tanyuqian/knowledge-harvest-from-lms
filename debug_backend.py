prompt = 'people can <ENT0> at <ENT1>.'
examples = [
    ("exercise", "gym"),
    ("study", "a study room"),
    ("sleep", "hotel")
]
model = "roberta-large"

prompt = prompt.replace(' ', '_').replace("<ENT0>", "A").replace("<ENT1>", "B").replace("<ENT2>", "C")
example_str = "^".join(["~".join([ent_i.replace(" ", "_") for ent_i in example]) for example in examples])
s = f"127.0.0.1:8000/predict/{model}/{prompt.replace(' ', '_')}/{example_str}"
# 0.0.0.0:1111/predict/roberta-large/B_is_the_location_for_A/flotation_device~boat^water~soft_drink^gear~car^giraffes~africa^trousers~suitcase
print(s)