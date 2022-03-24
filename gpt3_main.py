import os
import json

from models.gpt3 import GPT3

gpt3 = GPT3()

prompt = '<ENT0> is part of <ENT1>'
ent_tuple = ['facebook', 'google']

print(gpt3.get_paraphrase_prompt(prompt=prompt, ent_tuple=ent_tuple))

# # prompt = '<ENT0> is part of <ENT1>'
# # prompt = 'The capital of <ENT0> is <ENT1>'
# # prompt = '<ENT0> plays <ENT1> music'
# prompt = '<ENT0> is a member of the <ENT1> political party'
# result = gpt3.get_ent_tuples(prompt=prompt, n=5)
#
# print(f'#retrieved tuples: {len(result)}')
#
# os.makedirs('outputs/', exist_ok=True)
# output_fn = prompt.replace(' ', '_').replace('<', '').replace('>', '')
# json.dump(result, open(f'outputs/{output_fn}.json', 'w'), indent=4)
