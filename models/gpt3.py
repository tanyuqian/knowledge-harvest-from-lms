import os
import openai


class GPT3:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")

    def call(self,
             prompt,
             engine="text-davinci-002",
             temperature=1.,
             max_tokens=30,
             top_p=1.,
             frequency_penalty=0,
             presence_penalty=0,
             logprobs=0,
             n=1):
        return openai.Completion.create(
            engine=engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            logprobs=logprobs,
            n=n)

