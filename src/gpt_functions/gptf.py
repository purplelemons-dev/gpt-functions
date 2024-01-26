import openai
from typing import Callable, Literal
from .util import Function


class gptf:
    def __init__(self, client: openai.OpenAI):
        self.client = client
        self.functions: list[Function] = []

    def __call__(self, function: Callable):
        def wrapper(*args, **kwargs):
            self.functions.append(
                Function(function.__name__, function.__doc__, function)
            )
            return function(*args, **kwargs)

        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def chat(
        self,
        messages: list[dict[Literal["role", "message"], str]],
        model=Literal["gpt-4", "gpt-4-0613", "gpt-4-1106-preview"],
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
    ):
        return self.client.chat.completions.create(
            model=model,
            prompt=messages,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            tools=self.functions,
        )
