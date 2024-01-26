from typing import Callable

class Function:
    def __init__(self, name, description, call:Callable):
        self.name = name
        self.description = description
        self.call = call
