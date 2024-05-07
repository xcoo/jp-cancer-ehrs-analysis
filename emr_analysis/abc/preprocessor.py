from abc import ABC, abstractmethod
from typing import Any


class Preprocessor(ABC):
    default_options: dict[str, Any] = {}

    def __init__(self, options: dict[str, Any] = {}):
        self.options = self.default_options | options

    @abstractmethod
    def preprocess(self, rows: list[list[str]]) -> list[list[str]]:
        pass
