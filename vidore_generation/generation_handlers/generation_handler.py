from abc import ABC, abstractmethod
from typing import List

from vidore_generation.dtos import Prompt


class GenerationHandler(ABC):
    @abstractmethod
    def generate_single_sample(self, prompt: Prompt) -> str:
        pass

    @abstractmethod
    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[str]:
        pass
