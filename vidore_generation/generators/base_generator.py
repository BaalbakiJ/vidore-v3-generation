import os
from typing import Dict, Optional

from jinja2 import Environment, FileSystemLoader


class BaseGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.environment = Environment(
            loader=FileSystemLoader(os.path.join("vidore_generation", "prompts"))
        )
        self.template = None

    def create_prompt(
        self, input_instance: Dict[str, str], template: Optional[str] = None
    ) -> str:
        if template is None:
            template = self.template
            if template is None:
                raise ValueError("Template is not set")
        prompt = template.render(**input_instance)
        return prompt
