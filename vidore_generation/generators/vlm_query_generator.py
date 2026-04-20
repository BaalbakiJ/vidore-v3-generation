import base64
import logging
import warnings
from typing import List

from pydantic import BaseModel, Field

from vidore_generation.dtos import ImageSection, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generators.base_generator import BaseGenerator
from vidore_generation.generators.samplers import (
    FigureModuleSampler,
    TableModuleSampler,
    TextModuleSampler,
)
from vidore_generation.generators.structs import QueryModule

warnings.filterwarnings("ignore", category=UserWarning)


# def encode_image(image):
#     """
#     Encodes a PIL Image object to a base64 string.
#     """
#     buffer = BytesIO()
#     image.save(buffer, format="JPEG")
#     return base64.b64encode(buffer.getvalue()).decode("utf-8")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class Query(BaseModel):
    query_type: str = Field(description="Type of query")
    query_format: str = Field(description="Format of the query")
    query: str = Field(description="Query")


class Queries(BaseModel):
    queries: List[Query]


class VLMQueryGenerator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.single_section_template = self.environment.get_template(
            "single_section_query_generation.j2"
        )
        self.multi_section_template = self.environment.get_template(
            "multi_section_query_generation.j2"
        )
        self.generation_handler = generation_handler
        self.language = language

    def sample_query_modules(self) -> List[QueryModule]:
        query_modules = []
        query_modules.extend(TextModuleSampler.sample(1))
        query_modules.extend(FigureModuleSampler.sample(1))
        query_modules.extend(TableModuleSampler.sample(1))
        return query_modules

    def generate_single_section_queries(
        self, sections_list: List[ImageSection]
    ) -> List[Query]:
        prompts = []
        for section in sections_list:
            content = [
                {
                    "type": "text",
                    "text": self.create_prompt(
                        {
                            "document_description": section.document_description,
                            "language": self.language,
                            "query_modules": self.sample_query_modules(),
                        },
                        template=self.single_section_template,
                    ),
                }
            ]
            for image_path in section.image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                    }
                )
            prompts.append(
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    arguments={"pydantic_schema": Queries},
                )
            )
        return self.generation_handler.generate_multiple_samples(
            prompts,
            semaphore_size=2,
            desc="Generating single section queries",
        )

    def generate_multi_section_queries(
        self, combinations: List[List[ImageSection]]
    ) -> List[Query]:
        prompts = []
        for combination in combinations:
            content = [
                {
                    "type": "text",
                    "text": self.create_prompt(
                        {
                            "language": self.language,
                            "query_modules": self.sample_query_modules(),
                        },
                        template=self.multi_section_template,
                    ),
                }
            ]
            for i, section in enumerate(combination):
                for image_path in section.image_paths:
                    content.append(
                        {
                            "type": "text",
                            "text": f"Images of section {i + 1}",
                        }
                    )
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        }
                    )
            prompts.append(
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    arguments={"pydantic_schema": Queries},
                )
            )
        return self.generation_handler.generate_multiple_samples(
            prompts,
            semaphore_size=2,
            desc="Generating single section queries",
        )

    # def export(
    #     self, output_dir: Path, summaries: List[FinalSummary], docid2filename: dict
    # ):
    #     os.makedirs(output_dir, exist_ok=True)
    #     for docid in docid2filename:
    #         filename = docid2filename[docid]
    #         print(filename)
    #         sub_summaries = [
    #             summary for summary in summaries if summary.filenames[0] == filename
    #         ]
    #         with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
    #             json.dump(
    #                 [
    #                     json.loads(summary.model_dump_json())
    #                     for summary in sub_summaries
    #                 ],
    #                 f,
    #                 indent=4,
    #             )
