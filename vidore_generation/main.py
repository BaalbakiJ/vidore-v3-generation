import json
import logging
import os
import random
import re
from pathlib import Path

import click
import anyascii
import litellm
import pypdfium2 as pdfium
import yaml
from tqdm import tqdm

from vidore_generation.dtos import LLMProviderConfig
from vidore_generation.pdf_parsing.extract_text_from_pdfs import parse_pdfs
from vidore_generation.pipelines.llm_pipeline import LLMPipeline
from vidore_generation.pipelines.vlm_pipeline import VLMPipeline

logging.getLogger("pydantic_core").setLevel(logging.ERROR)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)


def load_config(ctx, param, value):
    if not value:
        return {}
    with open(value, "r") as f:
        ctx.params["config"] = yaml.safe_load(f)
    return ctx.params["config"]


def _parse_llm_provider(config: dict) -> LLMProviderConfig:
    return LLMProviderConfig(**config["llm_provider"])


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("documents_dir", type=click.Path(exists=True, path_type=Path))
def create_images(documents_dir):
    assert "/pdfs" in str(documents_dir)
    imgs_path = str(documents_dir).replace("/pdfs", "/imgs")
    os.makedirs(imgs_path, exist_ok=True)
    for filename in tqdm(os.listdir(documents_dir)):
        file_stem = filename.split(".")[0]
        doc_folder_path = os.path.join(imgs_path, file_stem)
        if filename.endswith(".pdf") and not os.path.exists(doc_folder_path):
            os.makedirs(doc_folder_path, exist_ok=True)
            doc = pdfium.PdfDocument(os.path.join(documents_dir, filename))
            number_of_pages = len(doc)
            for page_number in range(number_of_pages):
                page = doc[page_number]
                img_path = os.path.join(
                    imgs_path, file_stem, f"{file_stem}_{page_number}.png"
                )
                page.render(scale=200 / 72).to_pil().save(img_path)


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
def extract_text_from_pdfs(dataset_path):
    parse_pdfs(dataset_path)


@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def llm(config):
    llm_provider = _parse_llm_provider(config)
    dataset_name = config["dataset_name"]
    dataset_dir = Path(os.path.join(config["documents_dir"], dataset_name))
    persona = config["persona"]
    debug = config["debug"]
    combination_iteration_nb = config["combination_iteration_nb"]
    sampling_multi_doc_ratio = config["sampling_multi_doc_ratio"]
    language = config["language"]
    filtered_summaries_nb = config["filtered_summaries_nb"]

    litellm.verbose = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.enable_cache("disk")
    os.makedirs(dataset_dir, exist_ok=True)

    llm_pipeline = LLMPipeline(
        dataset_dir,
        model_name=llm_provider.lm_model_name,
        persona=persona,
        debug=debug,
        combination_iteration_nb=combination_iteration_nb,
        sampling_multi_doc_ratio=sampling_multi_doc_ratio,
        language=language,
        filtered_summaries_nb=filtered_summaries_nb,
        extra_kwargs=llm_provider.lm_extra_kwargs,
    )
    llm_pipeline.run()


@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def vlm(config):
    llm_provider = _parse_llm_provider(config)
    dataset_name = config["dataset_name"]
    dataset_dir = Path(os.path.join(config["documents_dir"], dataset_name))
    persona = config["persona"]
    debug = config["debug"]
    language = config["language"]
    combination_iteration_nb = config.get("combination_iteration_nb", 20)
    sampling_multi_doc_ratio = config.get("sampling_multi_doc_ratio", 0.5)
    section_size = config.get("section_size", 5)

    litellm.verbose = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.enable_cache("disk")
    os.makedirs(dataset_dir, exist_ok=True)

    vlm_pipeline = VLMPipeline(
        dataset_dir,
        lm_model_name=llm_provider.lm_model_name,
        vl_model_name=llm_provider.vl_model_name,
        persona=persona,
        section_size=section_size,
        debug=debug,
        combination_iteration_nb=combination_iteration_nb,
        sampling_multi_doc_ratio=sampling_multi_doc_ratio,
        language=language,
        lm_extra_kwargs=llm_provider.lm_extra_kwargs,
        vl_extra_kwargs=llm_provider.vl_extra_kwargs,
    )
    vlm_pipeline.run()


@cli.command()
@click.argument("documents_dir", type=click.Path(exists=True, path_type=Path))
def check_extractions(documents_dir):
    pdfs_dir = os.path.join(documents_dir, "pdfs")
    markdowns_dir = os.path.join(documents_dir, "markdowns")
    random.seed(42)

    pdf_set = set([x[:-4] for x in os.listdir(pdfs_dir) if x.endswith(".pdf")])
    md_set = set([x[:-3] for x in os.listdir(markdowns_dir) if x.endswith(".md")])
    if pdf_set != md_set:
        print("pdfs not in markdowns: ", pdf_set - md_set)
        print("markdowns not in pdfs: ", md_set - pdf_set)
        raise ValueError("Number of pdfs and markdowns is not the same")
    for filename in tqdm(sorted(os.listdir(pdfs_dir))):
        try:
            if filename.endswith(".pdf"):
                doc = pdfium.PdfDocument(os.path.join(pdfs_dir, filename))
                number_of_pages = len(doc)
                with open(
                    os.path.join(markdowns_dir, f"{filename.split('.')[0]}.md"), "r"
                ) as f:
                    markdown_text = f.read()
                number_of_md_pages = len(markdown_text.split("<!-- page break -->"))
                if number_of_pages != number_of_md_pages:
                    print(
                        f"Number of pages in {filename} is {number_of_pages} but {number_of_md_pages} in the markdown file"
                    )
                markdowns = markdown_text.split("<!-- page break -->")
                # print(filename)
                # for i, markdown in random.sample(list(enumerate(markdowns)), 2):
                #     print(f"Page {i + 1}")
                #     print("-" * 100)
                #     print(markdown[:100])
                #     print("-" * 100)
                # print("=" * 100)
        except Exception as e:
            print(f"Error processing {filename}: {e}")



@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def normalize_docs(config):
    data_dir = os.path.join(config["documents_dir"], config["dataset_name"])
    pdf_nb = len(os.listdir(os.path.join(data_dir, "pdfs")))
    markdown_exists = os.path.exists(os.path.join(data_dir, "markdowns"))
    if markdown_exists:
        markdown_nb = len(os.listdir(os.path.join(data_dir, "markdowns")))
        assert set([x[:-4] for x in os.listdir(os.path.join(data_dir, "pdfs"))]) == set(
            [x[:-3] for x in os.listdir(os.path.join(data_dir, "markdowns"))]
        )

    for filename in os.listdir(os.path.join(data_dir, "pdfs")):
        base_name = filename[:-4]
        new_base_name = anyascii.anyascii(base_name)
        new_base_name = re.sub(r"[\s\-]", "_", new_base_name).lower()
        os.rename(
            os.path.join(data_dir, "pdfs", base_name + ".pdf"),
            os.path.join(data_dir, "pdfs", new_base_name + ".pdf"),
        )
        if markdown_exists:
            os.rename(
                os.path.join(data_dir, "markdowns", f"{base_name}.md"),
                os.path.join(data_dir, "markdowns", f"{new_base_name}.md"),
            )
    assert len(os.listdir(os.path.join(data_dir, "pdfs"))) == pdf_nb
    if markdown_exists:
        assert len(os.listdir(os.path.join(data_dir, "markdowns"))) == markdown_nb


@cli.command()
@click.argument(
    "input-file",
    type=click.Path(exists=True),
    required=True,
    default="experiments/summaries.json",
)
@click.argument(
    "generation-config-path",
    type=click.Path(exists=True),
    required=True,
    default="experiments/generation_config.json",
)
def generate_queries_vidore_juicer(input_file, generation_config_path):
    from vidore_generation.query_generation.vidore_juicer.generate_custom_queries import (
        run_vidore_juicer_generation,
    )

    run_vidore_juicer_generation(input_file, generation_config_path)


@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def postprocess_queries(config):
    """Filter and rephrase vidore-juicer queries."""
    import random

    from jinja2 import Environment, FileSystemLoader

    from vidore_generation.query_generation.filter_queries import filter_queries
    from vidore_generation.query_generation.rephrase_queries import rephrase_queries

    litellm.verbose = False
    litellm.suppress_debug_info = True
    litellm.drop_params = True
    litellm.enable_cache("disk")
    random.seed(42)

    llm_provider = _parse_llm_provider(config)
    dataset_name = config["dataset_name"]
    language = config["language"]
    debug = config["debug"]
    environment = Environment(loader=FileSystemLoader("vidore_generation/prompts"))
    output_dir = os.path.join(config["documents_dir"], dataset_name)

    if not os.path.exists(
        os.path.join(
            output_dir, "queries", f"vidore_juicer_{dataset_name}_queries_filtered.json"
        )
    ):
        filter_queries(
            environment=environment,
            model_name=llm_provider.lm_model_name,
            queries_path=os.path.join(
                output_dir, "queries", f"vidore_juicer_{dataset_name}_queries.json"
            ),
            debug=debug,
            extra_kwargs=llm_provider.lm_extra_kwargs,
        )

    vidore_juicer_rephrased_queries = rephrase_queries(
        environment=environment,
        model_name=llm_provider.lm_model_name,
        queries_path=os.path.join(
            output_dir, "queries", f"vidore_juicer_{dataset_name}_queries_filtered.json"
        ),
        language=language,
        extra_kwargs=llm_provider.lm_extra_kwargs,
    )

    final_queries = []
    for query_json in vidore_juicer_rephrased_queries:
        if random.random() < 0.5:
            query_json.pop("rephrased_query")
            query_json["generation_process"] = "vidore_juicer"
            query_json["original_query"] = None
        else:
            query_json["original_query"] = query_json["query"]
            query_json["query"] = query_json.pop("rephrased_query")
            query_json["generation_process"] = "vidore_juicer_rephrased"
        final_queries.append(query_json)

    with open(
        os.path.join(output_dir, "queries", f"final_{dataset_name}_queries.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(final_queries, file)


