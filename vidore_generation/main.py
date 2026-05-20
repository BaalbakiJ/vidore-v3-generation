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

from vidore_generation.dtos import Failed, LLMProviderConfig, Prompt
from vidore_generation.generation_handlers.factory import make_generation_handler
from vidore_generation.generation_schemas import Summary
from vidore_generation.page_filtering.page_manifest import (
    build_page_manifest as build_page_manifest_files,
    get_excluded_image_page_numbers_by_filename,
    load_page_manifest,
)
from vidore_generation.pipelines.llm_pipeline import LLMPipeline
from vidore_generation.pipelines.visual_summary_pipeline import VisualSummaryPipeline
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
@click.option(
    "--respect-page-manifest",
    is_flag=True,
    help="Skip pages excluded by page_manifest.jsonl.",
)
def create_images(documents_dir: Path, respect_page_manifest: bool):
    if documents_dir.name != "pdfs":
        raise click.ClickException(
            "create-images expects a path to a directory named 'pdfs'"
        )

    excluded_page_numbers_by_filename: dict[str, set[int]] = {}
    if respect_page_manifest:
        manifest_path = documents_dir.parent / "page_manifest.jsonl"
        if not manifest_path.exists():
            raise click.ClickException(
                "Page manifest not found. Run "
                "'vidore-generation build-page-manifest --config ...' first: "
                f"{manifest_path}"
            )
        manifest_rows = load_page_manifest(manifest_path)
        excluded_page_numbers_by_filename = (
            get_excluded_image_page_numbers_by_filename(
                manifest_rows,
                "exclude_from_image_rendering",
            )
        )

    imgs_path = documents_dir.parent / "imgs"
    imgs_path.mkdir(exist_ok=True)
    for pdf_path in tqdm(sorted(documents_dir.iterdir())):
        if pdf_path.suffix != ".pdf":
            continue

        file_stem = pdf_path.stem
        doc_folder_path = imgs_path / file_stem
        if doc_folder_path.exists():
            continue

        doc_folder_path.mkdir(exist_ok=True)
        doc = pdfium.PdfDocument(pdf_path)
        number_of_pages = len(doc)
        excluded_page_numbers = excluded_page_numbers_by_filename.get(
            file_stem,
            set(),
        )
        for page_number in range(number_of_pages):
            if page_number in excluded_page_numbers:
                continue
            page = doc[page_number]
            img_path = doc_folder_path / f"{file_stem}_{page_number}.png"
            page.render(scale=200 / 72).to_pil().save(img_path)


@cli.command("build-page-manifest")
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def build_page_manifest_command(config):
    dataset_dir = Path(os.path.join(config["documents_dir"], config["dataset_name"]))
    pdfs_dir = dataset_dir / "pdfs"
    page_rows, summary_rows = build_page_manifest_files(
        pdfs_dir=pdfs_dir,
        output_dir=dataset_dir,
    )
    toc_page_count = sum(1 for row in page_rows if row["is_toc_page"])

    click.echo(f"PDFs analyzed: {len(summary_rows)}")
    click.echo(f"Pages analyzed: {len(page_rows)}")
    click.echo(f"TOC pages detected: {toc_page_count}")
    click.echo(f"Page manifest JSONL: {dataset_dir / 'page_manifest.jsonl'}")
    click.echo(f"Page manifest CSV: {dataset_dir / 'page_manifest.csv'}")
    click.echo(f"TOC summary CSV: {dataset_dir / 'toc_detection_summary.csv'}")


@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True, path_type=Path))
def extract_text_from_pdfs(dataset_path):
    from vidore_generation.pdf_parsing.extract_text_from_pdfs import parse_pdfs

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
        llm_provider=llm_provider,
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
        llm_provider=llm_provider,
    )
    vlm_pipeline.run()


@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def visual_summaries(config):
    llm_provider = _parse_llm_provider(config)
    dataset_name = config["dataset_name"]
    dataset_dir = Path(os.path.join(config["documents_dir"], dataset_name))
    visual_summary_config = config.get("visual_summary") or {}
    default_document_description = (
        "Internal technical, validation, regulatory, product, and RAQA documents."
    )

    pipeline = VisualSummaryPipeline(
        dataset_dir=dataset_dir,
        llm_provider=llm_provider,
        language=config.get("language", "english"),
        persona=config.get("persona", ""),
        debug=config.get("debug", False),
        section_size=visual_summary_config.get("section_size", 1),
        stride=visual_summary_config.get("stride", 1),
        max_windows=visual_summary_config.get("max_windows"),
        max_windows_per_document=visual_summary_config.get(
            "max_windows_per_document"
        ),
        max_documents=visual_summary_config.get("max_documents"),
        max_summary_words=visual_summary_config.get("max_summary_words", 250),
        filtered_summaries_nb=visual_summary_config.get(
            "filtered_summaries_nb",
            config.get("filtered_summaries_nb", 50),
        ),
        overwrite_existing=visual_summary_config.get("overwrite_existing", False),
        document_description=visual_summary_config.get(
            "document_description",
            default_document_description,
        ),
        respect_page_manifest=visual_summary_config.get(
            "respect_page_manifest",
            False,
        ),
        use_visual_document_descriptions=visual_summary_config.get(
            "use_visual_document_descriptions",
            False,
        ),
        max_description_pages=visual_summary_config.get(
            "max_description_pages",
            3,
        ),
        max_description_words=visual_summary_config.get(
            "max_description_words",
            150,
        ),
        overwrite_descriptions=visual_summary_config.get(
            "overwrite_descriptions",
            False,
        ),
        use_visual_combined_summaries=visual_summary_config.get(
            "use_visual_combined_summaries",
            False,
        ),
        overwrite_combined_summaries=visual_summary_config.get(
            "overwrite_combined_summaries",
            False,
        ),
        use_visual_summary_judging=visual_summary_config.get(
            "use_visual_summary_judging",
            False,
        ),
        overwrite_judgments=visual_summary_config.get(
            "overwrite_judgments",
            False,
        ),
        visual_judgment_min_grade=visual_summary_config.get(
            "visual_judgment_min_grade",
            4,
        ),
        combination_iteration_nb=config.get("combination_iteration_nb", 20),
        sampling_multi_doc_ratio=config.get("sampling_multi_doc_ratio", 0.5),
    )
    filtered_summaries = pipeline.run()
    click.echo(
        "Visual summaries written to "
        f"{dataset_dir / 'filtered_summaries' / 'filtered_summaries.json'} "
        f"({len(filtered_summaries)} filtered summaries)."
    )


@cli.command()
@click.option(
    "--config",
    type=click.Path(),
    callback=load_config,
    is_eager=True,
    expose_value=False,
    help="Path to config file.",
)
def test_bedrock(config):
    from botocore.exceptions import BotoCoreError, ClientError

    llm_provider = _parse_llm_provider(config)
    if llm_provider.provider != "bedrock":
        raise click.ClickException(
            "test-bedrock requires llm_provider.provider=bedrock"
        )

    try:
        handler = make_generation_handler(llm_provider, "lm")
        result = handler.generate_single_sample(
            Prompt(
                messages=[
                    {
                        "role": "user",
                        "content": 'Return JSON: {"summary": "Bedrock works"}',
                    }
                ],
                arguments={"pydantic_schema": Summary},
            )
        )
    except (BotoCoreError, ClientError) as error:
        raise click.ClickException(
            "Bedrock smoke test failed "
            f"for model={llm_provider.lm_model_name}, "
            f"region={llm_provider.aws_region}, "
            f"profile={llm_provider.aws_profile}: {error}"
        ) from error
    if isinstance(result, Failed):
        raise click.ClickException(result.error or "Bedrock smoke test failed")
    if not isinstance(result, Summary):
        raise click.ClickException(f"Unexpected Bedrock smoke test result: {result}")
    click.echo(result.model_dump_json())


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
            llm_provider=llm_provider,
        )

    vidore_juicer_rephrased_queries = rephrase_queries(
        environment=environment,
        model_name=llm_provider.lm_model_name,
        queries_path=os.path.join(
            output_dir, "queries", f"vidore_juicer_{dataset_name}_queries_filtered.json"
        ),
        language=language,
        extra_kwargs=llm_provider.lm_extra_kwargs,
        llm_provider=llm_provider,
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


