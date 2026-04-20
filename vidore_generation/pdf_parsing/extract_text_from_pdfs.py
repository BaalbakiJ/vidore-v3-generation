import io
import json
import logging
import os
from pathlib import Path
from typing import List, Union

import fsspec
import pypdfium2 as pdfium
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.io import DocumentStream
from tqdm import tqdm

from vidore_generation.pdf_parsing.custom_markdown_serializer import (
    custom_export_to_markdown,
)

_log = logging.getLogger(__name__)

NUMBER_OF_JOBS = 4


def upload_directory_to_gcs_fsspec(local_dir, gcs_path):
    fs = fsspec.filesystem("gcs")  # Requires gcsfs to be installed

    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            gcs_file_path = f"{gcs_path.rstrip('/')}/{rel_path}"

            with open(local_path, "rb") as f_local:
                with fs.open(gcs_file_path, "wb") as f_gcs:
                    f_gcs.write(f_local.read())
                    print(f"Uploaded {local_path} to {gcs_file_path}")


def concat_exports(
    conv_results: List[ConversionResult], export_type: str
) -> Union[str, List[dict]]:
    if all(
        conv_result.status == ConversionStatus.SUCCESS for conv_result in conv_results
    ):
        if export_type == "markdown":
            return "\n\n<!-- page break -->\n\n".join(
                [
                    custom_export_to_markdown(
                        conv_result.document,
                        page_break_placeholder="<!-- page break -->",
                        mark_annotations=True,
                    )
                    if isinstance(conv_result, ConversionResult)
                    else conv_result
                    for conv_result in conv_results
                ]
            )
        elif export_type == "html":
            return "\n".join(
                [conv_result.document.export_to_html() for conv_result in conv_results]
            )
        elif export_type == "doctags":
            return "\n".join(
                [
                    conv_result.document.export_to_doctags()
                    for conv_result in conv_results
                ]
            )
        elif export_type == "json":
            return [
                conv_result.document.export_to_dict() for conv_result in conv_results
            ]
        else:
            raise ValueError(f"Invalid export type: {export_type}")
    else:
        raise ValueError("Not all documents were successfully converted.")


def export_document(
    conv_results: List[Union[ConversionResult, str]],
    markdown_path: Path,
    extractions_path: Path,
    output_filename: Path,
):
    print(output_filename)
    markdown_filename = os.path.join(markdown_path, f"{output_filename}.md")
    os.makedirs(os.path.dirname(markdown_filename), exist_ok=True)

    if any(isinstance(conv_result, str) for conv_result in conv_results):
        with open(markdown_filename, "w", encoding="utf-8") as fp:
            fp.write(concat_exports(conv_results, "markdown"))

    else:
        # Export Docling document format to markdown:
        with open(markdown_filename, "w", encoding="utf-8") as fp:
            fp.write(concat_exports(conv_results, "markdown"))

        with open(
            os.path.join(extractions_path, f"{output_filename}.json"),
            "w",
            encoding="utf-8",
        ) as fp:
            json.dump(concat_exports(conv_results, "json"), fp)

        with open(
            os.path.join(extractions_path, f"{output_filename}.html"),
            "w",
            encoding="utf-8",
        ) as fp:
            fp.write(concat_exports(conv_results, "html"))

        with open(
            os.path.join(extractions_path, f"{output_filename}.doctags.txt"),
            "w",
            encoding="utf-8",
        ) as fp:
            fp.write(concat_exports(conv_results, "doctags"))


logging.basicConfig(level=logging.WARNING)


def convert_image(sample):
    img_byte_arr = io.BytesIO()
    sample["image"].save(img_byte_arr, format="PNG")

    buf = io.BytesIO(img_byte_arr.getvalue())
    return DocumentStream(name=f"{sample['doc-id']}_{sample['corpus-id']}", stream=buf)


def parse_pdfs(dataset_path):
    dataset_path = Path(dataset_path)
    markdown_path = dataset_path.parent / "markdowns"
    extractions_path = dataset_path.parent / "extractions"
    os.makedirs(markdown_path, exist_ok=True)
    os.makedirs(extractions_path, exist_ok=True)

    pipeline_options = PdfPipelineOptions()
    # pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_picture_description = True
    pipeline_options.enable_remote_services = True  # <-- this is required!
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url="https://api.fireworks.ai/inference/v1/chat/completions",
        params=dict(
            model="accounts/fireworks/models/kimi-k2p5",
            max_completion_tokens=1000,
        ),
        prompt=(
            "Analyze and describe the visual content in detail. "
            "For graphs and charts: identify key data points, trends, and patterns. "
            "For tables: summarize the data structure, key values, and relationships. "
            "For images: describe the main elements, composition, and context."
            "Write a description with no more than 200 words."
        ),
        headers={
            "Authorization": "Bearer " + os.getenv("FIREWORKS_API_KEY"),
        },
        timeout=90,
    )

    accelerator_options = AcceleratorOptions(
        num_threads=8, device=AcceleratorDevice.MPS
    )
    pipeline_options.accelerator_options = accelerator_options

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    # for dataset_path in ["report_pdfs", "bank_pdfs"]:
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            try:
                if file.endswith(".pdf"):
                    input_filepath = os.path.join(root, file)
                    output_filename = file.replace(".pdf", "")
                    if os.path.exists(
                        os.path.join(markdown_path, f"{output_filename}.md")
                    ):
                        print(f"Skipping {output_filename} because it already exists")
                        continue
                    else:
                        print(f"Converting {output_filename}")
                        # get number of pages
                        doc = pdfium.PdfDocument(input_filepath)
                        num_pages = len(doc)
                        conv_results = []
                        if num_pages > 80:
                            # Create a temporary directory to store split PDFs
                            split_pdf_dir = "split_pdfs"
                            os.makedirs(split_pdf_dir, exist_ok=True)
                            split_pdf_paths = []

                            # Split the PDF into chunks of 30 pages or less
                            chunk_size = 30
                            for i in range(0, num_pages, chunk_size):
                                chunk_start = i
                                chunk_end = min(i + chunk_size, num_pages)
                                chunk_doc = pdfium.PdfDocument.new()
                                chunk_doc.import_pages(doc, list(range(chunk_start, chunk_end)))
                                chunk_filename = f"{output_filename}_part_{chunk_start + 1}_to_{chunk_end}.pdf"
                                chunk_path = os.path.join(split_pdf_dir, chunk_filename)
                                chunk_doc.save(chunk_path)
                                split_pdf_paths.append(chunk_path)
                            # Optionally, you could process each chunk separately here
                            # For now, just process the first chunk as an example
                            input_filepath = split_pdf_paths[0]
                            for split_pdf_path in tqdm(
                                split_pdf_paths, leave=False, desc="Converting chunks"
                            ):
                                md_path = os.path.join(
                                    split_pdf_dir,
                                    f"{output_filename}_part_{i + 1}_to_{i + chunk_size}.md",
                                )
                                if os.path.exists(md_path):
                                    with open(md_path, "r") as f:
                                        conv_result = f.read()
                                else:
                                    conv_result = doc_converter.convert(
                                        split_pdf_path, raises_on_error=True
                                    )
                                conv_results.append(conv_result)
                        else:
                            conv_result = doc_converter.convert(
                                input_filepath, raises_on_error=True
                            )
                            conv_results.append(conv_result)

                        print(f"Exporting document {output_filename}")
                        export_document(
                            conv_results,
                            markdown_path,
                            extractions_path,
                            output_filename=output_filename,
                        )
                        # raise Exception("Not implemented")

                # _log.info(f"Document conversion complete in {end_time:.2f} seconds.")

                # if failure_count > 0:
                #     raise RuntimeError(
                #         f"The example failed converting {failure_count} on {len(docs)}."
                #     )
            except Exception as e:
                # raise e
                _log.warning(f"Error converting document {file}: {e}")
                os.makedirs(os.path.join(extractions_path, "errors"), exist_ok=True)
                with open(
                    os.path.join(extractions_path, "errors", f"{file}_error.json"), "w"
                ) as f:
                    json.dump({"error": str(e)}, f)

    # upload_directory_to_gcs_fsspec(
    #     local_dir=markdown_path, gcs_path=f"gcs://vidore-generation/{markdown_path}"
    # )
    # upload_directory_to_gcs_fsspec(
    #     local_dir=extractions_path,
    #     gcs_path=f"gcs://vidore-generation/{extractions_path}",
    # )
