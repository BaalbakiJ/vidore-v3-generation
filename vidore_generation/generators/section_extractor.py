import json
import logging
import os
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from jinja2 import Environment, FileSystemLoader
from rapidfuzz import fuzz
from tqdm import tqdm
from anyascii import anyascii as unidecode

from vidore_generation.dtos import Document, Prompt, Section, TOCCheck
from vidore_generation.generation_handlers.generation_handler import GenerationHandler

warnings.filterwarnings("ignore", category=UserWarning)


class SectionExtractor:
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        maximum_number_of_pages_per_section: int = 10,
        minimum_number_of_pages_per_section: int = 5,
        max_tokens: int = 20_000,
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        self.model_name = model_name
        self.maximum_number_of_pages_per_section = maximum_number_of_pages_per_section
        self.minimum_number_of_pages_per_section = minimum_number_of_pages_per_section
        self.max_tokens = max_tokens
        self.environment = Environment(
            loader=FileSystemLoader(
                os.path.join(
                    "vidore_generation",
                    "prompts",
                )
            )
        )
        self.template = self.environment.get_template("check_toc.j2")
        self.generation_handler = generation_handler
        self.logger = logger
        self.language = language

    @staticmethod
    def extract_toc_from_document(markdown: str) -> List[Tuple[str, int]]:
        table_of_contents = []
        in_image_description = False

        counter = 1
        for line in markdown.split("\n"):
            if "<!-- page break -->" in line:
                counter += 1
            if line.startswith('<!--<annotation kind="description">-->'):
                in_image_description = True
            elif "<!--<annotation/>-->" in line:
                in_image_description = False
            if not in_image_description:
                if line.startswith("# "):
                    table_of_contents.append((line, [], counter))
                elif line.startswith("## "):
                    if len(table_of_contents) == 0:
                        table_of_contents.append((None, [], counter))
                    table_of_contents[-1][1].append((line, [], counter))
                elif line.startswith("### "):
                    if len(table_of_contents) == 0:
                        table_of_contents.append((None, [], counter))
                    if len(table_of_contents[-1][1]) == 0:
                        table_of_contents[-1][1].append((None, [], counter))
                    table_of_contents[-1][1][-1][1].append((line, [], counter))
                elif line.startswith("#### "):
                    if len(table_of_contents) == 0:
                        table_of_contents.append((None, [], counter))
                    if len(table_of_contents[-1][1]) == 0:
                        table_of_contents[-1][1].append((None, [], counter))
                    if len(table_of_contents[-1][1][-1][1]) == 0:
                        table_of_contents[-1][1][-1][1].append((None, [], counter))
                    table_of_contents[-1][1][-1][1][-1][1].append((line, [], counter))
                elif line.startswith("##### "):
                    if len(table_of_contents) == 0:
                        table_of_contents.append((None, [], counter))
                    if len(table_of_contents[-1][1]) == 0:
                        table_of_contents[-1][1].append((None, [], counter))
                    if len(table_of_contents[-1][1][-1][1]) == 0:
                        table_of_contents[-1][1][-1][1].append((None, [], counter))
                    if len(table_of_contents[-1][1][-1][1][-1][1]) == 0:
                        table_of_contents[-1][1][-1][1][-1][1].append(
                            (None, [], counter)
                        )
                    table_of_contents[-1][1][-1][1][-1][1][-1][1].append(
                        (line, [], counter)
                    )

        return table_of_contents

    def flatten_table_of_contents(
        self, table_of_contents: list, flattened_table_of_contents: list
    ):
        for item in table_of_contents:
            if item[0] is not None:
                flattened_table_of_contents.append((item[0], item[2]))
            # assert isinstance(item[1], list), f"Expected list, got "{item[1]}" of type {type(item[1])}"
            self.flatten_table_of_contents(item[1], flattened_table_of_contents)
        return flattened_table_of_contents

    @staticmethod
    def get_number(line: str):
        if "|" in line:
            cells = line.split("|")
            for cell in cells:
                # if cell.strip().isdigit():
                if re.search(r"\d+", cell.strip()) and not re.search(
                    r"[a-z]+", cell.strip().lower()
                ):
                    digit_match = re.search(r"\d+", cell.strip())
                    return [int(digit_match.group())]
        else:
            line = line.strip()
            if line.startswith("-"):
                line = line[1:].strip()
            normalized_line = re.sub(r"[^a-zA-Z0-9,.\s]+", "", line).strip()
            digits = list(re.finditer(r"\d+", normalized_line))
            return [
                int(digit.group())
                for digit in digits
                if digit.start() == 0 or digit.end() == len(line)
            ]

    @staticmethod
    def deduplicate_titles(titles: List[Tuple[str, int]]):
        text_set = set()
        final_titles = []
        for title in titles:
            normalized_title = re.sub(r"[^a-zA-Z\s]+", "", title[0]).strip()
            same_page_titles = [
                re.sub(r"[^a-zA-Z\s]+", "", title_[0]).strip()
                for title_ in final_titles
                if title_[1] == title[1]
            ]
            if not any(
                [
                    fuzz.partial_ratio(normalized_title, title_) > 80
                    for title_ in same_page_titles
                ]
            ):
                text_set.add(normalized_title + str(title[1]))
                final_titles.append(title)
        return final_titles

    def filter_tables_of_contents(self, tables_of_contents: List[Tuple[str, int]]):
        first_filtered_tables_of_contents = []
        for table_of_content, page_nb in tables_of_contents:
            toc_element_nb = 0
            nb_of_lines_with_too_much_numbers = 0
            for line in table_of_content.split("\n"):
                normalized_line = re.sub(r"[^a-z0-9]\s+", " ", line.strip()).strip()
                if (
                    re.search(r"\d+", normalized_line)
                    and re.search(r"[a-z]+", normalized_line)
                    and len(line.strip()) < 150
                ):
                    nb_of_numbers = len(re.findall(r"\d+", normalized_line))
                    if nb_of_numbers < 4:
                        toc_element_nb += 1
                    else:
                        nb_of_lines_with_too_much_numbers += 1
            if nb_of_lines_with_too_much_numbers > 5:
                continue
            if toc_element_nb > 8:
                first_filtered_tables_of_contents.append((table_of_content, page_nb))
        filtered_tables_of_contents = []
        for page, page_nb in tqdm(first_filtered_tables_of_contents):
            normalized_page = re.sub(r"\s+", " ", page.lower())
            normalized_page = re.sub("table of contents", "", normalized_page)
            toc_check = self.generation_handler.generate_single_sample(
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": self.environment.get_template(
                                "check_toc.j2"
                            ).render(page=normalized_page, language=self.language),
                        }
                    ],
                    arguments={"pydantic_schema": TOCCheck, "max_tokens": 10_000},
                )
            )
            if toc_check.has_table_of_contents:
                filtered_tables_of_contents.append((page, page_nb))
        return filtered_tables_of_contents

    def extract_table_of_content_pages(self, markdown: str):
        pages = markdown.split("<!-- page break -->")
        tables_of_contents_pages = []
        for i, page in enumerate(pages):
            normalized_page = unidecode(re.sub(r"\s+", " ", page.lower()))
            if any(
                matcher in normalized_page
                for matcher in [
                    "table of contents",
                    "| page |",
                    "index",
                    "page no.",
                    "table des matieres",
                ]
            ):
                tables_of_contents_pages.append((page, i))
        tables_of_contents_pages = self.filter_tables_of_contents(
            tables_of_contents_pages
        )
        titles = []
        for toc_page, page_nb in tables_of_contents_pages:
            first_nb = None
            for line in toc_page.split("\n"):
                normalized_line = re.sub(r"\s+", " ", line.replace("|", "").strip())
                numbers = self.get_number(line)
                if numbers:
                    if first_nb is None and numbers[0] < page_nb - 10:
                        self.logger.debug("============BROKEN===============")
                        self.logger.debug(toc_page)
                        titles = []
                        break
                    titles.append((normalized_line.strip(), numbers[0]))
                    first_nb = numbers[0]
        return self.deduplicate_titles(sorted(titles, key=lambda x: x[1]))

    @staticmethod
    def split_page(page: str):
        for pattern in ["# ", "## ", "### ", "#### ", "##### "]:
            if page.strip().startswith(pattern):
                return "", page
            else:
                if "\n" + pattern in page:
                    index = page.index("\n" + pattern)
                    return page[:index], page[index:]
        return "", page

    def divide_section(
        self,
        section: str,
        document: Document,
        division_size: int = 5,
        start_page_index: int = 0,
    ) -> List[Section]:
        pages = section.split("<!-- page break -->")
        sections = []
        current_section = ""
        range_start = start_page_index
        for i in range(0, len(pages), division_size):
            upper_offset = min(i + division_size, len(pages))
            first_half, second_half = self.split_page(pages[upper_offset - 1])
            if len(first_half) > 0:
                current_section += (
                    "<!-- page break -->".join(pages[i : upper_offset - 1])
                    + "<!-- page break -->"
                    + first_half
                )
            else:
                current_section += "<!-- page break -->".join(pages[i:upper_offset])
            sections.append(
                Section(
                    document_id=document.id,
                    filename=document.filename,
                    document_description=document.document_description.description,
                    section=current_section,
                    page_numbers=list(
                        range(range_start, start_page_index + upper_offset)
                    ),
                )
            )
            if len(first_half) > 0:
                current_section = second_half + "<!-- page break -->"
                range_start = start_page_index + upper_offset - 1
            else:
                current_section = ""
                range_start = start_page_index + upper_offset
        return sections

    def get_sections(
        self, document: Document, titles: list, separators: List[str]
    ) -> List[Section]:
        markdown = document.content
        if len(separators) == 0:
            return self.divide_section(markdown, document)
        else:
            start_index = 0
            offsets = []
            for separator, page_nb_from_toc in separators:
                for i, (title, real_page_number) in enumerate(titles[start_index:]):
                    if (
                        fuzz.partial_ratio(separator, re.sub("#+", "", title).strip())
                        > 90
                    ):
                        offsets.append(real_page_number - page_nb_from_toc)
                        start_index = i + 1
                        break
            if offsets:
                actual_offset = Counter(offsets).most_common(1)[0][0]
            else:
                actual_offset = 0

            pages = markdown.split("<!-- page break -->")
            sections = []
            current_section = ""
            current_nb_of_pages = 0
            last_section_page_nb = 0
            first_page_index = 0
            for separator, page_nb_from_toc in separators:
                actual_page_nb = page_nb_from_toc + actual_offset - 1
                if actual_page_nb > len(pages):
                    break
                else:
                    separator_section = "<!-- page break -->".join(
                        pages[last_section_page_nb:actual_page_nb]
                    )
                    current_nb_of_pages += actual_page_nb - last_section_page_nb
                    if len(current_section) > 0:
                        current_section += "<!-- page break -->"
                    current_section += separator_section
                    if current_nb_of_pages > self.maximum_number_of_pages_per_section:
                        if len(current_section) > 0:
                            sections.extend(
                                self.divide_section(
                                    current_section,
                                    document,
                                    start_page_index=first_page_index,
                                )
                            )
                            current_section = ""
                            current_nb_of_pages = 0
                            first_page_index = actual_page_nb
                    elif (
                        current_nb_of_pages >= self.minimum_number_of_pages_per_section
                    ):
                        sections.append(
                            Section(
                                document_id=document.id,
                                filename=document.filename,
                                document_description=document.document_description.description,
                                section=current_section,
                                page_numbers=list(
                                    range(first_page_index, actual_page_nb)
                                ),
                            )
                        )
                        current_section = ""
                        current_nb_of_pages = 0
                        first_page_index = actual_page_nb
                    last_section_page_nb = actual_page_nb
            if actual_page_nb + 1 < len(pages):
                if len(current_section) > 0:
                    current_section += "<!-- page break -->"
                rest_of_document = current_section + "<!-- page break -->".join(
                    pages[actual_page_nb:]
                )
                current_nb_of_pages += len(pages) - actual_page_nb
                if current_nb_of_pages <= self.maximum_number_of_pages_per_section:
                    sections.append(
                        Section(
                            document_id=document.id,
                            filename=document.filename,
                            document_description=document.document_description.description,
                            section=rest_of_document,
                            page_numbers=list(range(first_page_index, len(pages))),
                        )
                    )
                else:
                    sections.extend(
                        self.divide_section(
                            rest_of_document,
                            document,
                            start_page_index=first_page_index,
                        )
                    )
            return sections

    def extract_sections(self, document: Document) -> List[Section]:
        document_markdown = document.content
        titles = self.flatten_table_of_contents(
            self.extract_toc_from_document(document_markdown), []
        )
        separators = self.extract_table_of_content_pages(document_markdown)
        sections = self.get_sections(document, titles, separators)
        return sections

    def export(self, output_dir: Path, sections: List[Section], filename: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
            json.dump(
                [json.loads(section.model_dump_json()) for section in sections],
                f,
                indent=4,
            )
