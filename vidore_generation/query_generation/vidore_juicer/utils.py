import base64
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

from vidore_generation.query_generation.vidore_juicer.structs import SectionSummary


def encode_image(path: Path) -> str:
    """Encodes an image to a base64 string."""
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def assign_number_of_questions(
    max_questions_per_page: int,
    max_textual_questions_per_page: int,
    max_tabular_questions_per_page: int,
    max_visual_questions_per_page: int,
    max_adversarial_questions_per_page: int,
    seed: Optional[int],
) -> Dict[str, int]:
    """Assigns a number of questions per type based on the given constraints."""
    available_types = ["adversarial"]
    available_types.append("textual")
    available_types.append("tabular")
    available_types.append("visual")
    counts = {"textual": 0, "tabular": 0, "visual": 0, "adversarial": 0}
    total_questions = 0
    rng = random.Random(seed)

    while total_questions < max_questions_per_page and available_types:
        question_type = rng.choice(available_types)
        if (
            question_type == "textual"
            and counts["textual"] < max_textual_questions_per_page
        ):
            counts["textual"] += 1
        elif (
            question_type == "tabular"
            and counts["tabular"] < max_tabular_questions_per_page
        ):
            counts["tabular"] += 1
        elif (
            question_type == "visual"
            and counts["visual"] < max_visual_questions_per_page
        ):
            counts["visual"] += 1
        elif (
            question_type == "adversarial"
            and counts["adversarial"] < max_adversarial_questions_per_page
        ):
            counts["adversarial"] += 1
        else:
            available_types.remove(question_type)
            continue
        total_questions += 1

    return counts


def get_images_path(filename: str, export_images_dir: Path) -> List[str]:
    """Gets the paths of images associated with a given filename."""
    name_file = filename.split(".")[0]
    path_images = []
    i = 0
    image_path = export_images_dir / name_file / f"{i}.jpg"
    while os.path.exists(image_path):
        path_images.append(image_path)
        i += 1
        image_path = export_images_dir / name_file / f"{i}.jpg"
    return path_images


def get_doc_list(doc_folder: Path) -> List[str]:
    """Returns a list of document names in the given folder."""
    return [doc.stem for doc in doc_folder.iterdir() if doc.is_file()]


def fuse_summaries(summaries: List[SectionSummary]) -> SectionSummary:
    """Fuses multiple section summaries into a single summary."""
    return SectionSummary(summary=" ".join([summary.summary for summary in summaries]))
