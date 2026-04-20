import json
import os
from collections import defaultdict
from typing import List

from vidore_generation.dtos import FinalSummary, Judgment, Section

filtered_summaries_nb = 400


class SummaryFilter:
    def __init__(self):
        pass

    def filter_summaries(
        self,
        single_section_summaries: List[FinalSummary],
        combined_summaries: List[FinalSummary],
        sections: List[Section],
        judgments: List[Judgment],
        filtered_summaries_nb: int = 400,
    ) -> List[FinalSummary]:
        number_of_single_section_summaries = len(single_section_summaries)
        summaries = single_section_summaries + combined_summaries

        sections_dict = defaultdict(list)
        for section in sections:
            sections_dict[section.filename].append(section)

        # assert sum([len(x) for x in sections.values()]) == number_of_single_section_summaries
        # assert len(summaries) == len(judgments)
        final_single_doc_summaries = {"visual": [], "non_visual": []}
        for summary, judgment in zip(
            single_section_summaries, judgments[:number_of_single_section_summaries]
        ):
            if all(
                judgment["grade"] > 3 for judgment in judgment.model_dump().values()
            ):
                summary_sections = []
                for filename in summary.filenames:
                    for page_numbers in summary.page_numbers:
                        for section in sections_dict[filename]:
                            if any(
                                page_nb in section.page_numbers
                                for page_nb in page_numbers
                            ):
                                summary_sections.append(section)
                if any(
                    "<!--<annotation kind=" in section.section
                    for section in summary_sections
                ):
                    final_single_doc_summaries["visual"].append(summary)
                else:
                    final_single_doc_summaries["non_visual"].append(summary)

        final_single_doc_combined_summaries = {"visual": [], "non_visual": []}
        final_multi_doc_combined_summaries = {"visual": [], "non_visual": []}
        matched_documents = set()

        for summary, judgment in zip(
            combined_summaries, judgments[number_of_single_section_summaries:]
        ):
            summary_id = " ".join(
                [
                    str(filename) + str(page_number)
                    for filename, page_number in zip(
                        summary.filenames, summary.page_numbers
                    )
                ]
            )
            if all(
                judgment["grade"] > 3 for judgment in judgment.model_dump().values()
            ):
                if summary_id in matched_documents:
                    continue
                matched_documents.add(summary_id)
                summary_sections = []
                for filename in summary.filenames:
                    for page_numbers in summary.page_numbers:
                        for section in sections_dict[filename]:
                            if any(
                                page_nb in section.page_numbers
                                for page_nb in page_numbers
                            ):
                                summary_sections.append(section)
                if len(set(summary.filenames)) == 1:
                    if any(
                        "<!--<annotation kind=" in section.section
                        for section in summary_sections
                    ):
                        final_single_doc_combined_summaries["visual"].append(summary)
                    else:
                        final_single_doc_combined_summaries["non_visual"].append(
                            summary
                        )
                else:
                    if any(
                        "<!--<annotation kind=" in section.section
                        for section in summary_sections
                    ):
                        final_multi_doc_combined_summaries["visual"].append(summary)
                    else:
                        final_multi_doc_combined_summaries["non_visual"].append(summary)

        print(
            "Nb of multi doc combined summaries with visual information:",
            len(final_multi_doc_combined_summaries["visual"]),
        )
        print(
            "Nb of combined summaries with visual information:",
            len(final_single_doc_combined_summaries["visual"]),
        )
        print(
            "Nb of single doc summaries with visual information:",
            len(final_single_doc_summaries["visual"]),
        )
        print(
            "Nb of multi doc combined summaries without visual information:",
            len(final_multi_doc_combined_summaries["non_visual"]),
        )
        print(
            "Nb of combined summaries without visual information:",
            len(final_single_doc_combined_summaries["non_visual"]),
        )
        print(
            "Nb of single doc summaries without visual information:",
            len(final_single_doc_summaries["non_visual"]),
        )

        summaries_by_priority = [
            final_multi_doc_combined_summaries["visual"],
            final_single_doc_combined_summaries["visual"],
            final_single_doc_summaries["visual"],
            final_multi_doc_combined_summaries["non_visual"],
            final_single_doc_combined_summaries["non_visual"],
            final_single_doc_summaries["non_visual"],
        ]
        summary_categories = [
            "summary based on sections from multiple documents with visual information",
            "summary based on sections from a single document with visual information",
            "summary based on a section from a single document with visual information",
            "summary based on sections from multiple documents without visual information",
            "summary based on sections from a single document without visual information",
            "summary based on a section from a single document without visual information",
        ]

        final_summaries = []
        missing_summaries_nb = filtered_summaries_nb
        counter = defaultdict(int)
        assert sum(len(summaries) for summaries in summaries_by_priority) >= filtered_summaries_nb, "Not enough summaries to filter from. Please lower the filtered_summaries_nb or check the input data."
        while len(final_summaries) < filtered_summaries_nb:
            for i, summaries in enumerate(summaries_by_priority):
                if i < 3:
                    max_nb_summaries_per_category = max(
                        1, missing_summaries_nb // len(summaries_by_priority) + 20
                    )
                else:
                    max_nb_summaries_per_category = max(
                        1, missing_summaries_nb // len(summaries_by_priority) - 20
                    )
                for _ in range(max_nb_summaries_per_category):
                    if counter[i] < len(summaries):
                        final_summary = summaries[counter[i]].copy()
                        final_summary.addition_reason = summary_categories[i]
                        final_summaries.append(final_summary)
                        counter[i] += 1
                    else:
                        break
            missing_summaries_nb = filtered_summaries_nb - len(final_summaries)
        print(counter)
        print("Nb of final summaries:", len(final_summaries))
        return final_summaries

    def export(self, output_dir: str, summaries: List[FinalSummary]) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "filtered_summaries.json"), "w") as file:
            json.dump(
                [json.loads(summary.model_dump_json()) for summary in summaries], file
            )
