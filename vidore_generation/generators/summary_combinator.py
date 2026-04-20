import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from tqdm import tqdm, trange
from umap import UMAP

from vidore_generation.dtos import (
    CombinedSummary,
    Document,
    FinalSummary,
    IndexedSummary,
    Prompt,
)
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import (
    CombinedSummaryGeneration,
    PairSummaryCombinations,
)
from vidore_generation.generators.base_generator import BaseGenerator


class SummaryCombinator(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        combination_iteration_nb: int = 20,
        sampling_multi_doc_ratio: float = 0.5,
        save_folder: str = None,
        debug: bool = False,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.pair_template = self.environment.get_template("pair_finding.j2")
        self.triplet_template = self.environment.get_template("triplet_finding.j2")
        self.combination_template = self.environment.get_template("combination.j2")
        self.generation_handler = generation_handler
        self.logger = logger
        self.combination_iteration_nb = combination_iteration_nb
        self.sampling_multi_doc_ratio = sampling_multi_doc_ratio
        self.save_folder = save_folder
        # Clustering attributes
        self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        self.debug = debug
        self.multi_doc_pair_number = 0
        self.multi_doc_triplet_number = 0
        self.language = language
        print("")

    def sample_pair(
        self,
        indeces_per_filename: Dict[str, List[int]],
    ):
        is_multi_doc = random.random() < self.sampling_multi_doc_ratio
        if is_multi_doc:
            filenames = random.sample(list(indeces_per_filename.keys()), k=2)
            pair = [
                random.choice(indeces_per_filename[filenames[0]]),
                random.choice(indeces_per_filename[filenames[1]]),
            ]
            return pair
        else:
            filename = random.choice(
                [
                    filename
                    for filename in indeces_per_filename.keys()
                    if len(indeces_per_filename[filename]) > 1
                ]
            )
            pair = random.sample(indeces_per_filename[filename], k=2)
            return pair

    def sample_triplet(
        self,
        indeces_per_filename: Dict[str, List[int]],
    ):
        is_multi_doc = random.random() < self.sampling_multi_doc_ratio
        if is_multi_doc:
            is_triple_doc = random.random() < 0.7
            if is_triple_doc:
                filenames = random.sample(list(indeces_per_filename.keys()), k=3)
                triplet = [
                    random.choice(indeces_per_filename[filenames[0]]),
                    random.choice(indeces_per_filename[filenames[1]]),
                    random.choice(indeces_per_filename[filenames[2]]),
                ]
            else:
                filenames = random.sample(list(indeces_per_filename.keys()), k=2)
                filenames.append(random.choice(filenames))
                triplet = [
                    random.choice(indeces_per_filename[filenames[0]]),
                    random.choice(indeces_per_filename[filenames[1]]),
                ]
                triplet.append(
                    random.choice(
                        [
                            x
                            for x in indeces_per_filename[filenames[2]]
                            if x not in triplet
                        ]
                    )
                )
            return triplet
        else:
            filename = random.choice(
                [
                    filename
                    for filename in indeces_per_filename.keys()
                    if len(indeces_per_filename[filename]) > 2
                ]
            )
            triplet = random.sample(indeces_per_filename[filename], k=3)
            return triplet

    def sample_pair_or_triplet(
        self,
        indeces_per_filename: Dict[str, List[int]],
        is_pair: bool,
    ):
        num_retries = 0
        while num_retries < 3:
            try:
                if is_pair:
                    return sorted(self.sample_pair(indeces_per_filename))
                else:
                    return sorted(self.sample_triplet(indeces_per_filename))
            except Exception as e:
                if "Cannot choose from an empty sequence" in str(
                    e
                ) or "Sample larger than population or is negative" in str(
                    e
                ) or "list index out of range" in str(e):
                    num_retries += 1
                else:
                    raise e
        return None

    def get_all_combinations(
        self,
        summaries_list: List[List[FinalSummary]],
        random_seeds: List[int],
    ) -> List[PairSummaryCombinations]:
        all_summaries = []
        all_document_ids = []
        all_summary_ids = []
        for i, summaries in enumerate(summaries_list):
            all_summaries.extend([summary for summary in summaries])
            all_document_ids.extend([i] * len(summaries))
            all_summary_ids.extend(list(range(len(summaries))))

        embeddings_folder = os.path.join(self.save_folder, "embeddings")
        os.makedirs(embeddings_folder, exist_ok=True)
        embeddings_path = os.path.join(embeddings_folder, "all_embeddings.npy")
        if self.save_folder is not None and os.path.exists(embeddings_path):
            all_embeddings = np.load(embeddings_path)
        else:
            all_embeddings = []
            batch_size = 4
            for i in trange(
                0, len(all_summaries), batch_size, desc="Encoding summaries"
            ):
                if i + 1 + batch_size >= len(all_summaries):
                    end_index = len(all_summaries)
                else:
                    end_index = i + batch_size
                all_embeddings.append(
                    self.embedding_model.encode(
                        [x.summary for x in all_summaries[i:end_index]]
                    )
                )
                if i + batch_size + 1 > len(all_summaries):
                    break
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            assert all_embeddings.shape[0] == len(all_summaries)
            np.save(embeddings_path, all_embeddings)

        print(f"Embeddings shape: {all_embeddings.shape}")
        assert all_embeddings.shape[0] == len(all_summaries)

        pairs = set()
        triplets = set()
        pair_doc_counter = Counter()
        triplet_doc_counter = Counter()
        for random_seed in tqdm(random_seeds, desc="Looking for pairs and triplets"):
            random.seed(random_seed)
            n_components = random.randint(5, 10)
            # min_cluster_size = random.randint(3, 10)
            min_cluster_size = 2
            dim_red_model = UMAP(
                n_neighbors=5,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                random_state=random_seed,
                transform_seed=random_seed,
            )
            cluster_model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            reduced_vectors = dim_red_model.fit_transform(all_embeddings)
            reduced_vectors = reduced_vectors - reduced_vectors.mean(axis=0)
            cluster_model.fit(reduced_vectors)
            labels = cluster_model.labels_.tolist()
            nb_labels = len(set([label for label in labels if label >= 0]))
            for i in range(nb_labels):
                label_indices = [j for j in range(len(all_summaries)) if labels[j] == i]
                label_doc_ids = [all_document_ids[j] for j in label_indices]
                indeces_per_filename = defaultdict(list)
                for j, filename in enumerate(label_doc_ids):
                    indeces_per_filename[filename].append(label_indices[j])
                pair = self.sample_pair_or_triplet(indeces_per_filename, is_pair=True)
                if pair is not None:
                    assert len(set(pair)) == 2
                    if (
                        len(
                            set(
                                [
                                    all_summaries[pair[0]].filenames[0],
                                    all_summaries[pair[1]].filenames[0],
                                ]
                            )
                        )
                        == 2
                    ):
                        self.multi_doc_pair_number += 1
                    pairs.add(tuple(pair))
                    pair_doc_counter[all_summaries[pair[0]].filenames[0]] += 1
                    pair_doc_counter[all_summaries[pair[1]].filenames[0]] += 1
                triplet = self.sample_pair_or_triplet(
                    indeces_per_filename, is_pair=False
                )
                if triplet is not None:
                    assert len(set(triplet)) == 3
                    if (
                        len(
                            set(
                                [
                                    all_summaries[triplet[0]].filenames[0],
                                    all_summaries[triplet[1]].filenames[0],
                                    all_summaries[triplet[2]].filenames[0],
                                ]
                            )
                        )
                        >= 2
                    ):
                        self.multi_doc_triplet_number += 1
                    triplets.add(tuple(triplet))
                    triplet_doc_counter[all_summaries[triplet[0]].filenames[0]] += 1
                    triplet_doc_counter[all_summaries[triplet[1]].filenames[0]] += 1
                    triplet_doc_counter[all_summaries[triplet[2]].filenames[0]] += 1
            print(
                f"Multi doc percentage: {(self.multi_doc_pair_number + self.multi_doc_triplet_number) / (len(pairs) + len(triplets))}"
            )
            print(f"Found {len(pairs)} pairs and {len(triplets)} triplets")
        if self.debug:
            random_pairs = random.sample(list(pairs), k=5)
            random_triplets = random.sample(list(triplets), k=5)
            for pair in random_pairs:
                print(f"**Summary 1**, {all_summaries[pair[0]].filenames[0]}:")
                print(all_summaries[pair[0]].summary)
                print(f"**Summary 2**, {all_summaries[pair[1]].filenames[0]}:")
                print(all_summaries[pair[1]].summary)
                print("-" * 100)
            print("=" * 100)
            print("=" * 100)
            for triplet in random_triplets:
                print(f"**Summary 1**, {all_summaries[triplet[0]].filenames[0]}:")
                print(all_summaries[triplet[0]].summary)
                print(f"**Summary 2**, {all_summaries[triplet[1]].filenames[0]}:")
                print(all_summaries[triplet[1]].summary)
                print(f"**Summary 3**, {all_summaries[triplet[2]].filenames[0]}:")
                print(all_summaries[triplet[2]].summary)
                print("-" * 100)
            print("=" * 100)
            print("Number of pair docs:", len(pair_doc_counter))
            print(pair_doc_counter)
            print("Number of triplet docs:", len(triplet_doc_counter))
            print(triplet_doc_counter)
            print(
                f"Multi doc pair percentage: {self.multi_doc_pair_number / len(pairs)}"
            )
            print(
                f"Multi doc triplet percentage: {self.multi_doc_triplet_number / len(triplets)}"
            )
            print(
                f"Multi doc percentage: {(self.multi_doc_pair_number + self.multi_doc_triplet_number) / (len(pairs) + len(triplets))}"
            )
            print(f"Found {len(pairs)} pairs and {len(triplets)} triplets")
        final_pairs = []
        for pair in pairs:
            final_pairs.append(
                [
                    all_summaries[pair[0]],
                    all_summaries[pair[1]],
                ]
            )
        final_triplets = []
        for triplet in triplets:
            final_triplets.append(
                [
                    all_summaries[triplet[0]],
                    all_summaries[triplet[1]],
                    all_summaries[triplet[2]],
                ]
            )
        return final_pairs, final_triplets

    def combine_summaries(
        self,
        documents: List[Document],
        summaries: List[FinalSummary],
        random_seeds: List[int] = [42],
    ) -> List[CombinedSummary]:
        # Initialize the lists
        summaries_list = []
        for doc in documents:
            summaries_list.append(
                [
                    summary
                    for summary in summaries
                    if summary.filenames[0] == doc.filename
                ]
            )

        pairs, triplets = self.get_all_combinations(summaries_list, random_seeds)

        # Rewrite the summmaries so that they are better written
        # summary_pairs = []
        # for pair in pairs:
        #     summary_pairs.append(
        #         {
        #             "summaries": pair,
        #         }
        #     )
        # summary_triplets = []
        # for triplet in triplets:
        #     summary_triplets.append(
        #         {
        #             "summaries": triplet,
        #         }
        #     )
        generated_combined_summaries_from_pairs = (
            self.generation_handler.generate_multiple_samples(
                [
                    Prompt(
                        messages=[
                            {
                                "role": "user",
                                "content": self.create_prompt(
                                    {
                                        "summaries": [x.summary for x in pair],
                                        "language": self.language,
                                    },
                                    self.combination_template,
                                ),
                            }
                        ],
                        arguments={"pydantic_schema": CombinedSummaryGeneration},
                    )
                    for pair in pairs
                ],
                desc="Generating combined summaries from pairs",
            )
        )
        generated_combined_summaries_from_triplets = (
            self.generation_handler.generate_multiple_samples(
                [
                    Prompt(
                        messages=[
                            {
                                "role": "user",
                                "content": self.create_prompt(
                                    {
                                        "summaries": [x.summary for x in triplet],
                                        "language": self.language,
                                    },
                                    self.combination_template,
                                ),
                            }
                        ],
                        arguments={"pydantic_schema": CombinedSummaryGeneration},
                    )
                    for triplet in triplets
                ],
                desc="Generating combined summaries from triplets",
            )
        )
        combined_summaries = []
        for pair, generated_combined_summary in zip(
            pairs, generated_combined_summaries_from_pairs
        ):
            combined_summaries.append(
                CombinedSummary(
                    summaries=[
                        IndexedSummary(
                            summary=pair[0].summary,
                            document_id=pair[0].document_ids[0],
                            filename=pair[0].filenames[0],
                            page_numbers=pair[0].page_numbers[0],
                            summary_id=pair[0].id,
                        ),
                        IndexedSummary(
                            summary=pair[1].summary,
                            document_id=pair[1].document_ids[0],
                            filename=pair[1].filenames[0],
                            page_numbers=pair[1].page_numbers[0],
                            summary_id=pair[1].id,
                        ),
                    ],
                    combined_summary=generated_combined_summary.combined_summary,
                )
            )
        for triplet, generated_combined_summary in zip(
            triplets, generated_combined_summaries_from_triplets
        ):
            combined_summaries.append(
                CombinedSummary(
                    summaries=[
                        IndexedSummary(
                            summary=triplet[0].summary,
                            document_id=triplet[0].document_ids[0],
                            filename=triplet[0].filenames[0],
                            page_numbers=triplet[0].page_numbers[0],
                            summary_id=triplet[0].id,
                        ),
                        IndexedSummary(
                            summary=triplet[1].summary,
                            document_id=triplet[1].document_ids[0],
                            filename=triplet[1].filenames[0],
                            page_numbers=triplet[1].page_numbers[0],
                            summary_id=triplet[1].id,
                        ),
                        IndexedSummary(
                            summary=triplet[2].summary,
                            document_id=triplet[2].document_ids[0],
                            filename=triplet[2].filenames[0],
                            page_numbers=triplet[2].page_numbers[0],
                            summary_id=triplet[2].id,
                        ),
                    ],
                    combined_summary=generated_combined_summary.combined_summary,
                )
            )
        return combined_summaries

    def export(self, output_dir: Path, combined_summaries: List[CombinedSummary]):
        os.makedirs(os.path.join(output_dir), exist_ok=True)
        with open(os.path.join(output_dir, "combined_summaries.json"), "w") as f:
            json.dump(
                [
                    json.loads(combined_summary.model_dump_json())
                    for combined_summary in combined_summaries
                ],
                f,
                indent=4,
            )

    def import_combined_summaries(self, output_dir: Path) -> List[CombinedSummary]:
        with open(os.path.join(output_dir, "combined_summaries.json"), "r") as f:
            data = json.load(f)
        return [FinalSummary(**item) for item in data]
