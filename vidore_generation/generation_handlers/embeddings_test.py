import hashlib
import json
import os

import click
import torch
import torch.multiprocessing as mp
from colpali_engine.models import ColQwen2, ColQwen2Processor
from datasets import load_dataset
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

BATCH_SIZE = 64
BIG_BATCH = BATCH_SIZE * 10
GPU_NB = 4
SAVE_PATH = "experiments/image_dataset_creation/colqwen2_embeddings_personal"


def get_hash_from_pil_image(image: Image.Image) -> str:
    image_bytes = image.tobytes()
    return {"image_hash": str(hashlib.sha256(image_bytes).hexdigest())}


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def pad_tensors(tensors):
    max_number_of_tokens = max([x.size(1) for x in tensors])
    padded_tensors = []
    for tensor in tensors:
        if tensor.size(1) < max_number_of_tokens:
            padding = torch.zeros(
                tensor.size(0),
                max_number_of_tokens - tensor.size(1),
                *tensor.shape[2:],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded_tensors.append(torch.cat([tensor, padding], dim=1))
        else:
            padded_tensors.append(tensor)
    return padded_tensors


def process_batch(rank, world_size, dataset_path, subset):
    # Setup process group
    setup(rank, world_size)

    # Load dataset
    dataset_name = dataset_path.replace("/", "_") + "_" + subset.replace("/", "_")
    print(f"Running inference for {dataset_name}")

    print(f"Loading dataset from {dataset_name}")
    queries_subset = subset.replace("corpus", "queries")
    corpus = load_dataset(dataset_path, subset)["train"]
    queries = load_dataset(dataset_path, queries_subset)["train"]
    queries_test = load_dataset(dataset_path, queries_subset)["test"]

    print(f"Corpus size: {len(corpus)}")
    print(f"Queries size: {len(queries)}")

    # Initialize model on this GPU
    print(f"Loading model on GPU {rank}")
    model = ColQwen2.from_pretrained(
        "/linkhome/rech/genrce01/ufk84kd/scratch/models/colqwen2-v1.0",
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{rank}",
        attn_implementation="flash_attention_2"
        if is_flash_attn_2_available()
        else None,
    ).eval()

    processor = ColQwen2Processor.from_pretrained(
        "/linkhome/rech/genrce01/ufk84kd/scratch/models/colqwen2-v1.0"
    )

    # Split dataset for this GPU
    corpus_size = len(corpus)
    queries_size = len(queries)
    queries_test_size = len(queries_test)
    samples_per_gpu_for_corpus = corpus_size // world_size
    samples_per_gpu_for_queries = queries_size // world_size
    samples_per_gpu_for_queries_test = queries_test_size // world_size
    start_idx_for_corpus = rank * samples_per_gpu_for_corpus
    start_idx_for_queries = rank * samples_per_gpu_for_queries
    end_idx_for_corpus = (
        start_idx_for_corpus + samples_per_gpu_for_corpus
        if rank < world_size - 1
        else corpus_size
    )
    end_idx_for_queries = (
        start_idx_for_queries + samples_per_gpu_for_queries
        if rank < world_size - 1
        else queries_size
    )
    start_idx_for_queries_test = rank * samples_per_gpu_for_queries_test
    end_idx_for_queries_test = (
        start_idx_for_queries_test + samples_per_gpu_for_queries_test
        if rank < world_size - 1
        else queries_test_size
    )
    # Process batches for this GPU
    print(f"Running inference on GPU {rank}")
    with torch.no_grad():
        image_hashes_list = []
        image_embeddings_list = []
        query_embeddings_list = []
        test_query_embeddings_list = []

        # for i in tqdm(range(start_idx_for_corpus, end_idx_for_corpus, BATCH_SIZE), desc=f"Processing corpus with GPU {rank}"):
        #     # if i + BIG_BATCH >= end_idx_for_corpus:
        #     tensor_save_path = f"{SAVE_PATH}/{dataset_name}_image_embeddings_gpu{rank}_{start_idx_for_corpus+i}.safetensors"
        #     print(f"At index {i}, tensor_save_path: {tensor_save_path}")
        #     if not os.path.exists(tensor_save_path):
        #         print(f"Processing corpus with GPU {rank} at index {i}")
        #         images = [x["image"] for x in corpus.select(range(i, min(i + BATCH_SIZE, end_idx_for_corpus)))]
        #         image_hashes = [get_hash_from_pil_image(x) for x in images]
        #         batch_images = processor.process_images(images).to(model.device)
        #         image_embeddings = model(**batch_images)
        #         image_embeddings_list.append(image_embeddings)
        #         image_hashes_list.append(image_hashes)
        #         if (i - start_idx_for_corpus) % BIG_BATCH == 0 and i > start_idx_for_corpus:
        #             gpu_image_embeddings = torch.cat(pad_tensors(image_embeddings_list), dim=0)
        #             save_file({"image_embeddings": gpu_image_embeddings.cpu()}, tensor_save_path)
        #             with open(f"{SAVE_PATH}/{dataset_name}_image_hashes_gpu{rank}_{start_idx_for_corpus+i}.json", "w") as f:
        #                 json.dump({"image_hashes": image_hashes_list}, f)
        #             image_embeddings_list = []
        #             image_hashes_list = []

        # if image_embeddings_list:
        #     tensor_save_path = f"{SAVE_PATH}/{dataset_name}_image_embeddings_gpu{rank}_final.safetensors"
        #     if not os.path.exists(tensor_save_path):
        #         print(f"Saving final embeddings for GPU {rank}")
        #         gpu_image_embeddings = torch.cat(pad_tensors(image_embeddings_list), dim=0)

        #         save_file({"image_embeddings": gpu_image_embeddings.cpu()}, tensor_save_path)
        #         with open(f"{SAVE_PATH}/{dataset_name}_image_hashes_gpu{rank}_final.json", "w") as f:
        #             json.dump({"image_hashes": image_hashes_list}, f)

        for i in tqdm(
            range(start_idx_for_queries, end_idx_for_queries, BATCH_SIZE),
            desc=f"Processing train queries with GPU {rank}",
        ):
            # if i + BIG_BATCH >= end_idx_for_queries:
            tensor_save_path = f"{SAVE_PATH}/{dataset_name}_query_embeddings_gpu{rank}_{start_idx_for_queries + i}.safetensors"
            if not os.path.exists(tensor_save_path):
                queries_list = [
                    x["original_query"]
                    for x in queries.select(
                        range(i, min(i + BATCH_SIZE, end_idx_for_queries))
                    )
                ]
                batch_queries = processor.process_queries(queries_list).to(model.device)
                query_embeddings = model(**batch_queries)
                query_embeddings_list.append(query_embeddings)

                # Save intermediate results
                if (
                    i - start_idx_for_queries
                ) % BIG_BATCH == 0 and i > start_idx_for_queries:
                    gpu_query_embeddings = torch.cat(
                        pad_tensors(query_embeddings_list), dim=0
                    )
                    save_file(
                        {"query_embeddings": gpu_query_embeddings.cpu()},
                        tensor_save_path,
                    )
                    with open(
                        f"{SAVE_PATH}/{dataset_name}_queries_gpu{rank}_{start_idx_for_queries + i}.json",
                        "w",
                    ) as f:
                        json.dump({"queries": queries_list}, f)

                    # Clear lists to free memory
                    query_embeddings_list = []

        # Save any remaining embeddings
        if query_embeddings_list:
            tensor_save_path = f"{SAVE_PATH}/{dataset_name}_query_embeddings_gpu{rank}_final.safetensors"
            if not os.path.exists(tensor_save_path):
                print(f"Saving final embeddings for GPU {rank}")
                gpu_query_embeddings = torch.cat(
                    pad_tensors(query_embeddings_list), dim=0
                )

                save_file(
                    {"query_embeddings": gpu_query_embeddings.cpu()}, tensor_save_path
                )
                with open(
                    f"{SAVE_PATH}/{dataset_name}_queries_gpu{rank}_final.json", "w"
                ) as f:
                    json.dump({"queries": queries_list}, f)

        for i in tqdm(
            range(start_idx_for_queries_test, end_idx_for_queries_test, BATCH_SIZE),
            desc=f"Processing test queries with GPU {rank}",
        ):
            tensor_save_path = f"{SAVE_PATH}/{dataset_name}_query_test_embeddings_gpu{rank}_{start_idx_for_queries_test + i}.safetensors"
            if not os.path.exists(tensor_save_path):
                queries_list = [
                    x["original_query"]
                    for x in queries_test.select(
                        range(i, min(i + BATCH_SIZE, end_idx_for_queries_test))
                    )
                ]
                batch_queries = processor.process_queries(queries_list).to(model.device)
                query_embeddings = model(**batch_queries)
                test_query_embeddings_list.append(query_embeddings)
                if (
                    i - start_idx_for_queries_test
                ) % BIG_BATCH == 0 and i > start_idx_for_queries_test:
                    gpu_query_embeddings = torch.cat(
                        pad_tensors(query_embeddings_list), dim=0
                    )
                    save_file(
                        {"query_embeddings": gpu_query_embeddings.cpu()},
                        tensor_save_path,
                    )
                    with open(
                        f"{SAVE_PATH}/{dataset_name}_queries_test_gpu{rank}_{start_idx_for_queries_test + i}.json",
                        "w",
                    ) as f:
                        json.dump({"queries": queries_list}, f)
                    test_query_embeddings_list = []

        if test_query_embeddings_list:
            tensor_save_path = f"{SAVE_PATH}/{dataset_name}_query_test_embeddings_gpu{rank}_final.safetensors"
            if not os.path.exists(tensor_save_path):
                print(f"Saving final embeddings for GPU {rank}")
                gpu_query_embeddings = torch.cat(
                    pad_tensors(query_embeddings_list), dim=0
                )
                save_file(
                    {"query_embeddings": gpu_query_embeddings.cpu()}, tensor_save_path
                )

    # Cleanup
    cleanup()


@click.command()
@click.argument("dataset_path", type=str)
@click.argument("subset", type=str)
def main(dataset_path: str, subset: str):
    world_size = GPU_NB
    mp.spawn(
        process_batch,
        args=(world_size, dataset_path, subset),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    # Launch processes for each GPU
    main()
