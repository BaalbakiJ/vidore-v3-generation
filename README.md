# ViDoRe Generation

Generate synthetic queries from a PDF document corpus for evaluating image retrieval models.

- 📑  [ColPali (ViDoRe V1)](https://arxiv.org/abs/2407.01449)
- 📑 [ViDoRe V2](https://arxiv.org/abs/2505.17166)
- 📑 [ViDoRe V3](https://arxiv.org/abs/2601.08620)

## Install

```bash
uv venv --python 3.10
uv sync
```

Set up your API keys by copying `.env.dist` to `.env` and filling in the relevant keys:

```bash
cp .env.dist .env
```

## Quick start

The full pipeline takes a folder of PDFs and produces a `final_queries.json` file ready for use. It runs in 4 steps:

```mermaid
graph LR
    A[/PDFs/] --> B[extract text]
    B --> C[generate summaries] --> D[generate queries] --> E[postprocess queries]
    E --> F[\final_queries.json\]
```

### Step 1 — Set up your data folder

Create a folder anywhere on disk. Inside it, create a `pdfs/` subfolder and put your PDF files there:

```
my_dataset/
└── pdfs/
    ├── document_1.pdf
    ├── document_2.pdf
    └── ...
```

### Step 2 — Write a config file

Create a YAML file (e.g. `configs/my_dataset.yaml`). Here is a minimal working example:

```yaml
# Unique name for your dataset — used to name all output folders and files
dataset_name: "my_dataset"

# Path to the folder that CONTAINS your dataset folder (i.e. the parent of my_dataset/)
documents_dir: "."

# LLM provider settings
llm_provider:
  # Any litellm-compatible model string, e.g.:
  #   "openai/gpt-5-nano"
  #   "fireworks_ai/kimi-k2p5"
  #   "anthropic/claude-3-5-haiku-20241022"
  lm_model_name: "openai/gpt-5-nano"

  # Optional: use a different model specifically for query generation and judging
  # Defaults to lm_model_name if not set
  # query_generation_model_name: "openai/gpt-5-nano"
  # judge_model_name: "openai/gpt-5-nano"

  # Extra parameters forwarded to the LLM (provider-specific, all optional)
  lm_extra_kwargs:
    temperature: 0.7
    top_p: 0.8

# Describe the target user who will search this document corpus.
# The more specific, the better the generated queries.
persona: "A student looking for information about physics."

# Language of the generated queries ("english", "french", "spanish", etc.)
language: "english"

# Target number of summaries to keep after filtering.
# A good starting point: 5–10× the number of PDFs.
filtered_summaries_nb: 50

# Number of multi-document summary combination iterations.
# Higher = more cross-document queries. Good default: 10–20.
combination_iteration_nb: 15

# Fraction of summaries that span multiple documents (0.0–1.0).
sampling_multi_doc_ratio: 0.5

# Print verbose LLM outputs during generation
debug: false
```

All available config fields with their defaults are documented in `configs/example.yaml`.

> [!WARNING]
> ### Automated script
> **It is strongly preferable to run each step individually (see below)**. Each step produces intermediate outputs worth inspecting before proceeding. Mistakes caught early save significant API costs.
>
> After setting up your documents, you can run everything at once using this convenience script:
>
> ```bash
> bash vidore-generation.sh my_dataset
> ```

### Step 3 — Extract text from PDFs

Warning : If you have big documents, it can take a while.

```bash
vidore-generation extract-text-from-pdfs my_dataset/pdfs
```

This creates `my_dataset/markdowns/` with one `.md` file per PDF.

Note that markdown extraction uses fireworks by default and kimi-k2.5. If you want to change that, you can modify the paths in the `parse_pdf` function of `vidore_generation/pdf_parsing/extract_text_from_pdfs`

Optionally verify the extraction succeeded (page counts match):

```bash
vidore-generation check-extractions my_dataset
```

### Step 4 — Generate summaries

```bash
vidore-generation llm --config configs/my_dataset.yaml
```

This is the main LLM step. It reads the markdowns, generates summaries per document section, combines them across documents, judges their quality, and writes the best ones to `my_dataset/filtered_summaries/filtered_summaries.json`.

Output folders created under `my_dataset/`:

| Folder | Contents |
|---|---|
| `descriptions/` | One-paragraph description of each document |
| `sections/` | Extracted sections per document |
| `summaries/` | Per-section summaries |
| `combined_summaries/` | Cross-document summaries |
| `judgments/` | Quality scores for each summary |
| `filtered_summaries/` | The final selection used for query generation |

### Optional visual summaries from rendered pages

If `my_dataset/imgs/` already contains rendered page images grouped by document, you can generate ViDoRe-compatible summaries directly from those images:

```bash
vidore-generation visual-summaries --config configs/my_dataset.yaml
```

This command does not run Docling and does not require `markdowns/`. It reads `imgs/<document_name>/<document_name>_<page_number>.png` (also `.jpg` and `.jpeg`) and writes `visual_summaries/visual_summaries.json`, per-document files under `summaries/`, and `filtered_summaries/filtered_summaries.json` for the existing query-generation step.

Page-numbering convention: in the visual pipeline, `FinalSummary.page_numbers` stores 0-based rendered image page numbers. These match `imgs/<document>/<document>_<page>.png` and `page_manifest.image_page_number`. In `page_manifest.jsonl`, `page_number` is 1-based for human/PDF display.

To optionally skip table-of-contents pages in this image-based flow, first create page metadata:

```bash
vidore-generation build-page-manifest --config configs/my_dataset.yaml
```

Then render images with manifest exclusions:

```bash
vidore-generation create-images my_dataset/pdfs --respect-page-manifest
```

Set `visual_summary.respect_page_manifest: true` to make `visual-summaries` skip already-rendered images excluded by `page_manifest.jsonl`.

Set `visual_summary.use_visual_document_descriptions: true` to generate one VLM-based description per document from page images. These are written to `descriptions/<filename>.json` and used as context for visual summaries; `visual_summary.document_description` remains the fallback context.

Set `visual_summary.use_visual_combined_summaries: true` to generate combined summaries from visual summaries using the existing `SummaryCombinator`. Combined summaries are written to `combined_summaries/combined_summaries.json`; this is the visual analogue of the original markdown `llm` combined-summary stage.

Set `visual_summary.use_visual_summary_judging: true` to judge candidate visual summaries using the existing `Judge`. Judgments are written to `judgments/judgments.json`, and `filtered_summaries/filtered_summaries.json` is selected from judgment scores instead of simple first-N slicing. This is the visual analogue of the original markdown `llm` judging/filtering stage.

The visual filter prioritizes multi-document combined summaries, same-document combined summaries, then single visual summaries. QREL export and query generation remain separate steps.

### Step 5 — Generate queries

```bash
vidore-generation generate-queries-vidore-juicer \
  my_dataset/filtered_summaries/filtered_summaries.json \
  configs/my_dataset.yaml
```

This generates queries from each filtered summary, judges their quality, and writes the survivors to `my_dataset/queries/vidore_juicer_my_dataset_queries.json`.

### Step 6 — Postprocess queries

```bash
vidore-generation postprocess-queries --config configs/my_dataset.yaml
```

Filters and rephrases the queries, then writes the final output to `my_dataset/queries/final_my_dataset_queries.json`.

Each entry in the file looks like:

```json
{
  "query": "What is the relationship between energy and mass?",
  "generation_process": "vidore_juicer_rephrased",
  "original_query": "How does E=mc² relate energy and mass?",
  "document_ids": ["..."],
  "filenames": ["document_1"],
  "page_numbers": [[12, 13]]
}
```

---

## Optional steps

### Normalize document names

If your PDF filenames contain spaces, accents, or special characters, normalize them first (run this before step 3):

```bash
vidore-generation normalize-docs configs/my_dataset.yaml
```

---

## Supported LLM providers

Any model supported by [litellm](https://docs.litellm.ai/docs/providers) works. Common examples:

| Provider | Model string |
|---|---|
| OpenAI | `openai/gpt-5-nano`, `openai/gpt-4o` |
| Fireworks | `fireworks_ai/kimi-k2p5`, `fireworks_ai/qwen3-235b-a22b-instruct-2507` |
| Anthropic | `anthropic/claude-3-5-haiku-20241022` |

Provider-specific parameters (e.g. `top_k` for Fireworks) can be set in `lm_extra_kwargs` — unsupported parameters are automatically dropped for providers that don't support them.

Set the corresponding API key in your `.env` file (see `.env.dist` for the full list).

### Bedrock usage and concurrency

Set `llm_provider.provider: "bedrock"` to use Amazon Bedrock Converse through the centralized Bedrock handler. Token usage is read from each Converse response and logged per batch. Cost is reported only as an estimate, and only when pricing is configured in YAML; prices are not hardcoded because provider pricing can change.

Lower `bedrock_max_concurrency` for models that throttle under parallel load, such as Pixtral. Set `bedrock_usage_log_path` to append one JSONL batch summary per generation batch without logging prompt or document contents.

```yaml
llm_provider:
  provider: "bedrock"
  aws_region: "eu-central-1"
  lm_model_name: "eu.amazon.nova-2-lite-v1:0"
  vl_model_name: "eu.mistral.pixtral-large-2502-v1:0"
  bedrock_max_concurrency: 2
  bedrock_retry_count: 6
  bedrock_retry_initial_sleep_seconds: 20.0
  bedrock_retry_backoff_multiplier: 2.0
  bedrock_retry_max_sleep_seconds: 180.0
  bedrock_usage_log_path: "datascience_dataset_pixtral/logs/bedrock_usage.jsonl"
  bedrock_pricing:
    eu.amazon.nova-2-lite-v1:0:
      input_per_1k_tokens_usd: null
      output_per_1k_tokens_usd: null
    eu.mistral.pixtral-large-2502-v1:0:
      input_per_1k_tokens_usd: null
      output_per_1k_tokens_usd: null
```

---

## Acknowledgement
The core code for this repo was contributed by [António Loison](https://github.com/antonioloison) during his work at Illuin Technology. We thank him for his contributions and helping shape the ViDoRe v3 benchmark.
