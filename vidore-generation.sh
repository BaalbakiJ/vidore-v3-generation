#!/usr/bin/env bash
# Run the full ViDoRe generation pipeline.
#
# Usage:
#   bash vidore-generation.sh <dataset_name>
#
# Example:
#   bash vidore-generation.sh my_dataset
#
# Assumes:
#   - PDFs are in <dataset_name>/pdfs/
#   - Config file is at configs/<dataset_name>.yaml

set -euo pipefail

# ── Arguments ────────────────────────────────────────────────────────────────

DATASET_NAME="${1:-}"

if [[ -z "$DATASET_NAME" ]]; then
    echo "Usage: bash $0 <dataset_name>"
    exit 1
fi

CONFIG_FILE="configs/${DATASET_NAME}.yaml"
PDFS_FOLDER="${DATASET_NAME}/pdfs"
DATASET_FOLDER="${DATASET_NAME}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: config file '$CONFIG_FILE' not found."
    exit 1
fi

if [[ ! -d "$PDFS_FOLDER" ]]; then
    echo "Error: PDFs folder '$PDFS_FOLDER' does not exist."
    exit 1
fi

PDF_COUNT=$(find "$PDFS_FOLDER" -maxdepth 1 -name "*.pdf" | wc -l | tr -d ' ')
if [[ "$PDF_COUNT" -eq 0 ]]; then
    echo "Error: no PDF files found in '$PDFS_FOLDER'."
    exit 1
fi
echo "Found $PDF_COUNT PDF file(s) in '$PDFS_FOLDER'."

# ── Step 0 — Normalize document names ────────────────────────────────────────

echo ""
echo "=== Step 0/5: Normalizing document names ==="
vidore-generation normalize-docs "$CONFIG_FILE"

# ── Step 1 — Extract text from PDFs ──────────────────────────────────────────

echo ""
echo "=== Step 1/5: Extracting text from PDFs ==="
vidore-generation extract-text-from-pdfs "$PDFS_FOLDER"

MARKDOWN_COUNT=$(find "$DATASET_FOLDER/markdowns" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
if [[ "$MARKDOWN_COUNT" -eq 0 ]]; then
    echo "Error: no markdown files found in '$DATASET_FOLDER/markdowns' after extraction."
    exit 1
fi
echo "Extracted $MARKDOWN_COUNT markdown file(s)."

# ── Step 2 — Generate summaries ───────────────────────────────────────────────

echo ""
echo "=== Step 2/5: Generating summaries ==="
vidore-generation llm --config "$CONFIG_FILE"

FILTERED_SUMMARIES="$DATASET_FOLDER/filtered_summaries/filtered_summaries.json"
if [[ ! -f "$FILTERED_SUMMARIES" ]]; then
    echo "Error: filtered summaries not found at '$FILTERED_SUMMARIES'."
    exit 1
fi
echo "Filtered summaries written to '$FILTERED_SUMMARIES'."

# ── Step 3 — Generate queries ─────────────────────────────────────────────────

echo ""
echo "=== Step 3/5: Generating queries ==="
vidore-generation generate-queries-vidore-juicer \
    "$FILTERED_SUMMARIES" \
    "$CONFIG_FILE"

# ── Step 4 — Postprocess queries ──────────────────────────────────────────────

echo ""
echo "=== Step 4/5: Postprocessing queries ==="
vidore-generation postprocess-queries --config "$CONFIG_FILE"

# ── Done ──────────────────────────────────────────────────────────────────────

echo ""
echo "=== Done! ==="
echo "Final queries are in $DATASET_FOLDER/queries/"
