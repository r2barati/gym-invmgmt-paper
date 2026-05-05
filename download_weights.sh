#!/usr/bin/env bash
# Download pre-trained gym-invmgmt agent checkpoints from Hugging Face.

set -euo pipefail

MODELS_DIR="data/models"
HF_REPO_ID="${HF_REPO_ID:-rezabarati/gym-invmgmt-weights}"
HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"

mkdir -p "$MODELS_DIR"

echo "Downloading pre-trained agent weights from https://huggingface.co/datasets/${HF_REPO_ID}"
echo "Target: ${MODELS_DIR}"

python3 - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print(
        "Missing dependency: huggingface_hub. Install with "
        "`python3 -m pip install huggingface_hub` and rerun.",
        file=sys.stderr,
    )
    raise SystemExit(1)

repo_id = os.environ.get("HF_REPO_ID", "rezabarati/gym-invmgmt-weights")
repo_type = os.environ.get("HF_REPO_TYPE", "dataset")
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    local_dir=".",
    allow_patterns=["data/models/*", "models_manifest.json"],
    token=token,
)
PY

echo "Done. Model checkpoints are available in ${MODELS_DIR}."
echo "Note: optional third-party LLM GGUF weights are not re-hosted here."
