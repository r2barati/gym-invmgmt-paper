#!/usr/bin/env python3
"""Download public retail datasets used to validate dataset adapters.

The benchmark M5 trace is handled separately.  This script downloads two
additional public retail-sales datasets into ``data/external/`` so adapter tests
can be run against complete real CSV files, not only synthetic fixtures.
"""

from __future__ import annotations

import shutil
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "external" / "data_external_manifest.json"


DATASETS = [
    {
        "name": "rossmann",
        "repo_id": "AiiN-aini/rossmann-store-sales",
        "repo_type": "dataset",
        "files": ["train.csv", "store.csv", "test.csv", "sample_submission.csv"],
    },
    {
        "name": "favorita",
        "repo_id": "t4tiana/store-sales-time-series-forecasting",
        "repo_type": "dataset",
        "files": [
            "train.csv",
            "stores.csv",
            "transactions.csv",
            "oil.csv",
            "holidays_events.csv",
            "test.csv",
            "sample_submission.csv",
        ],
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only when optional dep missing
        raise SystemExit(
            "huggingface_hub is required. Install with `pip install -e '.[llm]'` "
            "or `pip install huggingface_hub`."
        ) from exc

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Files are downloaded from public HuggingFace dataset mirrors. "
            "Hashes below pin the local artifact bytes used by adapter tests."
        ),
        "datasets": [],
    }

    for dataset in DATASETS:
        out_dir = PROJECT_ROOT / "data" / "external" / dataset["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n== {dataset['name']} -> {out_dir}")
        dataset_record = {
            "name": dataset["name"],
            "repo_id": dataset["repo_id"],
            "repo_type": dataset["repo_type"],
            "revision": dataset.get("revision", "main"),
            "repo_url": f"https://huggingface.co/datasets/{dataset['repo_id']}",
            "files": [],
        }
        for filename in dataset["files"]:
            target = out_dir / Path(filename).name
            if target.exists() and target.stat().st_size > 0:
                print(f"exists {target.relative_to(PROJECT_ROOT)} ({target.stat().st_size:,} bytes)")
            else:
                print(f"downloading {dataset['repo_id']}:{filename}")
                cached = hf_hub_download(
                    repo_id=dataset["repo_id"],
                    repo_type=dataset["repo_type"],
                    filename=filename,
                    revision=dataset.get("revision", "main"),
                )
                shutil.copy2(cached, target)
                print(f"saved {target.relative_to(PROJECT_ROOT)} ({target.stat().st_size:,} bytes)")
            dataset_record["files"].append(
                {
                    "filename": filename,
                    "local_path": str(target.relative_to(PROJECT_ROOT)),
                    "size_bytes": target.stat().st_size,
                    "sha256": _sha256(target),
                }
            )
        manifest["datasets"].append(dataset_record)

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
