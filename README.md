# gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods

An open-source simulation framework bridging Operations Research (OR) and Machine Learning (ML) for multi-echelon supply chain optimization.

## Project Links

- Paper/code repository: [r2barati/gym-invmgmt-paper](https://github.com/r2barati/gym-invmgmt-paper)
- arXiv paper: [arXiv:2605.11355](https://arxiv.org/abs/2605.11355)
- Standalone environment package: [r2barati/gym-invmgmt](https://github.com/r2barati/gym-invmgmt)
- Trained checkpoint archive: [rezabarati/gym-invmgmt-weights](https://huggingface.co/datasets/rezabarati/gym-invmgmt-weights)

## Repository Structure

- `gym_invmgmt/`: The core environment package.
  - `data_adapters.py`: Dataset adapters for M5-style wide sales files,
    generic long-format demand CSVs, and node/edge CSV topology conversion.
- `agents/`: Implementation of 28 evaluation agent configurations plus 1 theoretical Oracle upper bound.
- `benchmarks/`: Scripts to run the evaluation matrix across 26 scenarios (22 main + 4 MARL supplementary).
- `results/`: Benchmark outputs.
  - `cache_v2/`: Per-agent CSV caches (one file per agent).
  - `diagnostics/`: Supplementary diagnostic runs not merged into the
    canonical benchmark artifact.
  - `benchmark_final_merged.csv`: The canonical merged CSV consumed by all figure scripts.
- `data/models/`: Pre-trained RL model checkpoints (hosted on HuggingFace — see below).
- `paper/`: LaTeX source for the manuscript.

## Pre-trained Models

To keep this repository lightweight, the pre-trained RL model checkpoints (`.zip` + `.pkl` files) are hosted on HuggingFace.
Download them and place them in `data/models/` to reproduce the exact results from the paper:

```bash
bash download_weights.sh
```

The checkpoint archive is hosted at
[`rezabarati/gym-invmgmt-weights`](https://huggingface.co/datasets/rezabarati/gym-invmgmt-weights)
and includes a SHA-256 `models_manifest.json`. It contains the trained
SB3/PPO/SAC/imitation checkpoints and matching `VecNormalize` statistics.
The optional third-party Qwen GGUF file used for local LLM diagnostics is not
re-hosted in this archive; place it at
`data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf` only if you plan to rerun
LLM experiments.

## Installation

```bash
pip install -e ".[all]"
```

## Dataset Ingestion

The package includes lightweight adapters for turning external data into
`CoreEnv` inputs:

- `m5_wide_csv_to_spec(...)`: selects a reproducible window from M5-style wide
  sales data and exposes it as an empirical demand trace.
- `wide_demand_csv_to_spec(...)`: general wide-format adapter for datasets with
  repeated period columns such as `d_1, d_2, ...` or `week_01, week_02, ...`.
  The M5 helper is a convenience wrapper over this function.
- `long_demand_csv_to_spec(...)`: parses ordinary long-format demand data and
  can either produce one global external trace or edge-specific `user_D`
  demand vectors for multi-retailer graphs.
- `topology_csvs_to_yaml(...)`: converts simple node/edge CSV files into the
  custom topology YAML format used by `CoreEnv(scenario="custom")`.
- `retail_store_csv_to_star_topology_yaml(...)`: infers a documented
  one-supplier star topology from retail store metadata when a dataset provides
  store identities but not physical replenishment edges.
- `hierarchy_csv_to_tree_topology_yaml(...)`: infers a tree-shaped DAG from
  hierarchy columns such as M5's `state_id → store_id → cat_id → dept_id →
  item_id → id` or Favorita's `state → city → cluster → store_nbr`.

These adapters preserve the distinction between demand extraction and graph
simulation: datasets can supply empirical demand, custom topology, or both.
The test suite validates the same adapter contract on M5-wide, Rossmann-style
daily store sales, and Favorita-style multi-store/item sales fixtures.

To validate against full public Rossmann and Favorita CSVs, download the source
files into the ignored `data/external/` folder and enable the opt-in tests:

```bash
python3 scripts/data/download_public_retail_datasets.py
GYM_INVMGMT_RUN_PUBLIC_DATA_TESTS=1 pytest tests/test_data_adapter_public_datasets.py -q
```

The canonical `D_WalmartM5` benchmark helper delegates to this adapter layer, so
M5 reruns use the same general ingestion path as the standalone public-dataset
tests.
For datasets with forecasting hierarchies, the resulting graph is
hierarchy-inferred rather than a physically observed replenishment network.

## Reproducibility Quick-Start

To download the released checkpoints, rerun the canonical benchmark artifact,
and validate the publication files:

```bash
# 1. Download pre-trained weights
bash download_weights.sh

# 2. Run fast evaluation on all scenarios (skipping slow LLMs)
cd benchmarks
python3 run_benchmarks.py --agent ALL --seeds 10

# 3. Verify tests, cached results, manifest, and citation metadata
python3 verify_publication_ready.py --quick
```

The manuscript figure-generation scripts are not shipped in this repository
snapshot; the canonical merged benchmark artifact is
`results/benchmark_final_merged.csv`.

## Reproducing Benchmarks

Run the benchmark suite to evaluate all agents across the scenarios:

```bash
cd benchmarks
python3 run_benchmarks.py --agent ALL
```

To include the LLM agents (slow, requires local GGUF model):

```bash
python3 run_benchmarks.py --agent ALL --include-llm
```

To regenerate the merged CSV from per-agent caches:

```bash
python3 run_benchmarks.py --merge
```

## Training Agents

To retrain the unified generalist models on the domain-randomized environments:

```bash
cd benchmarks
python3 train_generalist.py --all
```

To train M5-specialist neural policies on the real M5 trace:

```bash
# Canonical paper topologies with real M5 demand
python3 train_m5_specialist.py --arch ppo-mlp --topology base --steps 50000
python3 train_m5_specialist.py --arch ppo-mlp --topology serial --steps 50000

# Hierarchy-inferred M5 topology (new custom-topology experiment)
python3 train_m5_specialist.py --arch ppo-mlp --topology hierarchy --steps 50000
```

The hierarchy mode changes the environment graph and action space, so its
results should be reported as a custom topology-transfer experiment rather than
a direct replacement for the canonical base/serial M5 rows.

## Results Manifest

The shipped `results/benchmark_final_merged.csv` was generated with:
- **Agents**: 29 registered IDs total for the non-LLM roster: 1 Oracle + 22 primary + 6 ablations
- **LLM baseline**: `LLM-Policy-C` is included in the merged artifact when
  generated with `--include-llm`; direct per-period LLM prompting variants are
  diagnostic only.
- **Scenarios**: 26 (16 core + 4 stationary replication + 4 MARL + 2 M5)
- **Seeds**: 10 canonical seeds per scenario (260 episodes per agent)
- **Supplementary LLM diagnostics**: `LLM-ZS-Direct` and `LLM-InvAgent-C`
  require explicit LLM evaluation and are not part of the registered non-LLM
  roster. Stopped direct-prompting diagnostics are stored under
  `results/diagnostics/`.

## Citing

If you use this benchmark, result artifact, or environment code in your
research, please cite the accompanying paper:

```bibtex
@misc{barati2026gyminvmgmt,
  title = {gym-invmgmt: An Open Benchmarking Framework for Inventory Management Methods},
  author = {Barati, Reza and Hu, Qinmin Vivian},
  year = {2026},
  eprint = {2605.11355},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2605.11355}
}
```

You can also cite the benchmark software release through
[`CITATION.cff`](CITATION.cff), which points GitHub's citation widget to the
paper as the preferred citation.
