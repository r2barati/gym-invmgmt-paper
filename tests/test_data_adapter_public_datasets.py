import os
from pathlib import Path

import numpy as np
import pytest

from gym_invmgmt import CoreEnv
from gym_invmgmt.data_adapters import (
    hierarchy_csv_to_tree_topology_yaml,
    long_demand_csv_to_spec,
    m5_wide_csv_to_spec,
    retail_store_csv_to_star_topology_yaml,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROSSMANN_TRAIN = PROJECT_ROOT / "data" / "external" / "rossmann" / "train.csv"
ROSSMANN_STORES = PROJECT_ROOT / "data" / "external" / "rossmann" / "store.csv"
FAVORITA_TRAIN = PROJECT_ROOT / "data" / "external" / "favorita" / "train.csv"
FAVORITA_STORES = PROJECT_ROOT / "data" / "external" / "favorita" / "stores.csv"
M5_TRAIN = PROJECT_ROOT / "data" / "external" / "m5" / "sales_train_evaluation.csv"
M5_METADATA = PROJECT_ROOT / "results" / "m5_window_metadata.json"


pytestmark = pytest.mark.skipif(
    os.environ.get("GYM_INVMGMT_RUN_PUBLIC_DATA_TESTS") != "1",
    reason="Set GYM_INVMGMT_RUN_PUBLIC_DATA_TESTS=1 to run full public-dataset adapter tests.",
)


def test_full_rossmann_dataset_can_drive_external_series_and_inferred_graph(tmp_path):
    if not ROSSMANN_TRAIN.exists() or not ROSSMANN_STORES.exists():
        pytest.skip("Missing Rossmann files; run scripts/data/download_public_retail_datasets.py")

    spec = long_demand_csv_to_spec(
        ROSSMANN_TRAIN,
        demand_col="Sales",
        time_col="Date",
        group_cols=["Store"],
        group_filter={"Open": 1},
        edge_map={"7": (7, 0), "8": (8, 0)},
        scenario="custom",
        scale_to_mean=20.0,
    )

    topology_yaml = tmp_path / "rossmann_star.yaml"
    cfg = retail_store_csv_to_star_topology_yaml(
        ROSSMANN_STORES,
        topology_yaml,
        store_id_col="Store",
        selected_store_ids=[7, 8],
        name="rossmann_two_store_star",
    )
    assert "finite-supplier star graph" in cfg["metadata"]["topology_assumption"]

    user_D = spec.env_kwargs["user_D"]
    assert len(user_D[(7, 0)]) > 750
    assert len(user_D[(8, 0)]) > 750
    assert np.isclose(user_D[(7, 0)].mean(), 20.0)
    assert np.isclose(user_D[(8, 0)].mean(), 20.0)
    assert np.all(user_D[(7, 0)] >= 0)
    assert np.all(user_D[(8, 0)] >= 0)

    kwargs = spec.core_env_kwargs()
    kwargs["config_path"] = str(topology_yaml)
    env = CoreEnv(**kwargs, num_periods=30)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    env.step(np.zeros(env.action_space.shape))
    assert env.D[0].shape == (2,)
    assert np.allclose(env.D[0], [user_D[(7, 0)][0], user_D[(8, 0)][0]])


def test_full_m5_adapter_matches_frozen_metadata():
    if not M5_TRAIN.exists() or not M5_METADATA.exists():
        pytest.skip("Missing M5 file or frozen metadata artifact.")

    import json

    with M5_METADATA.open("r", encoding="utf-8") as f:
        frozen = json.load(f)

    spec = m5_wide_csv_to_spec(
        M5_TRAIN,
        num_periods=int(frozen["num_periods"]),
        base_mu=float(frozen["base_mu"]),
        search_days=int(frozen["search_days"]),
        min_window_mean=float(frozen["min_window_mean"]),
        min_nonzero_frac=float(frozen["min_nonzero_frac"]),
        max_to_mean_cap=float(frozen["max_to_mean_cap"]),
    )

    assert spec.metadata["series_id"] == frozen["series_id"]
    assert spec.metadata["start_day"] == frozen["start_day"]
    assert spec.metadata["end_day"] == frozen["end_day"]
    assert np.allclose(spec.metadata["raw_series"], frozen["raw_series"])
    assert np.allclose(spec.demand_config["external_series"], frozen["scaled_series"])


def test_full_m5_hierarchy_can_infer_tree_topology(tmp_path):
    if not M5_TRAIN.exists() or not M5_METADATA.exists():
        pytest.skip("Missing M5 file or frozen metadata artifact.")

    import json

    with M5_METADATA.open("r", encoding="utf-8") as f:
        frozen = json.load(f)

    topology_yaml = tmp_path / "m5_hierarchy_tree.yaml"
    cfg = hierarchy_csv_to_tree_topology_yaml(
        M5_TRAIN,
        topology_yaml,
        hierarchy_cols=["state_id", "store_id", "cat_id", "dept_id", "item_id", "id"],
        group_filter={"id": frozen["series_id"]},
        name="m5_selected_series_hierarchy",
    )

    assert cfg["metadata"]["leaf_count"] == 1
    assert cfg["metadata"]["topology_assumption"].startswith("tree-shaped")

    env = CoreEnv(scenario="custom", config_path=str(topology_yaml), num_periods=30)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert len(env.network.retail_links) == 1


def test_full_favorita_dataset_can_drive_edge_mapped_custom_graph(tmp_path):
    if not FAVORITA_TRAIN.exists() or not FAVORITA_STORES.exists():
        pytest.skip("Missing Favorita files; run scripts/data/download_public_retail_datasets.py")

    spec = long_demand_csv_to_spec(
        FAVORITA_TRAIN,
        demand_col="sales",
        time_col="date",
        group_cols=["store_nbr"],
        group_filter={"family": "GROCERY I"},
        edge_map={"1": (1, 0), "2": (2, 0)},
        scenario="custom",
    )

    user_D = spec.env_kwargs["user_D"]
    assert set(user_D) == {(1, 0), (2, 0)}
    assert len(user_D[(1, 0)]) > 1500
    assert len(user_D[(2, 0)]) > 1500
    assert np.all(user_D[(1, 0)] >= 0)
    assert np.all(user_D[(2, 0)] >= 0)

    topology_yaml = tmp_path / "favorita_star.yaml"
    retail_store_csv_to_star_topology_yaml(
        FAVORITA_STORES,
        topology_yaml,
        store_id_col="store_nbr",
        selected_store_ids=[1, 2],
        name="favorita_two_store_star",
    )

    kwargs = spec.core_env_kwargs()
    kwargs["config_path"] = str(topology_yaml)
    env = CoreEnv(**kwargs, num_periods=30)
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))

    assert env.D.shape == (30, 2)
    assert np.allclose(env.D[0], [user_D[(1, 0)][0], user_D[(2, 0)][0]])


def test_full_favorita_store_hierarchy_can_infer_tree_topology(tmp_path):
    if not FAVORITA_STORES.exists():
        pytest.skip("Missing Favorita stores file; run scripts/data/download_public_retail_datasets.py")

    topology_yaml = tmp_path / "favorita_store_hierarchy.yaml"
    cfg = hierarchy_csv_to_tree_topology_yaml(
        FAVORITA_STORES,
        topology_yaml,
        hierarchy_cols=["state", "city", "cluster", "store_nbr"],
        group_filter={"state": "Pichincha"},
        name="favorita_pichincha_store_hierarchy",
    )

    assert cfg["metadata"]["path_count"] >= 1
    assert cfg["metadata"]["leaf_count"] >= 1

    env = CoreEnv(scenario="custom", config_path=str(topology_yaml), num_periods=2)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert len(env.network.retail_links) == cfg["metadata"]["leaf_count"]
