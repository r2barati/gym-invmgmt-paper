#!/usr/bin/env python3
"""Train/evaluate M5-specialist neural policies.

This script is intentionally separate from ``train_generalist.py`` because M5
can be evaluated in two different ways:

* ``base`` / ``serial``: the canonical paper topology with real M5 demand.
* ``hierarchy``: a custom topology inferred from the M5 forecasting hierarchy.

The hierarchy topology changes observation/action dimensions, so existing
generalist checkpoints are not compatible.  This script supports MLP PPO and SAC
specialists, whose policies do not assume a fixed base/serial graph.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from gym_invmgmt import CoreEnv
from gym_invmgmt.data_adapters import hierarchy_csv_to_tree_topology_yaml
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from m5_demand_utils import DEFAULT_METADATA_PATH, DEFAULT_SALES_PATH, get_m5_demand_windows


MODELS_DIR = PROJECT_ROOT / "data" / "models"
LOGS_DIR = PROJECT_ROOT / "training_logs"
RESULTS_DIR = PROJECT_ROOT / "results"
NUM_PERIODS = 30


def _load_m5_series() -> tuple[np.ndarray, dict]:
    windows = get_m5_demand_windows(num_periods=NUM_PERIODS, base_mu=20.0)
    series = np.asarray(windows["volatile"], dtype=float)
    metadata = json.loads(DEFAULT_METADATA_PATH.read_text(encoding="utf-8"))
    return series, metadata


def _m5_hierarchy_topology(metadata: dict) -> tuple[Path, tuple[str, str]]:
    topology_path = RESULTS_DIR / "m5_hierarchy_topology.yaml"
    hierarchy_csv_to_tree_topology_yaml(
        DEFAULT_SALES_PATH,
        topology_path,
        hierarchy_cols=["state_id", "store_id", "cat_id", "dept_id", "item_id", "id"],
        group_filter={"id": metadata["series_id"]},
        name="m5_selected_series_hierarchy",
    )
    retail_edge = (f"id={metadata['series_id']}", "MARKET")
    return topology_path, retail_edge


def make_m5_env(topology: str, is_blind: bool = False):
    series, metadata = _load_m5_series()

    def _init():
        demand_config = {
            "type": "stationary",
            "base_mu": float(np.mean(series)),
            "use_goodwill": False,
            "external_series": series,
            "noise_scale": 0.0,
        }
        kwargs = {}
        scenario = "network" if topology == "base" else topology

        if topology == "hierarchy":
            topology_path, retail_edge = _m5_hierarchy_topology(metadata)
            scenario = "custom"
            kwargs["config_path"] = str(topology_path)
            kwargs["user_D"] = {retail_edge: series}
            # Demand is supplied edge-wise by user_D; keep the demand engine as a
            # harmless fallback for any unmapped retail edge.
            demand_config = {
                "type": "stationary",
                "base_mu": float(np.mean(series)),
                "use_goodwill": False,
                "noise_scale": 0.0,
            }

        env = CoreEnv(
            scenario=scenario,
            demand_config=demand_config,
            num_periods=NUM_PERIODS,
            backlog=True,
            **kwargs,
        )
        env = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env

    return _init


def build_model(arch: str, vec_env, log_dir: Path):
    if arch == "ppo-mlp":
        return PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs={"net_arch": [256, 128]},
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            tensorboard_log=str(log_dir),
            verbose=1,
        )
    if arch == "sac":
        return SAC(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            policy_kwargs={"net_arch": [256, 256]},
            tensorboard_log=str(log_dir),
            verbose=1,
        )
    raise ValueError(f"Unsupported M5 specialist arch: {arch}")


def evaluate(model, vec_env, seeds: list[int]) -> list[float]:
    rewards = []
    vec_env.training = False
    vec_env.norm_reward = False
    for seed in seeds:
        vec_env.seed(seed)
        obs = vec_env.reset()
        done = [False]
        total = 0.0
        steps = 0
        while not done[0] and steps < NUM_PERIODS + 5:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _info = vec_env.step(action)
            total += float(reward[0])
            steps += 1
        rewards.append(total)
    return rewards


def train(args) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    log_dir = LOGS_DIR / f"m5-{args.arch}_{args.topology}{'_blind' if args.blind else ''}"
    log_dir.mkdir(parents=True, exist_ok=True)

    vec_env = DummyVecEnv([make_m5_env(args.topology, is_blind=args.blind)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=(args.arch != "sac"),
        clip_obs=10.0,
        gamma=0.995,
    )

    model = build_model(args.arch, vec_env, log_dir)
    print("=" * 72)
    print(f"M5 SPECIALIST TRAINING | arch={args.arch} topology={args.topology} blind={args.blind}")
    print(f"Obs: {vec_env.observation_space.shape} | Act: {vec_env.action_space.shape}")
    print(f"Steps: {args.steps:,}")
    print("=" * 72)

    model.learn(total_timesteps=args.steps)

    suffix = f"m5-{args.arch}_{args.topology}{'_blind' if args.blind else ''}"
    model_path = MODELS_DIR / suffix
    stats_path = MODELS_DIR / f"{suffix}_vecnorm.pkl"
    model.save(str(model_path))
    with stats_path.open("wb") as f:
        pickle.dump(vec_env, f)

    rewards = evaluate(model, vec_env, args.eval_seeds)
    print(f"Saved {model_path}.zip")
    print(f"Saved {stats_path}")
    print(f"Eval rewards: {np.round(rewards, 3).tolist()}")
    print(f"Eval mean ± std: {np.mean(rewards):+.2f} ± {np.std(rewards):.2f}")
    vec_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["ppo-mlp", "sac"], default="ppo-mlp")
    parser.add_argument("--topology", choices=["base", "serial", "hierarchy"], default="base")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--blind", action="store_true")
    parser.add_argument("--eval-seeds", type=int, nargs="*", default=[42, 123, 456])
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
