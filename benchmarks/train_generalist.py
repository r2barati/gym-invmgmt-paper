#!/usr/bin/env python3
"""
train_generalist.py — Unified generalist training

Trains a SINGLE model per (architecture, topology) pair across demand variations:
  - Topology: FIXED per model (base and serial have different obs/act dims)
  - Demands: stationary, trend, seasonal, shock, combined, M5-like
  - Goodwill: on/off (50% each)
  - Fulfillment: backlog/lost-sales (50% each)
  - Noise scale: [0.5, 2.0] uniform

Supported architectures:
  --arch ppo-mlp        PPO with MLP policy (no feature extractor)
  --arch ppo-gnn        PPO with GNNFeaturesExtractorV3
  --arch ppo-transformer PPO with TransformerFeaturesExtractor (node-token, spatial only)
  --arch st-ppo         PPO with TransformerTemporalExtractor (spatio-temporal, frame-stacked)
  --arch sac         SAC with MLP policy
  --arch residual    PPO with SharedMLPExtractor + proportional residual actions

Usage:
  python train_generalist.py --arch ppo-mlp --steps 200000
  python train_generalist.py --arch ppo-mlp --steps 200000 --topo serial
  python train_generalist.py --arch ppo-gnn --steps 200000
  python train_generalist.py --arch sac --steps 200000
  python train_generalist.py --all --steps 200000  # Train all archs × both topos
"""
import os
import sys
import pickle
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

import gymnasium as gym
from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from agents.heuristic_agent import HeuristicAgent

# Models dir
MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'training_logs')

# ---------------------------------------------------------------------------
# Enhanced Domain Randomization — covers full paper scenario space
# Topology is fixed per model: base=(71,)/(11,), serial=(15,)/(3,)
# ---------------------------------------------------------------------------
EFFECT_OPTIONS = ['trend', 'seasonal', 'shock']


class FullDomainRandomizationWrapper(gym.Wrapper):
    """
    Wrapper that randomizes scenario dimensions on each reset().

    Covers: demand effects, goodwill, fulfillment mode, noise scale,
    and occasional M5-like external series.
    Topology is FIXED (base and serial have different obs/act dimensions).
    """

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        base_env = self.env.unwrapped
        seed = kwargs.get('seed')
        rng = np.random.default_rng(seed) if seed is not None else getattr(
            base_env, 'np_random', np.random.default_rng()
        )

        # Random demand effects (each with 50% chance)
        effects = [e for e in EFFECT_OPTIONS if rng.random() < 0.5]

        # 15% chance of M5-like external series
        external_series = None
        if rng.random() < 0.15:
            n = 30
            t = np.arange(n)
            base = 20.0 + 0.02 * t * rng.uniform(0, 2)
            seasonal = 1 + 0.3 * np.sin(2 * np.pi * t / 7)
            demand = rng.poisson(np.maximum(base * seasonal, 0.5))
            spikes = rng.random(n) < 0.05
            demand[spikes] = (demand[spikes] * 3).astype(int)
            external_series = demand.astype(float)

        # Random goodwill and fulfillment
        use_goodwill = bool(rng.choice([False, True]))
        backlog = bool(rng.choice([False, True]))
        noise_scale = float(rng.uniform(0.5, 2.0))

        base_env.demand_engine.effects = effects
        base_env.demand_engine.type = 'combined_chaos' if effects else 'stationary'
        base_env.demand_engine.use_goodwill = use_goodwill
        base_env.demand_engine.noise_scale = noise_scale
        base_env.demand_engine.sentiment = 1.0
        
        base_env.demand_engine.external_series = external_series

        base_env.backlog = backlog

        return self.env.reset(**kwargs)

    @staticmethod
    def make_randomized_env(seed=0, topology='network',
                            use_feature_wrapper=True, is_blind=False,
                            enhanced=True, grouped=True):
        """Factory that creates a new env with randomized demand config."""
        def _init():
            env = CoreEnv(
                scenario=topology,
                demand_config={'type': 'stationary', 'base_mu': 20},
                num_periods=30,
                backlog=True,
            )

            # Wrapper handles randomization on reset
            env = FullDomainRandomizationWrapper(env)

            if use_feature_wrapper:
                env = DomainFeatureWrapper(env, is_blind=is_blind,
                                           enhanced=enhanced, grouped=grouped)

            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            env = Monitor(env)
            return env
        return _init


def make_train_env(arch, seed=0, topology='network', is_blind=False, obs_level='v2',
                   n_history=4):
    """Create a training environment appropriate for the architecture.

    Observation levels:
      'raw'     — CoreEnv only, no augmented features (71d)
      'v1'      — DomainFeatureWrapper(enhanced=False) — 3 features/node (96d)
      'v2'      — DomainFeatureWrapper(enhanced=True)  — 8 features/node (129d, default)
      'perlink' — PerLinkFeatureWrapper — 14 features/link (154d for base topology)
                  Same obs as Residual but with standard actions (no heuristic wrapper).
                  Used to isolate per-link representation vs. heuristic bootstrapping.

    All architectures now receive the SAME observation by default (V2),
    ensuring fair architecture-vs-architecture comparisons.
    """
    if obs_level == 'perlink':
        # PerLink obs: same as Residual's observation, but standard action space
        from gym_invmgmt.wrappers.per_link_wrapper import PerLinkFeatureWrapper
        def _init():
            env = CoreEnv(scenario=topology, demand_config={'type': 'stationary', 'base_mu': 20},
                          num_periods=30, backlog=True)
            
            # Wrapper handles randomization on reset
            env = FullDomainRandomizationWrapper(env)
            
            heuristic = HeuristicAgent(env.unwrapped, is_blind=is_blind)
            env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=is_blind)
            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            env = Monitor(env)
            return env
        return _init

    if arch == 'st-ppo':
        # ST-PPO: add TemporalFrameStack after DomainFeatureWrapper
        from gym_invmgmt.wrappers.temporal_frame_stack import TemporalFrameStack
        def _init():
            env = CoreEnv(scenario=topology,
                          demand_config={'type': 'stationary', 'base_mu': 20},
                          num_periods=30, backlog=True)
            env = FullDomainRandomizationWrapper(env)
            env = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)
            env = TemporalFrameStack(env, n_history=n_history)
            env = RescaleAction(env, min_action=-1.0, max_action=1.0)
            env = Monitor(env)
            return env
        return _init

    use_features = (obs_level != 'raw')
    enhanced = (obs_level == 'v2')
    return FullDomainRandomizationWrapper.make_randomized_env(
        seed=seed, topology=topology, use_feature_wrapper=use_features,
        is_blind=is_blind, enhanced=enhanced, grouped=True,
    )


def make_eval_env(arch, seed=99, topology='network', is_blind=False, obs_level='v2',
                  n_history=4):
    """Create a deterministic eval env (stationary, fixed topology)."""
    def _init():
        env = CoreEnv(
            scenario=topology,
            demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
            num_periods=30, backlog=True,
        )
        if obs_level == 'perlink':
            from gym_invmgmt.wrappers.per_link_wrapper import PerLinkFeatureWrapper
            heuristic = HeuristicAgent(env, is_blind=is_blind)
            env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=is_blind)
        elif obs_level != 'raw':
            enhanced = (obs_level == 'v2')
            env = DomainFeatureWrapper(env, is_blind=is_blind,
                                        enhanced=enhanced, grouped=True)
        if arch == 'st-ppo':
            from gym_invmgmt.wrappers.temporal_frame_stack import TemporalFrameStack
            env = TemporalFrameStack(env, n_history=n_history)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def _make_residual_env(seed=0, topology='network', is_blind=False,
                       randomize=True, reward_lambda=0.001):
    """Create a Residual RL env with PerLink features + ResidualActionWrapper.

    The Residual agent uses a fundamentally different pipeline:
      CoreEnv → [FullDomainRandomizationWrapper] → PerLinkFeatureWrapper
              → ProportionalResidualWrapper → Monitor

    This matches the shipped residual checkpoints.
    """
    from gym_invmgmt.wrappers.per_link_wrapper import PerLinkFeatureWrapper
    from gym_invmgmt.wrappers.residual_action import ProportionalResidualWrapper

    def _init():
        if randomize:
            # Start with an initial config; FullDomainRandomizationWrapper
            # will resample demand/goodwill/backlog on every reset()
            env = CoreEnv(scenario=topology,
                          demand_config={'type': 'stationary', 'base_mu': 20},
                          num_periods=30, backlog=True)
            env = FullDomainRandomizationWrapper(env)
        else:
            env = CoreEnv(scenario=topology,
                          demand_config={'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
                          num_periods=30, backlog=True)

        heuristic = HeuristicAgent(env.unwrapped, is_blind=is_blind)
        env = PerLinkFeatureWrapper(env, heuristic_agent=heuristic, is_blind=is_blind)
        env = ProportionalResidualWrapper(env, heuristic_agent=heuristic,
                                          max_pct=0.5, reward_lambda=reward_lambda)
        env = Monitor(env)
        return env
    return _init


# ---------------------------------------------------------------------------
# Architecture-specific model builders
# ---------------------------------------------------------------------------

def build_model(arch, vec_env, log_dir, topology='network'):
    """Build the appropriate model for the given architecture."""
    # GNN and Transformer extractors encode graph structure at init,
    # so they need the correct scenario name.
    extractor_scenario = 'base' if topology == 'network' else 'serial'

    if arch == 'ppo-mlp':
        policy_kwargs = dict(
            net_arch=[256, 128],  # Paper: MLP [256, 128]
        )
        return PPO(
            'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log=log_dir, verbose=1,
        )

    elif arch == 'ppo-gnn':
        from gym_invmgmt.extractors.gnn_extractor_v3 import GNNFeaturesExtractorV3
        policy_kwargs = dict(
            features_extractor_class=GNNFeaturesExtractorV3,
            features_extractor_kwargs=dict(
                features_dim=256, scenario=extractor_scenario,
                hidden_dim=64, n_layers=3,
            ),
            net_arch=[256, 128],
        )
        return PPO(
            'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log=log_dir, verbose=1,
        )

    elif arch == 'ppo-transformer':
        from gym_invmgmt.extractors.transformer_extractor import TransformerFeaturesExtractor
        policy_kwargs = dict(
            features_extractor_class=TransformerFeaturesExtractor,
            features_extractor_kwargs=dict(
                features_dim=256, scenario=extractor_scenario,
                d_model=64, n_heads=4, n_layers=3, dropout=0.1,
            ),
            net_arch=[256, 128],
        )
        return PPO(
            'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log=log_dir, verbose=1,
        )

    elif arch == 'st-ppo':
        from gym_invmgmt.extractors.transformer_temporal_extractor import TransformerTemporalExtractor
        policy_kwargs = dict(
            features_extractor_class=TransformerTemporalExtractor,
            features_extractor_kwargs=dict(
                features_dim=256, scenario=extractor_scenario,
                n_history=4, d_model=64, n_heads=4, n_layers=3, dropout=0.1,
            ),
            net_arch=[256, 128],
        )
        return PPO(
            'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log=log_dir, verbose=1,
        )

    elif arch == 'sac':
        return SAC(
            'MlpPolicy', vec_env,
            learning_rate=3e-4, batch_size=256,
            gamma=0.99, tau=0.005,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=log_dir, verbose=1,
        )

    elif arch == 'residual':
        from gym_invmgmt.extractors.shared_mlp import SharedMLPExtractor
        policy_kwargs = dict(
            features_extractor_class=SharedMLPExtractor,
            features_extractor_kwargs=dict(
                features_dim=128,
                features_per_link=14,
                hidden_dim=64,
            ),
            net_arch=[128, 64],
        )
        return PPO(
            'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
            tensorboard_log=log_dir, verbose=1,
        )

    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(arch, total_timesteps=200_000, topology='network',
          is_blind=False, obs_level='v2'):
    os.makedirs(MODELS_DIR, exist_ok=True)
    topo_label = 'base' if topology == 'network' else 'serial'

    # Build descriptive config suffix for file naming
    blind_tag = '_blind' if is_blind else ''
    obs_tag = f'_obs{obs_level}' if obs_level != 'v2' else ''
    config_suffix = f'{arch}_{topo_label}{obs_tag}{blind_tag}'

    log_dir = os.path.join(LOGS_DIR, config_suffix)
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized training env (fixed topology, randomized demands)
    if arch == 'residual':
        # Residual uses per-link observations + proportional residual actions.
        vec_env = DummyVecEnv([_make_residual_env(
            seed=0, topology=topology, is_blind=is_blind)])
    else:
        vec_env = DummyVecEnv([make_train_env(
            arch, seed=0, topology=topology, is_blind=is_blind, obs_level=obs_level)])
    vec_env = VecNormalize(vec_env, norm_obs=True,
                           norm_reward=(arch != 'sac'),
                           clip_obs=10.0, gamma=0.995)

    # Create eval env (same topology, stationary baseline)
    if arch == 'residual':
        eval_vec = DummyVecEnv([_make_residual_env(
            seed=99, topology=topology, is_blind=is_blind,
            randomize=False, reward_lambda=0.0)])
    else:
        eval_vec = DummyVecEnv([make_eval_env(
            arch, seed=99, topology=topology, is_blind=is_blind, obs_level=obs_level)])
    # Eval env must share VecNormalize stats with the training env
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, gamma=0.995, training=False)
    eval_vec.obs_rms = vec_env.obs_rms  # Explicitly share the running stats

    # Build model
    model = build_model(arch, vec_env, log_dir, topology=topology)
    total_params = sum(p.numel() for p in model.policy.parameters())

    print("=" * 70)
    print(f"  GENERALIST TRAINING — {arch.upper()} ({topo_label})")
    print(f"  Steps: {total_timesteps:,} | Params: {total_params:,}")
    print(f"  Obs: {vec_env.observation_space.shape} | Act: {vec_env.action_space.shape}")
    print(f"  Topology: {topology} (fixed) | Demand: randomized")
    print(f"  Obs level: {obs_level} | Blind: {is_blind}")
    print(f"  Output: {MODELS_DIR}/{config_suffix}.zip")
    print("=" * 70)

    # Eval callback
    eval_cb = EvalCallback(
        eval_vec, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    # Save deployed model: strictly pair the final model with the final VecNormalize stats
    # (Using best_model.zip without its exact VecNormalize state risks drift)
    model_path = os.path.join(MODELS_DIR, config_suffix)
    stats_path = os.path.join(MODELS_DIR, f'{config_suffix}_vecnorm.pkl')

    model.save(model_path)
    print(f"\n  Final model deployed: {model_path}.zip (paired exactly with final stats)")

    with open(stats_path, 'wb') as f:
        pickle.dump(vec_env, f)
    print(f"  VecNorm saved: {stats_path}")

    # Report best eval
    try:
        eval_log = np.load(os.path.join(log_dir, 'evaluations.npz'))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        print(f"  Best eval: {eval_log['results'][best_idx].mean():+.1f} "
              f"at step {eval_log['timesteps'][best_idx]}")
    except Exception:
        pass

    vec_env.close()
    eval_vec.close()
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generalist RL training')
    parser.add_argument('--arch', type=str, default=None,
                        choices=['ppo-mlp', 'ppo-gnn', 'ppo-transformer', 'st-ppo', 'sac', 'residual'],
                        help='Architecture to train')
    parser.add_argument('--steps', type=int, default=200_000,
                        help='Total training timesteps')
    parser.add_argument('--topo', type=str, default='network',
                        choices=['network', 'serial'],
                        help='Topology (default: network/base)')
    parser.add_argument('--blind', action='store_true',
                        help='Train with is_blind=True (no demand model access)')
    parser.add_argument('--obs-level', type=str, default='v2',
                        choices=['raw', 'v1', 'v2', 'perlink'],
                        help='Observation level: raw (71d), v1 (96d), v2 (129d), perlink (154d)')
    parser.add_argument('--all', action='store_true',
                        help='Train all architectures × both topologies')
    args = parser.parse_args()

    if args.all:
        for arch in ['ppo-mlp', 'ppo-gnn', 'ppo-transformer', 'st-ppo', 'sac']:
            for topo in ['network', 'serial']:
                print(f"\n{'#' * 70}")
                print(f"  TRAINING: {arch} / {topo} / obs={args.obs_level} / blind={args.blind}")
                print(f"{'#' * 70}")
                train(arch, args.steps, topology=topo,
                      is_blind=args.blind, obs_level=args.obs_level)
    elif args.arch:
        # Guard: Transformer architectures require V2 grouped domain features
        if args.arch in {'ppo-transformer', 'st-ppo'} and args.obs_level != 'v2':
            parser.error(f"{args.arch} requires --obs-level v2 (got '{args.obs_level}'). "
                         f"Transformer extractors parse V2 grouped node/global features.")
        train(args.arch, args.steps, topology=args.topo,
              is_blind=args.blind, obs_level=args.obs_level)
    else:
        parser.print_help()
