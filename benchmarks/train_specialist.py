#!/usr/bin/env python3
"""
train_specialist.py — Train scenario-specific (specialist) models.

Unlike the generalist (domain randomization), each specialist is trained
on a FIXED scenario configuration. This isolates the 'generalization tax'
and provides an expert-level RL baseline for the manuscript.

Usage:
  python train_specialist.py --arch ppo-mlp --scenario stationary_bl --steps 200000
  python train_specialist.py --all --steps 200000
"""
import os, sys, pickle, argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper

MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'training_logs')

# ---------------------------------------------------------------------------
# Scenario definitions — each matches a benchmark grid scenario
# ---------------------------------------------------------------------------
SCENARIOS = {
    'stationary_bl': {
        'demand_config': {'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
        'backlog': True,
    },
    'ts_bl': {
        'demand_config': {'type': 'combined_chaos', 'effects': ['trend', 'seasonal'],
                         'base_mu': 20, 'use_goodwill': False},
        'backlog': True,
    },
    'ts_ls': {
        'demand_config': {'type': 'combined_chaos', 'effects': ['trend', 'seasonal'],
                         'base_mu': 20, 'use_goodwill': False},
        'backlog': False,
    },
    'tss_bl': {
        'demand_config': {'type': 'combined_chaos', 'effects': ['trend', 'seasonal', 'shock'],
                         'base_mu': 20, 'use_goodwill': False},
        'backlog': True,
    },
    'tss_gw_bl': {
        'demand_config': {'type': 'combined_chaos', 'effects': ['trend', 'seasonal', 'shock'],
                         'base_mu': 20, 'use_goodwill': True},
        'backlog': True,
    },
    'stationary_ls': {
        'demand_config': {'type': 'stationary', 'base_mu': 20, 'use_goodwill': False},
        'backlog': False,
    },
}


def make_specialist_env(scenario_key, topology='network', seed=0, obs_level='v1'):
    """Create a fixed-scenario env (no domain randomization)."""
    scfg = SCENARIOS[scenario_key]
    def _init():
        env = CoreEnv(
            scenario=topology,
            demand_config=dict(scfg['demand_config']),
            num_periods=30,
            backlog=scfg['backlog'],
        )
        if obs_level != 'raw':
            enhanced = (obs_level == 'v2')
            env = DomainFeatureWrapper(env, is_blind=False, enhanced=enhanced, grouped=True)
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = Monitor(env)
        return env
    return _init


def train_specialist(scenario_key, steps=200_000, topology='network', obs_level='v1', n_envs=4):
    """Train a specialist PPO-MLP model for a single scenario."""
    topo_label = 'base' if topology == 'network' else 'serial'
    model_name = f'ppo-mlp_{topo_label}_specialist_{scenario_key}'
    model_path = os.path.join(MODELS_DIR, model_name)
    stats_path = os.path.join(MODELS_DIR, f'{model_name}_vecnorm.pkl')
    log_dir = os.path.join(LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)

    print(f'\n{"=" * 70}')
    print(f'  SPECIALIST TRAINING: {scenario_key}')
    print(f'  Topology: {topo_label} | Obs: {obs_level} | Steps: {steps}')
    print(f'{"=" * 70}')

    # Training env (fixed scenario, varied seeds via reset)
    env_fns = [make_specialist_env(scenario_key, topology, seed=i, obs_level=obs_level) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Eval env (same scenario, different seed)
    eval_vec = DummyVecEnv([make_specialist_env(scenario_key, topology, seed=99, obs_level=obs_level)])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_vec.obs_rms = vec_env.obs_rms  # Explicitly share the running stats
    eval_vec.training = False

    # PPO model (same hyperparameters as generalist)
    model = PPO(
        'MlpPolicy', vec_env,
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
        tensorboard_log=log_dir, verbose=1,
    )

    eval_cb = EvalCallback(
        eval_vec, best_model_save_path=log_dir,
        log_path=log_dir, eval_freq=5_000,
        n_eval_episodes=5, deterministic=True, verbose=1,
    )

    model.learn(total_timesteps=steps, callback=eval_cb, progress_bar=True)

    # Save
    model.save(model_path)
    vec_env.training = False
    with open(stats_path, 'wb') as f:
        pickle.dump(vec_env, f)

    print(f'\n  OK Specialist saved: {model_path}.zip')

    # Report
    try:
        eval_log = np.load(os.path.join(log_dir, 'evaluations.npz'))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        print(f'  Best eval: {eval_log["results"][best_idx].mean():+.1f} '
              f'at step {eval_log["timesteps"][best_idx]}')
    except Exception:
        pass

    vec_env.close()
    eval_vec.close()
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specialist RL Training')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help='Scenario to train on')
    parser.add_argument('--steps', type=int, default=200_000)
    parser.add_argument('--topo', type=str, default='network',
                        choices=['network', 'serial'])
    parser.add_argument('--obs-level', type=str, default='v1',
                        choices=['raw', 'v1', 'v2'])
    parser.add_argument('--all', action='store_true',
                        help='Train all scenarios')
    args = parser.parse_args()

    if args.all:
        for scenario in SCENARIOS:
            train_specialist(scenario, args.steps, args.topo, args.obs_level)
    elif args.scenario:
        train_specialist(args.scenario, args.steps, args.topo, args.obs_level)
    else:
        parser.print_help()
