#!/usr/bin/env python3
"""Train specialist PPO-MLP [256,128] on a single scenario.

This is a Perez-inspired specialist protocol: one PPO model per
(topology × demand × fulfillment) scenario, without domain randomization.
It is not a one-to-one reproduction of Perez et al. (2021), which used a
different PPO stack, action discretization, architecture, and budget.

Usage:
    python train_specialist_ppo.py --scenario stationary_bl --steps 500000
    python train_specialist_ppo.py --scenario stationary_ls --steps 500000
    python train_specialist_ppo.py --all --steps 500000
"""
import os, sys, argparse
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

MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")

# ---------------------------------------------------------------------------
# Scenario configs matching the base rows of the B_PaperReplication block
# ---------------------------------------------------------------------------
SCENARIOS = {
    "stationary_bl": {
        "topology": "network",
        "demand_type": "stationary",
        "backlog": True,
    },
    "stationary_ls": {
        "topology": "network",
        "demand_type": "stationary",
        "backlog": False,
    },
}


def make_env(cfg, seed=0, obs_level='v2'):
    """Create a single env for a fixed scenario (NO domain randomization)."""
    def _init():
        env = CoreEnv(
            scenario=cfg['topology'],
            demand_config={'type': cfg['demand_type'], 'base_mu': 20},
            num_periods=30,
            backlog=cfg['backlog'],
        )
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        enhanced = (obs_level == 'v2')
        env = DomainFeatureWrapper(
            env, is_blind=False, enhanced=enhanced, grouped=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def train_specialist(scenario_key, total_timesteps=500_000):
    cfg = SCENARIOS[scenario_key]
    topo = 'base' if cfg['topology'] == 'network' else 'serial'

    os.makedirs(MODELS_DIR, exist_ok=True)
    log_dir = os.path.join(SCRIPT_DIR, "logs", f"specialist_{scenario_key}")
    os.makedirs(log_dir, exist_ok=True)

    # Training env (fixed scenario, no randomization)
    vec_env = DummyVecEnv([make_env(cfg, seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.995)

    # Eval env (same scenario, different seed)
    eval_vec = DummyVecEnv([make_env(cfg, seed=999)])
    eval_vec = VecNormalize(eval_vec, norm_obs=True, norm_reward=False,
                           clip_obs=10.0, gamma=0.995)
    # Keep evaluation on the training observation scale and prevent eval
    # rollouts from adapting normalization statistics.
    eval_vec.obs_rms = vec_env.obs_rms
    eval_vec.training = False

    # Build PPO with [256, 128] to match paper spec
    policy_kwargs = dict(net_arch=[256, 128])
    model = PPO(
        'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
        tensorboard_log=log_dir, verbose=1,
    )

    model_name = f"ppo-mlp_{topo}_specialist_{scenario_key}"
    eval_log_dir = os.path.join(SCRIPT_DIR, "eval_logs", model_name)
    os.makedirs(eval_log_dir, exist_ok=True)

    eval_cb = EvalCallback(
        eval_vec,
        best_model_save_path=os.path.join(MODELS_DIR),
        log_path=eval_log_dir,
        eval_freq=5_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n{'#' * 70}")
    print(f"  SPECIALIST TRAINING: {scenario_key}")
    print(f"  Topology: {topo} | Demand: {cfg['demand_type']} | BL: {cfg['backlog']}")
    print(f"  Steps: {total_timesteps:,} | Params: {total_params:,}")
    print(f"{'#' * 70}\n")

    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    # Deploy final model paired with final VecNormalize stats
    # (Using best_model.zip without its exact VecNormalize state risks drift)
    model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    vecnorm_path = os.path.join(MODELS_DIR, f"{model_name}_vecnorm.pkl")

    model.save(model_path)
    print(f"\n  Final model deployed: {model_path} (paired exactly with final stats)")

    vec_env.save(vecnorm_path)
    print(f"  VecNorm saved: {vecnorm_path}")

    try:
        eval_log = np.load(os.path.join(eval_log_dir, "evaluations.npz"))
        best_idx = np.argmax(eval_log['results'].mean(axis=1))
        print(f"  Best eval: {eval_log['results'].mean(axis=1)[best_idx]:+.1f} "
              f"at step {eval_log['timesteps'][best_idx]}")
    except Exception:
        pass

    vec_env.close()
    eval_vec.close()
    return model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specialist PPO-MLP training')
    parser.add_argument('--scenario', type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help='Scenario to train on')
    parser.add_argument('--steps', type=int, default=500_000,
                        help='Total training timesteps')
    parser.add_argument('--all', action='store_true',
                        help='Train all scenarios')
    args = parser.parse_args()

    if args.all:
        for key in SCENARIOS:
            train_specialist(key, args.steps)
    elif args.scenario:
        train_specialist(args.scenario, args.steps)
    else:
        parser.print_help()
