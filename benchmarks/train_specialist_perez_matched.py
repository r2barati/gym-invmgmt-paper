#!/usr/bin/env python3
"""Train a Perez-inspired specialist PPO-MLP comparison model.

Perez setup: Ray/RLlib PPO, [256,256] FFN, 70K episodes (2.1M steps),
discrete actions, single-scenario specialist.

Our setup: SB3 PPO, [256,256] FFN, 2.1M steps, continuous actions,
single-scenario specialist. This narrows the comparison but is still not
an exact reproduction because the RL framework, action parameterization,
normalization, and implementation details differ.

Approach: train with VecNormalize, save training VecNorm stats,
then evaluate post-training using the saved normaliser.
"""
import os, sys, argparse, shutil
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import RescaleAction

from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper

MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "models")

SCENARIOS = {
    "stationary_bl": {"topology": "network", "demand_type": "stationary", "backlog": True},
    "stationary_ls": {"topology": "network", "demand_type": "stationary", "backlog": False},
}


def make_env(cfg, seed=0):
    def _init():
        env = CoreEnv(
            scenario=cfg['topology'],
            demand_config={'type': cfg['demand_type'], 'base_mu': 20},
            num_periods=30,
            backlog=cfg['backlog'],
        )
        env = RescaleAction(env, min_action=-1.0, max_action=1.0)
        env = DomainFeatureWrapper(env, is_blind=False, enhanced=True, grouped=True)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def evaluate(model, vec_env_path, cfg, n_seeds=10):
    """Evaluate model using saved VecNormalize stats."""
    eval_env = DummyVecEnv([make_env(cfg, seed=0)])
    eval_env = VecNormalize.load(vec_env_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    profits = []
    for seed in range(n_seeds):
        # Most reliable DummyVecEnv reseeding is fresh creation
        eval_env = DummyVecEnv([make_env(cfg, seed=seed)])
        eval_env = VecNormalize.load(vec_env_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
        
        obs = eval_env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = eval_env.step(action)
            # Use the UN-normalised reward from Monitor
            total += infos[0].get('reward', reward[0])
            done = dones[0]
        profits.append(total)
        eval_env.close()

    return np.mean(profits), np.std(profits)


def train(scenario_key, total_timesteps=2_100_000):
    cfg = SCENARIOS[scenario_key]
    topo = 'base' if cfg['topology'] == 'network' else 'serial'

    os.makedirs(MODELS_DIR, exist_ok=True)
    log_dir = os.path.join(SCRIPT_DIR, "logs", f"perez_matched_{scenario_key}")
    os.makedirs(log_dir, exist_ok=True)

    # Training env
    vec_env = DummyVecEnv([make_env(cfg, seed=0)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=0.995)

    # Match Perez: [256, 256] two-layer FFN
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        'MlpPolicy', vec_env, policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=2048, batch_size=512, n_epochs=10,
        gamma=0.995, gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, target_kl=0.03,
        tensorboard_log=log_dir, verbose=1,
    )

    model_name = f"ppo-mlp_{topo}_perez_matched_{scenario_key}"

    # Checkpoint every 200K steps so we can pick the best
    ckpt_dir = os.path.join(SCRIPT_DIR, "checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = CheckpointCallback(
        save_freq=200_000,
        save_path=ckpt_dir,
        name_prefix="rl_model",
        save_vecnormalize=True,
        verbose=1,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"\n{'#' * 70}")
    print(f"  PEREZ-MATCHED TRAINING: {scenario_key}")
    print(f"  Architecture: [256, 256] (matching Perez two-layer 256-node)")
    print(f"  Steps: {total_timesteps:,} (matching 70K episodes × 30)")
    print(f"  Params: {total_params:,}")
    print(f"{'#' * 70}\n")

    model.learn(total_timesteps=total_timesteps, callback=ckpt_cb)

    # Save final model + vecnorm
    final_model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    final_vecnorm_path = os.path.join(MODELS_DIR, f"{model_name}_vecnorm.pkl")
    model.save(final_model_path)
    vec_env.save(final_vecnorm_path)
    print(f"\n  Final model: {final_model_path}")
    print(f"  VecNorm: {final_vecnorm_path}")

    # Evaluate final model
    mean_p, std_p = evaluate(model, final_vecnorm_path, cfg)
    print(f"  Final eval: {mean_p:+.1f} ± {std_p:.1f}")

    # Also evaluate all checkpoints to find the best
    best_profit = mean_p
    best_ckpt = "final"
    ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.zip') and 'vecnormalize' not in f])
    for ckpt_file in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_file)
        vecn_file = ckpt_file.replace('.zip', '_vecnormalize.pkl')
        vecn_path = os.path.join(ckpt_dir, vecn_file)
        if not os.path.exists(vecn_path):
            continue
        ckpt_model = PPO.load(ckpt_path)
        m, s = evaluate(ckpt_model, vecn_path, cfg)
        step_num = ckpt_file.split('_')[-2]
        print(f"    Checkpoint {step_num}: {m:+.1f} ± {s:.1f}")
        if m > best_profit:
            best_profit = m
            best_ckpt = ckpt_path
            # Copy best to models dir
            shutil.copy2(ckpt_path, final_model_path)
            shutil.copy2(vecn_path, final_vecnorm_path)

    print(f"\n  Best checkpoint: {best_profit:+.1f} from {best_ckpt}")

    vec_env.close()
    return final_model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, choices=list(SCENARIOS.keys()))
    parser.add_argument('--steps', type=int, default=2_100_000)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for key in SCENARIOS:
            train(key, args.steps)
    elif args.scenario:
        train(args.scenario, args.steps)
    else:
        parser.print_help()
