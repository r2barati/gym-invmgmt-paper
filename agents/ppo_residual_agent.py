"""
ppo_residual_agent.py — Residual RL evaluation agent.

Matches the shipped residual checkpoints:
  CoreEnv → PerLinkFeatureWrapper → ProportionalResidualWrapper
          → VecNormalize → PPO(SharedMLPExtractor)
"""

import os
import pickle
import numpy as np
from stable_baselines3 import PPO

try:
    import agents.checkpoint_compat  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != 'agents':
        raise
    import checkpoint_compat  # noqa: F401

from agents.newsvendor_heuristic_agent import NewsvendorHeuristicAgent
from gym_invmgmt.wrappers.per_link_wrapper import PerLinkFeatureWrapper
from gym_invmgmt.extractors.shared_mlp import SharedMLPExtractor


class ResidualRLAgent:
    """
    Evaluates the trained SharedMLP residual policy.

    The model predicts a proportional delta per reorder link. Evaluation applies
    the same mapping used during training: ``base_action * (1 + delta)``.
    """

    def __init__(self, env, model_path='ppo_residual_v2.zip',
                 stats_path='data/models/vec_normalize_residual_v2.pkl',
                 max_pct=0.5, is_blind=False):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.max_pct = max_pct
        self.is_blind = is_blind

        # Build the same heuristic used during training
        self.heuristic = NewsvendorHeuristicAgent(env, is_blind=is_blind)

        # Per-link observations match the saved SharedMLP policy input.
        self.feature_wrapper = PerLinkFeatureWrapper(
            env, heuristic_agent=self.heuristic, is_blind=is_blind)

        if os.path.exists(model_path) and os.path.exists(stats_path):
            custom_objects = {
                "SharedMLPExtractor": SharedMLPExtractor,
            }
            self.model = PPO.load(model_path, custom_objects=custom_objects)

            with open(stats_path, 'rb') as f:
                self.vec_normalize = pickle.load(f)
            self.vec_normalize.training = False
            self.vec_normalize.norm_reward = False
            self.is_loaded = True
        else:
            print(f"[ResidualRLAgent] Warning: files not found ({model_path}, {stats_path})")
            self.is_loaded = False

    def get_action(self, obs, current_period):
        if not self.is_loaded or current_period >= self.env.num_periods:
            return np.zeros(len(self.env.network.reorder_links))

        self.env.unwrapped._update_state()
        per_link_obs = self.feature_wrapper.observation(obs)

        base_action = self.heuristic.get_action(self.env.unwrapped.state, current_period)
        if isinstance(base_action, dict):
            base_arr = np.zeros(len(self.env.network.reorder_links))
            for edge, val in base_action.items():
                if edge in self.env.network.reorder_map:
                    base_arr[self.env.network.reorder_map[edge]] = val
            base_action = base_arr
        base_action = np.asarray(base_action, dtype=np.float64)

        norm_obs = self.vec_normalize.normalize_obs(per_link_obs.reshape(1, -1)).squeeze(0)
        delta, _ = self.model.predict(norm_obs, deterministic=True)
        delta = np.clip(np.asarray(delta, dtype=np.float64), -self.max_pct, self.max_pct)

        final_action = np.clip(base_action * (1.0 + delta), self.env.action_space.low, self.env.action_space.high)
        return final_action
