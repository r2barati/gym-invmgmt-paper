"""
ppo_transformer_agent.py — PPO-Transformer (Node-Token) Evaluation Agent

Matches the training pipeline in train_generalist.py --arch ppo-transformer:
  Pipeline: CoreEnv → DomainFeatureWrapper(enhanced=True, grouped=True)
            → RescaleAction → VecNormalize → PPO(TransformerFeaturesExtractor)

The agent:
  1. Augments raw observations with DomainFeatureWrapper features
  2. Normalizes via saved VecNormalize stats
  3. Predicts actions in [-1, 1] from the PPO model
  4. Reverse-rescales to [0, action_high]
"""

import os
import pickle
import numpy as np

from stable_baselines3 import PPO
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.extractors.transformer_extractor import TransformerFeaturesExtractor


class PPOTransformerAgent:
    def __init__(self, env, model_path: str = 'data/models/ppo-transformer_base.zip',
                 stats_path: str = 'data/models/ppo-transformer_base_vecnorm.pkl',
                 deterministic: bool = True, is_blind: bool = False):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.deterministic = deterministic
        self.action_high = env.action_space.high.copy()
        self.action_low = env.action_space.low.copy()

        # Must match training: enhanced=True, grouped=True
        self.feature_wrapper = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)

        # Load model with TransformerFeaturesExtractor in scope
        custom_objects = {
            "TransformerFeaturesExtractor": TransformerFeaturesExtractor
        }
        self.model = PPO.load(model_path, custom_objects=custom_objects)

        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        self.norm_env.training = False
        self.norm_env.norm_reward = False

        # Safety check: verify obs dimension matches VecNormalize stats
        test_obs = self._augment_obs(env.observation_space.sample())
        expected_dim = self.norm_env.obs_rms.mean.shape[0]
        assert test_obs.shape[0] == expected_dim, (
            f"Obs dim mismatch: augmented={test_obs.shape[0]} vs VecNormalize={expected_dim}. "
            f"Check DomainFeatureWrapper config or VecNormalize pkl."
        )

    def _augment_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        return self.feature_wrapper._augment_obs(raw_obs)

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_2d = obs.reshape(1, -1)
        return self.norm_env.normalize_obs(obs_2d).squeeze(0)

    def _rescale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        scaled = np.clip(scaled_action, -1.0, 1.0)
        raw = self.action_low + (scaled + 1.0) / 2.0 * (self.action_high - self.action_low)
        return np.maximum(raw, 0.0)

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        augmented_obs = self._augment_obs(obs)
        norm_obs = self._normalise_obs(augmented_obs)
        scaled_action, _ = self.model.predict(norm_obs, deterministic=self.deterministic)
        raw_action = self._rescale_action(np.array(scaled_action, dtype=np.float64))
        return raw_action
