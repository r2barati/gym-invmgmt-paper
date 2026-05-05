"""
st_ppo_agent.py — ST-PPO (Spatio-Temporal Transformer PPO) Evaluation Agent

Matches the training pipeline in train_generalist.py --arch st-ppo:
  Pipeline: CoreEnv → DomainFeatureWrapper(enhanced=True, grouped=True)
            → TemporalFrameStack(n_history=4)
            → RescaleAction → VecNormalize → PPO(TransformerTemporalExtractor)

Unlike PPO-Transformer (node-token, spatial-only attention), ST-PPO performs true
spatio-temporal self-attention by maintaining a sliding window of recent
observations and treating each (node, timestep) pair as a separate token.

The agent:
  1. Augments raw observations with DomainFeatureWrapper features
  2. Maintains a frame-stack buffer of the last n_history observations
  3. Normalizes via saved VecNormalize stats
  4. Predicts actions in [-1, 1] from the PPO model
  5. Reverse-rescales to [0, action_high]
"""

import os
import pickle
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.extractors.transformer_temporal_extractor import TransformerTemporalExtractor


class STPPOAgent:
    def __init__(self, env, model_path: str = 'data/models/st-ppo_base.zip',
                 stats_path: str = 'data/models/st-ppo_base_vecnorm.pkl',
                 deterministic: bool = True, is_blind: bool = False,
                 n_history: int = 4):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.deterministic = deterministic
        self.action_high = env.action_space.high.copy()
        self.action_low = env.action_space.low.copy()
        self.n_history = n_history

        # Must match training: enhanced=True, grouped=True
        self.feature_wrapper = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)

        # Frame-stack buffer
        self._buffer = deque(maxlen=n_history)
        self._initialized = False

        # Load model with TransformerTemporalExtractor in scope
        custom_objects = {
            "TransformerTemporalExtractor": TransformerTemporalExtractor
        }
        self.model = PPO.load(model_path, custom_objects=custom_objects)

        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        self.norm_env.training = False
        self.norm_env.norm_reward = False

        # Safety check: verify stacked obs dimension matches VecNormalize stats
        test_obs = self._augment_obs(env.observation_space.sample())
        expected_dim = self.norm_env.obs_rms.mean.shape[0]
        expected_single_dim = expected_dim // n_history
        assert test_obs.shape[0] == expected_single_dim, (
            f"Obs dim mismatch: augmented={test_obs.shape[0]} vs "
            f"VecNormalize expects single_frame={expected_single_dim} "
            f"(stacked={expected_dim}, n_history={n_history}). "
            f"Check DomainFeatureWrapper config or VecNormalize pkl."
        )

    def _augment_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        return self.feature_wrapper._augment_obs(raw_obs)

    def _get_stacked_obs(self) -> np.ndarray:
        """Concatenate frame buffer into a single flat vector."""
        return np.concatenate(list(self._buffer), axis=0)

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        obs_2d = obs.reshape(1, -1)
        return self.norm_env.normalize_obs(obs_2d).squeeze(0)

    def _rescale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        scaled = np.clip(scaled_action, -1.0, 1.0)
        raw = self.action_low + (scaled + 1.0) / 2.0 * (self.action_high - self.action_low)
        return np.maximum(raw, 0.0)

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        augmented_obs = self._augment_obs(obs)

        # Initialize buffer on first call (fill with copies of first obs)
        if not self._initialized or current_period == 0:
            self._buffer.clear()
            for _ in range(self.n_history):
                self._buffer.append(augmented_obs.copy())
            self._initialized = True
        else:
            self._buffer.append(augmented_obs.copy())

        stacked_obs = self._get_stacked_obs()
        norm_obs = self._normalise_obs(stacked_obs)
        scaled_action, _ = self.model.predict(norm_obs, deterministic=self.deterministic)
        raw_action = self._rescale_action(np.array(scaled_action, dtype=np.float64))
        return raw_action
