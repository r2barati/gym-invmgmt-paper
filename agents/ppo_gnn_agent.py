"""
ppo_gnn_agent.py — PPO-GNN Evaluation Agent (V3 Directed MPNN)

Matches the training pipeline in train_generalist.py --arch ppo-gnn:
  Pipeline: CoreEnv → DomainFeatureWrapper(enhanced=True, grouped=True)
            → RescaleAction → VecNormalize → PPO(GNNFeaturesExtractorV3)

The V3 extractor uses:
  - Directed upstream/downstream message passing
  - Edge-conditioned MPNN with attention
  - Residual connections + LayerNorm + BatchNorm
  - Grouped V2 feature layout (8 node feats, 10 global feats)

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
from gym_invmgmt.extractors.gnn_extractor_v3 import GNNFeaturesExtractorV3


class PPOGNNAgent:
    def __init__(self, env, model_path: str = 'data/models/gnn-v3_base.zip',
                 stats_path: str = 'data/models/gnn-v3_base_vecnorm.pkl',
                 deterministic: bool = True, is_blind: bool = False):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.deterministic = deterministic
        self.action_high = env.action_space.high.copy()
        self.action_low = env.action_space.low.copy()

        # Must match training: enhanced=True, grouped=True for V3
        self.feature_wrapper = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)

        # Install checkpoint-compatibility module aliases
        try:
            import checkpoint_compat  # noqa: F401
        except ImportError:
            import agents.checkpoint_compat  # noqa: F401

        # Load model — custom_objects ensures V3 class is available
        custom_objects = {
            "GNNFeaturesExtractorV3": GNNFeaturesExtractorV3,
        }
        self.model = PPO.load(model_path, custom_objects=custom_objects)

        # Load VecNormalize stats
        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        self.norm_env.training = False
        self.norm_env.norm_reward = False

        # Safety assertion: obs dimension must match VecNormalize stats
        expected_dim = self.norm_env.obs_rms.mean.shape[0]
        test_obs = self._augment_obs(np.zeros(env.observation_space.shape[0]))
        if test_obs.shape[0] != expected_dim:
            raise ValueError(
                f"PPOGNNAgent obs dimension mismatch: augmented obs has "
                f"{test_obs.shape[0]} dims but VecNormalize expects {expected_dim}. "
                f"Check DomainFeatureWrapper config matches training."
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
