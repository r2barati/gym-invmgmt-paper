"""
sac_eval_agent.py — SAC Evaluation Agent

Matches the training pipeline in train_generalist.py:
  Pipeline: CoreEnv → [DomainFeatureWrapper] → RescaleAction([-1,1])
            → DummyVecEnv → VecNormalize → SAC

The agent:
  1. Augments raw obs with domain features (matching training obs_level)
  2. Normalizes via saved VecNormalize stats
  3. Predicts deterministic actions in [-1, 1] from SAC
  4. Reverse-rescales to [0, action_high]
"""

import pickle
import numpy as np

from stable_baselines3 import SAC


class SACAgent:
    def __init__(self, env, model_path: str = 'data/models/sac_baseline.zip',
                 stats_path: str = 'data/models/vec_normalize_sac.pkl',
                 obs_level: str = 'v2', is_blind: bool = False,
                 verbose: bool = False):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.action_high = env.action_space.high.copy()
        self.action_low = env.action_space.low.copy()

        # --- Feature augmentation (matches training pipeline) ---
        self.obs_level = obs_level
        self.feat_wrapper = None
        if obs_level != 'raw':
            from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
            enhanced = (obs_level == 'v2')
            self.feat_wrapper = DomainFeatureWrapper(
                env, is_blind=is_blind, enhanced=enhanced, grouped=True)

        self.model = SAC.load(model_path)

        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        self.norm_env.training = False
        self.norm_env.norm_reward = False

        if verbose:
            print(
                f"[SACAgent] Loaded model from '{model_path}'\n"
                f"[SACAgent] Loaded VecNormalize stats from '{stats_path}'\n"
                f"[SACAgent] obs_level={obs_level}, is_blind={is_blind}\n"
                f"[SACAgent] Action range: [{self.action_low.min():.1f}, "
                f"{self.action_high.max():.1f}]"
            )

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Augment raw observation with domain features if configured."""
        if self.feat_wrapper is not None:
            return self.feat_wrapper._augment_obs(obs)
        return obs

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
        scaled_action, _ = self.model.predict(norm_obs, deterministic=True)
        raw_action = self._rescale_action(np.array(scaled_action, dtype=np.float64))
        return raw_action
