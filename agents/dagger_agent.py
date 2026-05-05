"""
dagger_agent.py — DAgger Evaluation Agent (Generalist + Specialist)

Supports two modes:
  1. Generalist: Loads ppo_gnn_dagger_best.zip (single model, all demand types)
  2. Specialist: Loads ppo_dagger_{demand_type}.zip (one model per demand type)

Architecture identical to GNN V3 (GNNFeaturesExtractorV3), trained via iterative
imitation learning (DAgger) on Oracle demonstrations.

For specialist mode, the caller must pass demand_type so the correct checkpoint
is loaded. The benchmark runner handles this automatically.
"""

import os
import pickle
import numpy as np

from stable_baselines3 import PPO
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.extractors.gnn_extractor_v3 import GNNFeaturesExtractorV3
import agents.checkpoint_compat  # Registers aliases for released checkpoints


# Maps combined demand labels to the dominant demand type for specialist selection
DEMAND_TYPE_MAP = {
    'stationary': 'stationary',
    'trend': 'trend',
    'seasonal': 'seasonal',
    'shock': 'shock',
    'trend+seasonal': 'trend',
    'trend+shock': 'shock',
    'seasonal+shock': 'shock',
    'trend+seasonal+shock': 'shock',
    'M5_volatile': 'shock',
}


class DAggerAgent:
    """
    Unified DAgger evaluation agent.

    Parameters
    ----------
    env : CoreEnv
        The raw environment instance.
    specialist : bool
        If True, loads demand-type-specific specialist model.
        If False, loads the generalist DAgger model.
    demand_type : str or None
        Required if specialist=True. The demand label from the scenario config
        (e.g., 'stationary', 'trend+seasonal+shock'). Mapped to the dominant
        demand type via DEMAND_TYPE_MAP.
    model_dir : str
        Directory containing model checkpoints.
    """

    def __init__(self, env, specialist: bool = False, demand_type: str = None,
                 model_dir: str = 'data/models',
                 model_path: str = None, stats_path: str = None,
                 deterministic: bool = True, is_blind: bool = False):
        self.env = env
        self.deterministic = deterministic
        self.action_high = env.action_space.high.copy()
        self.action_low = env.action_space.low.copy()

        # Same feature wrapper as GNN V3
        self.feature_wrapper = DomainFeatureWrapper(env, is_blind=is_blind, enhanced=True, grouped=True)

        # Resolve model paths (explicit overrides take priority)
        if model_path is not None and stats_path is not None:
            pass  # Use the provided paths directly
        elif specialist and demand_type is not None:
            mapped = DEMAND_TYPE_MAP.get(demand_type, 'stationary')
            model_path = os.path.join(model_dir, f'ppo_dagger_{mapped}.zip')
            stats_path = os.path.join(model_dir, f'vec_normalize_dagger_{mapped}.pkl')
        else:
            model_path = os.path.join(model_dir, 'ppo_gnn_dagger_best.zip')
            stats_path = os.path.join(model_dir, 'vec_normalize_gnn_dagger.pkl')

        self.model_path = model_path
        self.stats_path = stats_path

        custom_objects = {
            "GNNFeaturesExtractorV3": GNNFeaturesExtractorV3
        }

        self.model = PPO.load(model_path, custom_objects=custom_objects)

        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        self.norm_env.training = False
        self.norm_env.norm_reward = False

        self.is_loaded = True
        self._label = f"DAgger-{'S:' + (DEMAND_TYPE_MAP.get(demand_type, '?') if demand_type else '?') if specialist else 'G'}"

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
