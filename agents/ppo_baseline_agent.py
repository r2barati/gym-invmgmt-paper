"""
rl_agent.py — RLAgent: loads a trained PPO model and evaluates it in the
              benchmark environment with strict normalization handling.

Critical guarantees:
  1. norm_env.training = False  → running stats frozen, not updated
  2. norm_env.norm_reward = False → reward NOT renormalised during eval
  3. obs is manually normalised via norm_env.normalize_obs()
  4. action is manually reverse-rescaled from [-1,1] back to [0, max_action]
     so it perfectly matches env.action_space expectations
"""

import os
import pickle
import numpy as np

from stable_baselines3 import PPO


class RLAgent:
    """
    Wraps a trained PPO model for use in the BenchmarkSuite evaluation loop.

    The benchmark loop calls:
        action = agent.get_action(obs, current_period)
        obs, reward, terminated, truncated, info = env.step(action)

    The raw environment's action_space is Box(0, high, ...), but the PPO
    was trained inside RescaleAction([-1, 1]).  get_action() applies the
    same normalisation/rescaling so the returned action is already in
    [0, action_space.high] and can be passed to env.step() directly.

    Parameters
    ----------
    env        : The raw (unwrapped) CoreEnv instance.
                 Used only to read action_space.high for reverse-rescaling.
    model_path : Path to the saved PPO zip (e.g. 'ppo_baseline.zip').
    stats_path : Path to the pickled VecNormalize object (vec_normalize.pkl).
    obs_level  : Observation level used during training ('raw', 'v1', 'v2').
    is_blind   : Whether the agent was trained with is_blind=True.
    verbose    : If True, print load messages. Set False for batch benchmarking.
    """

    def __init__(self, env, model_path: str, stats_path: str,
                 obs_level: str = 'v2', is_blind: bool = False,
                 verbose: bool = True):
        self.env = env
        self.model_path = model_path
        self.stats_path = stats_path
        self.action_high = env.action_space.high.copy()   # shape: (n_actions,)
        self.action_low  = env.action_space.low.copy()    # all zeros

        # --- Feature augmentation (matches training pipeline) ---
        self.obs_level = obs_level
        self.feat_wrapper = None
        if obs_level != 'raw':
            from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
            enhanced = (obs_level == 'v2')
            self.feat_wrapper = DomainFeatureWrapper(
                env, is_blind=is_blind, enhanced=enhanced, grouped=True)

        # --- Load PPO model ---
        self.model = PPO.load(model_path)

        # --- Load VecNormalize stats ---
        with open(stats_path, 'rb') as f:
            self.norm_env = pickle.load(f)

        # Freeze the normalisation statistics so they cannot
        # drift during evaluation and distort obs comparisons.
        self.norm_env.training = False

        # Disable reward normalisation; the benchmark loop
        # accumulates raw rewards for fair comparison with other agents.
        self.norm_env.norm_reward = False

        # Safety assertion: obs dimension must match VecNormalize stats
        expected_dim = self.norm_env.obs_rms.mean.shape[0]
        test_obs = self._augment_obs(np.zeros(env.observation_space.shape[0]))
        if test_obs.shape[0] != expected_dim:
            raise ValueError(
                f"RLAgent obs dimension mismatch: augmented obs has "
                f"{test_obs.shape[0]} dims but VecNormalize expects {expected_dim}. "
                f"Check obs_level={obs_level} and is_blind={is_blind} match training."
            )

        if verbose:
            print(
                f"[RLAgent] Loaded model from '{model_path}'\n"
                f"[RLAgent] Loaded VecNormalize stats from '{stats_path}'\n"
                f"[RLAgent] obs_level={obs_level}, is_blind={is_blind}\n"
                f"[RLAgent] Action range: [{self.action_low.min():.1f}, "
                f"{self.action_high.max():.1f}]"
            )

    # -----------------------------------------------------------------------
    # Action pipeline
    # -----------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Augment raw observation with domain features if configured."""
        if self.feat_wrapper is not None:
            return self.feat_wrapper._augment_obs(obs)
        return obs

    def _normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply VecNormalize observation normalisation to a single raw obs vector.

        VecNormalize.normalize_obs() expects shape (n_envs, obs_dim) or (obs_dim,).
        We reshape to (1, obs_dim) to match the (1,) DummyVecEnv used in training,
        then squeeze back to (obs_dim,) for model.predict().
        """
        obs_2d = obs.reshape(1, -1)
        return self.norm_env.normalize_obs(obs_2d).squeeze(0)

    def _rescale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Reverse the RescaleAction wrapper's mapping.

        During training, RescaleAction mapped raw_action ∈ [0, high] → [-1, 1].
        The inverse maps policy output ∈ [-1, 1] → [0, high]:

            raw = (policy + 1) / 2 * high

        This is the exact inverse of Gymnasium's RescaleAction transform
        (linear interpolation between [min_action, max_action]).
        """
        # Clip to [-1, 1] to guard against tiny floating-point overflows
        scaled = np.clip(scaled_action, -1.0, 1.0)

        # Linear map: [-1, 1] → [low, high]
        # Since low=0: raw = (scaled + 1) / 2 * high
        raw = self.action_low + (scaled + 1.0) / 2.0 * (self.action_high - self.action_low)
        return np.maximum(raw, 0.0)   # safety clip: no negative orders

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        """
        Returns an action array in [0, action_space.high], ready for env.step().

        Steps
        -----
        1. Augment raw observation with domain features (matching training).
        2. Normalise the augmented obs using frozen VecNormalize statistics.
        3. Run model.predict(deterministic=True) → policy outputs ∈ [-1, 1].
        4. Reverse-rescale back to [0, action_space.high].
        """
        # Step 1: Augment observation (matching training wrapper chain)
        augmented_obs = self._augment_obs(obs)

        # Step 2: Normalise observation
        norm_obs = self._normalise_obs(augmented_obs)

        # Step 3: Get deterministic policy action in [-1, 1]
        scaled_action, _ = self.model.predict(norm_obs, deterministic=True)

        # Step 4: Map back to environment's native action space [0, high]
        raw_action = self._rescale_action(np.array(scaled_action, dtype=np.float64))

        return raw_action
