"""
transfer_per_edge_policy.py — Unified Per-Edge Actor-Critic Policy for Transfer

A topology-invariant ActorCriticPolicy that works with ANY extractor implementing
the compute_per_edge_actions() interface. Supports:
  - BAMPNNPoolExtractor (Bidirectional Attentive MPNN with pooling)
  - ResidualGCNPoolExtractor (Enriched GCN with heuristic anchor)

Key design: The actor generates N actions where N = number of reorder edges
in the CURRENT (possibly transferred) topology, not the training topology.
This is the fundamental mechanism enabling zero-shot topological transfer.
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from typing import Tuple, Optional

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule


class TransferPerEdgePolicy(ActorCriticPolicy):
    """
    Unified per-edge ActorCritic policy for topology-invariant transfer.

    Works with any extractor that exposes:
      - forward(obs) → features (for critic)
      - compute_per_edge_actions() → (batch, n_edges) actions in [-1, 1]

    Args:
        extractor_class: BAMPNNPoolExtractor or ResidualGCNPoolExtractor
        extractor_kwargs: kwargs passed to the extractor
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        extractor_class=None,
        extractor_kwargs: dict = None,
        value_hidden_dim: int = 64,
        **kwargs
    ):
        self._extractor_class = extractor_class
        self._extractor_kwargs = extractor_kwargs or {}
        self._value_hidden_dim = value_hidden_dim

        # Remove kwargs that ActorCriticPolicy doesn't expect
        kwargs.pop('features_extractor_class', None)
        kwargs.pop('features_extractor_kwargs', None)

        features_dim = self._extractor_kwargs.get('features_dim', 128)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=extractor_class,
            features_extractor_kwargs=self._extractor_kwargs,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        """Override to build per-edge actor + value critic."""
        features_dim = self._extractor_kwargs.get('features_dim', 128)

        # Value network: from pooled features
        self.value_net_mlp = nn.Sequential(
            nn.Linear(features_dim, self._value_hidden_dim),
            nn.ReLU(),
            nn.Linear(self._value_hidden_dim, self._value_hidden_dim),
            nn.ReLU(),
        )

        # Per-edge actor: built into extractor.compute_per_edge_actions()
        # We only need the log_std for the Gaussian distribution
        n_actions = self.action_space.shape[0]
        self.log_std = nn.Parameter(
            torch.zeros(n_actions, dtype=torch.float32),
            requires_grad=True
        )

        # SB3-required feature tensor; per-edge actions are produced separately
        self.mlp_extractor = _MLPExtractorAdapter()

    def _build(self, lr_schedule: Schedule) -> None:
        """Override build to set up custom networks."""
        self._build_mlp_extractor()

        # Value head
        self.value_net = nn.Linear(self._value_hidden_dim, 1)

        # Action distribution
        n_actions = self.action_space.shape[0]
        self.action_dist = DiagGaussianDistribution(n_actions)

        # Optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )

    def _get_action_mean(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features and per-edge action means."""
        features = self.extract_features(obs)
        action_mean = self.features_extractor.compute_per_edge_actions()
        return features, action_mean

    def _pad_or_trim_actions(self, action_mean: torch.Tensor) -> torch.Tensor:
        """Handle dimension mismatch between training and transfer topologies."""
        n_actions = self.action_space.shape[0]
        if action_mean.shape[1] < n_actions:
            pad = torch.zeros(
                action_mean.shape[0], n_actions - action_mean.shape[1],
                device=action_mean.device
            )
            return torch.cat([action_mean, pad], dim=1)
        elif action_mean.shape[1] > n_actions:
            return action_mean[:, :n_actions]
        return action_mean

    def _log_std_for(self, action_mean: torch.Tensor) -> torch.Tensor:
        """Get correctly sized log_std for arbitrary topology transfers."""
        n_act = action_mean.shape[1]
        n_train_act = self.log_std.shape[0]
        if n_act <= n_train_act:
            log_std = self.log_std[:n_act]
        else:
            pad = torch.zeros(n_act - n_train_act, dtype=torch.float32, device=self.device)
            log_std = torch.cat([self.log_std, pad])
        return log_std.expand_as(action_mean)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        """Full forward pass for action selection."""
        features, action_mean = self._get_action_mean(obs)
        action_mean = self._pad_or_trim_actions(action_mean)

        log_std = self.log_std.expand_as(action_mean)

        # Critic
        value_features = self.value_net_mlp(features)
        values = self.value_net(value_features)

        # Distribution
        distribution = self.action_dist.proba_distribution(action_mean, log_std)

        if deterministic:
            actions = distribution.mode()
        else:
            actions = distribution.sample()

        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO loss computation."""
        features, action_mean = self._get_action_mean(obs)
        action_mean = self._pad_or_trim_actions(action_mean)

        log_std = self.log_std.expand_as(action_mean)

        value_features = self.value_net_mlp(features)
        values = self.value_net(value_features)

        distribution = self.action_dist.proba_distribution(action_mean, log_std)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution for given observations."""
        _, action_mean = self._get_action_mean(obs)
        action_mean = self._pad_or_trim_actions(action_mean)
        log_std = self.log_std.expand_as(action_mean)
        return self.action_dist.proba_distribution(action_mean, log_std)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict value for given observations."""
        features = self.extract_features(obs)
        value_features = self.value_net_mlp(features)
        return self.value_net(value_features)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """Predict action (for evaluation, including transfer)."""
        self.set_training_mode(False)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32).to(self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            features = self.extract_features(obs_tensor)
            action_mean = self.features_extractor.compute_per_edge_actions()

            # For eval: return actions matching CURRENT topology (not training)
            if deterministic:
                actions = action_mean
            else:
                n_act = action_mean.shape[1]
                log_std = self._log_std_for(action_mean)
                dist = DiagGaussianDistribution(n_act)
                dist = dist.proba_distribution(action_mean, log_std)
                actions = dist.sample()

        actions = actions.cpu().numpy()
        if actions.shape[0] == 1:
            actions = actions[0]
        return actions, state


class _MLPExtractorAdapter(nn.Module):
    """Small adapter that satisfies SB3's mlp_extractor interface."""

    def __init__(self):
        super().__init__()
        self.latent_dim_pi = 1
        self.latent_dim_vf = 1

    def forward(self, features):
        return features, features
