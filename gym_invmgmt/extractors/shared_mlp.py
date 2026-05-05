"""
SharedMLPExtractor — Per-link shared MLP feature extractor for SB3.

Processes each reorder link's 14-feature vector with shared weights, appends a
global mean-pooled context to every link, then compresses the flattened result
to a fixed policy feature dimension. This matches the shipped residual
checkpoints serialized as ``src.models.shared_mlp_extractor.SharedMLPExtractor``.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SharedMLPExtractor(BaseFeaturesExtractor):
    """Weight-sharing extractor for per-link observations.

    Args:
        observation_space: Box of shape ``n_links * features_per_link``.
        features_dim: Output feature dimension for the PPO policy head.
        features_per_link: Number of features per link in the augmented obs.
        hidden_dim: Hidden dimension for the per-link MLP.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        features_per_link: int = 14,
        hidden_dim: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.features_per_link = features_per_link
        self.hidden_dim = hidden_dim

        obs_dim = observation_space.shape[0]
        if obs_dim % features_per_link != 0:
            raise ValueError(
                f"Observation dim {obs_dim} is not divisible by "
                f"features_per_link {features_per_link}"
            )
        self.n_links = obs_dim // features_per_link

        # Per-link shared MLP (weight sharing across all links)
        self.shared_mlp = nn.Sequential(
            nn.Linear(features_per_link, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Compress [per_link_hidden || global_context] for each link to a
        # fixed-size representation expected by the saved PPO heads.
        concat_dim = self.n_links * (2 * hidden_dim)
        self.compress = nn.Sequential(
            nn.Linear(concat_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        link_features = observations.view(batch_size, self.n_links, self.features_per_link)

        # Shared MLP over the link dimension.
        per_link_hidden = self.shared_mlp(link_features)

        # Global context, broadcast back to each link.
        global_context = per_link_hidden.mean(dim=1)
        global_expanded = global_context.unsqueeze(1).expand(-1, self.n_links, -1)

        combined = torch.cat([per_link_hidden, global_expanded], dim=-1)
        flat = combined.reshape(batch_size, -1)

        return self.compress(flat)

    def get_shared_mlp_state(self) -> dict:
        """Export shared MLP weights for cross-topology transfer."""
        return self.shared_mlp.state_dict()

    def load_shared_mlp_state(self, state_dict: dict):
        """Import shared MLP weights from another topology's model."""
        self.shared_mlp.load_state_dict(state_dict)
