"""
GNNFeaturesExtractor — Graph Neural Network feature extractor for SB3.

Uses Graph Convolutional Network (GCN) layers to process per-node features
through the supply chain network topology.  The adjacency matrix is derived
from the environment's DAG, enabling structural message passing.

Architecture:
  1. Extract per-node features from DomainFeatureWrapper (inv_pos, lt_target, gap)
  2. Build adjacency matrix from supply chain graph
  3. 2-layer GCN: X' = ReLU(Ã·H·W)  where Ã = D^{-½}·A·D^{-½}
  4. Flatten + concatenate with base obs + global features
  5. Compression MLP → features_dim

CPU-tuned: hidden_channels=32, features_dim=128 (from plan).

"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.network_topology import SupplyChainNetwork


def _core_observation_dim(net: SupplyChainNetwork) -> int:
    """Return CoreEnv's raw observation dimension for this topology."""
    return (
        net.pipeline_length
        + len(net.main_nodes)
        + len(net.retail_links) * 2
        + 2
    )


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """GCN-based feature extractor for supply chain observations.

    Expects observations from DomainFeatureWrapper:
      [raw_obs(70), inv_pos(n_main), lt_target(n_main), gap(n_main), dem_vel(1), norm_time(1)]

    Args:
        observation_space: Box space from DomainFeatureWrapper.
        features_dim: Output feature dimension (default 128 for CPU).
        scenario: Network topology name (for building adjacency).
        hidden_dim: GCN hidden channels (default 32 for CPU).
        config_path: Optional YAML config path for custom topologies.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 128,
        scenario: str = 'network',
        hidden_dim: int = 32,
        config_path: str = None,
    ):
        super().__init__(observation_space, features_dim)

        # Build topology to extract adjacency
        net = SupplyChainNetwork(scenario=scenario, num_periods=30, config_path=config_path)
        self.main_nodes = sorted(
            n for n in net.graph.nodes()
            if n not in net.market and n not in net.rawmat
        )
        self.n_main = len(self.main_nodes)

        # Determine dimensions
        n_augmented = 3 * self.n_main + 2
        self.base_dim = observation_space.shape[0] - n_augmented
        expected_dim = _core_observation_dim(net) + n_augmented
        actual_dim = observation_space.shape[0]
        if actual_dim != expected_dim:
            raise ValueError(
                "GNNFeaturesExtractor is a V1 extractor and requires "
                "DomainFeatureWrapper(enhanced=False). "
                f"Expected observation dimension {expected_dim} for "
                f"scenario={scenario!r}, got {actual_dim}. Use "
                "GNNFeaturesExtractorV3 for enhanced/grouped V2 features."
            )

        # Build normalized adjacency matrix Ã = D^{-½} A D^{-½}
        A = np.zeros((self.n_main, self.n_main), dtype=np.float32)
        node_to_idx = {node: i for i, node in enumerate(self.main_nodes)}

        for u, v in net.graph.edges():
            if u in node_to_idx and v in node_to_idx:
                i, j = node_to_idx[u], node_to_idx[v]
                A[i, j] = 1.0
                A[j, i] = 1.0  # Undirected message passing

        # Self-loops + degree normalization
        A += np.eye(self.n_main, dtype=np.float32)
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        self.register_buffer('A', torch.tensor(A_norm))

        # GCN layers
        node_feat_dim = 3  # (inv_pos, lt_target, gap)
        self.hidden_dim = hidden_dim

        self.gcn1 = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Final compression MLP
        gnn_out_dim = self.n_main * hidden_dim
        combined_dim = self.base_dim + gnn_out_dim + 2  # +2 for global features

        self.compress = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split raw obs and augmented features
        base_obs = observations[:, :self.base_dim]
        augmented = observations[:, self.base_dim:]

        # Extract per-node features
        inv_pos = augmented[:, :self.n_main]
        lt_targets = augmented[:, self.n_main:2 * self.n_main]
        gaps = augmented[:, 2 * self.n_main:3 * self.n_main]
        global_feats = augmented[:, -2:]

        # Node feature matrix: (batch, n_main, 3)
        X = torch.stack([inv_pos, lt_targets, gaps], dim=-1)

        # GCN Layer 1: H1 = ReLU(Ã · X · W1)
        AX = torch.matmul(self.A, X)
        H1 = torch.relu(self.gcn1(AX))

        # GCN Layer 2: H2 = ReLU(Ã · H1 · W2)
        AH1 = torch.matmul(self.A, H1)
        H2 = torch.relu(self.gcn2(AH1))

        # Flatten + concatenate
        gnn_flat = H2.view(batch_size, -1)
        combined = torch.cat([base_obs, gnn_flat, global_feats], dim=1)

        return self.compress(combined)
