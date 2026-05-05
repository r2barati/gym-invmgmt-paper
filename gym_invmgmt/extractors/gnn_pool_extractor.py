"""
gnn_pool_extractor.py — Topology-Invariant GNN with Graph-Level Mean Pooling

Key differences from gnn_extractor.py (GNNFeaturesExtractor):
  1. Uses MEAN POOLING over nodes → fixed output dim regardless of topology
  2. Dynamically determines n_main from observation size
  3. Supports set_transfer_topology() for zero-shot topological transfer
  4. Exposes per-node embeddings for per-edge action generation

Architecture:
  Input:  [inv_pos(n), lt_targets(n), gaps(n), demand_vel, norm_time]
          → n_main = (obs_dim - 2) // 3
  GCN:    (batch, n, 3) → GCN1 → ReLU → GCN2 → ReLU → (batch, n, hidden_dim)
  Pool:   mean over nodes → (batch, hidden_dim)
  Output: concat(pool, global) → compress → (batch, features_dim)
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.network_topology import SupplyChainNetwork


def build_adjacency(scenario):
    """Build normalized adjacency matrix for a given topology."""
    net = SupplyChainNetwork(scenario=scenario)
    main_nodes = sorted(
        [n for n in net.graph.nodes()
         if n not in net.market and n not in net.rawmat]
    )
    n_main = len(main_nodes)

    A = np.zeros((n_main, n_main), dtype=np.float32)
    node_to_idx = {node: i for i, node in enumerate(main_nodes)}

    for u, v in net.graph.edges():
        if u in main_nodes and v in main_nodes:
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0  # Undirected message passing

    # Self-loops + degree normalization
    A += np.eye(n_main, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    return A_norm, main_nodes, net


def build_edge_index(scenario):
    """Build edge list mapping reorder links to main node indices.

    Returns:
        edge_pairs: list of (src_idx, dst_idx) for each reorder link
        n_edges: number of reorder edges
        edge_features: np.ndarray of shape (n_edges, 4) with static per-edge
            features [price, lead_time, pipeline_hold, is_rawmat_source]
    """
    net = SupplyChainNetwork(scenario=scenario)
    main_nodes = sorted(
        [n for n in net.graph.nodes()
         if n not in net.market and n not in net.rawmat]
    )
    node_to_idx = {node: i for i, node in enumerate(main_nodes)}

    edge_pairs = []
    edge_features = []
    for u, v in net.reorder_links:
        edge_data = net.graph.edges[(u, v)]
        price = edge_data.get('p', 0.0)
        lead_time = float(edge_data.get('L', 0))
        pipeline_hold = edge_data.get('g', 0.0)
        is_rawmat = 1.0 if u in net.rawmat else 0.0

        edge_features.append([price, lead_time, pipeline_hold, is_rawmat])

        u_in = u in node_to_idx
        v_in = v in node_to_idx
        if u_in and v_in:
            edge_pairs.append((node_to_idx[u], node_to_idx[v]))
        elif not u_in and v_in:
            edge_pairs.append((node_to_idx[v], node_to_idx[v]))
        elif u_in and not v_in:
            edge_pairs.append((node_to_idx[u], node_to_idx[u]))
        else:
            edge_pairs.append((0, 0))

    edge_features = np.array(edge_features, dtype=np.float32)
    return edge_pairs, len(net.reorder_links), edge_features


class GNNPoolingExtractor(BaseFeaturesExtractor):
    """
    Topology-invariant GNN feature extractor with graph-level mean pooling.

    Produces a fixed-size feature vector regardless of the number of nodes,
    enabling zero-shot transfer between topologies.
    """

    EDGE_FEAT_DIM = 4  # [price, lead_time, pipeline_hold, is_rawmat_source]

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 scenario: str = 'base',
                 hidden_dim: int = 64):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim
        self.node_feat_dim = 3  # (inv_pos, lt_target, gap)

        # Build default adjacency (training topology)
        A_norm, main_nodes, net = build_adjacency(scenario)
        expected_obs_dim = 3 * len(main_nodes) + 2
        actual_obs_dim = observation_space.shape[0]
        if actual_obs_dim != expected_obs_dim:
            raise ValueError(
                "GNNPoolingExtractor is a V1 graph-only extractor and requires "
                "observations shaped as [3 node features + 2 globals]. "
                f"Expected dimension {expected_obs_dim} for scenario={scenario!r}, "
                f"got {actual_obs_dim}. Use a V2-compatible pooling extractor "
                "for DomainFeatureWrapper(enhanced=True, grouped=True)."
            )
        self.register_buffer('A', torch.tensor(A_norm))
        self._default_n_main = len(main_nodes)
        self._scenario = scenario

        # Build edge index + edge features for per-edge actions
        edge_pairs, n_edges, edge_features = build_edge_index(scenario)
        self._edge_pairs = edge_pairs
        self._n_edges = n_edges
        self.register_buffer('edge_features', torch.tensor(edge_features))

        # GCN layers (topology-invariant weights)
        self.gcn1 = nn.Linear(self.node_feat_dim, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Compression: mean-pooled (hidden_dim) + global features (2) → features_dim
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim + 2, features_dim),
            nn.ReLU(),
        )

        # Per-edge action MLP: concat(src_emb, dst_emb, edge_feats) → scalar action
        # Edge features let the shared MLP differentiate edges by cost structure
        edge_mlp_input_dim = hidden_dim * 2 + self.EDGE_FEAT_DIM
        self.edge_action_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Output in [-1, 1] for RescaleAction
        )

        # Transfer state
        self._transfer_A = None
        self._transfer_n_main = None
        self._transfer_edge_pairs = None
        self._transfer_n_edges = None
        self._transfer_edge_features = None

    def set_transfer_topology(self, scenario):
        """Configure for zero-shot transfer to a different topology."""
        A_norm, main_nodes, net = build_adjacency(scenario)
        self._transfer_A = torch.tensor(A_norm).to(self.A.device)
        self._transfer_n_main = len(main_nodes)

        edge_pairs, n_edges, edge_features = build_edge_index(scenario)
        self._transfer_edge_pairs = edge_pairs
        self._transfer_n_edges = n_edges
        self._transfer_edge_features = torch.tensor(edge_features).to(self.A.device)

    def clear_transfer(self):
        """Reset to native topology."""
        self._transfer_A = None
        self._transfer_n_main = None
        self._transfer_edge_pairs = None
        self._transfer_n_edges = None
        self._transfer_edge_features = None

    def _get_active_config(self, obs_dim):
        """Determine which topology config to use based on obs dimension."""
        if self._transfer_A is not None:
            return (self._transfer_A, self._transfer_n_main,
                    self._transfer_edge_pairs, self._transfer_edge_features)
        return self.A, self._default_n_main, self._edge_pairs, self.edge_features

    def _gcn_forward(self, node_feats, A):
        """Run GCN layers on node features with given adjacency.

        Args:
            node_feats: (batch, n_nodes, 3)
            A: (n_nodes, n_nodes) normalized adjacency

        Returns:
            node_embeddings: (batch, n_nodes, hidden_dim)
        """
        # GCN Layer 1: X' = ReLU(A @ X @ W1)
        AX = torch.matmul(A, node_feats)
        H1 = torch.relu(self.gcn1(AX))

        # GCN Layer 2: H' = ReLU(A @ H1 @ W2)
        AH1 = torch.matmul(A, H1)
        H2 = torch.relu(self.gcn2(AH1))

        return H2

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that produces graph-level features via mean pooling.

        Also stores node_embeddings and edge_pairs for per-edge action generation.
        """
        batch_size = observations.shape[0]
        obs_dim = observations.shape[1]

        # Dynamically determine n_main from obs size: obs = 3*n + 2
        if (obs_dim - 2) % self.node_feat_dim != 0:
            raise ValueError(
                "GNNPoolingExtractor received an observation that does not "
                f"match the V1 3*n+2 schema: dimension={obs_dim}."
            )
        n_main = (obs_dim - 2) // 3

        # Get appropriate adjacency + edge features
        A, expected_n, edge_pairs, edge_feats = self._get_active_config(obs_dim)
        if n_main != expected_n:
            raise ValueError(
                "GNNPoolingExtractor observation topology does not match the "
                f"active graph config: observation has {n_main} nodes, "
                f"active topology has {expected_n}."
            )

        # Parse observation
        inv_pos = observations[:, :n_main]
        lt_targets = observations[:, n_main:2 * n_main]
        gaps = observations[:, 2 * n_main:3 * n_main]
        global_feats = observations[:, -2:]

        # Stack per-node features: (batch, n_main, 3)
        node_feats = torch.stack([inv_pos, lt_targets, gaps], dim=-1)

        # GCN forward
        node_embeddings = self._gcn_forward(node_feats, A)  # (batch, n, hidden)

        # Store for per-edge action generation
        self._last_node_embeddings = node_embeddings
        self._last_edge_pairs = edge_pairs
        self._last_edge_features = edge_feats

        # MEAN POOL over nodes → (batch, hidden_dim) — TOPOLOGY INVARIANT
        pooled = node_embeddings.mean(dim=1)

        # Concat global features and compress
        combined = torch.cat([pooled, global_feats], dim=1)
        return self.compress(combined)

    def compute_per_edge_actions(self):
        """
        Generate per-edge actions from stored node embeddings + edge features.

        Must be called AFTER forward() in the same forward pass.

        The edge_action_mlp receives:
            concat(src_node_emb, dst_node_emb, edge_static_features)
        where edge_static_features = [price, lead_time, pipeline_hold, is_rawmat]
        This allows the shared MLP to differentiate edges by their cost structure.

        Returns:
            actions: (batch, n_edges) tensor of per-edge actions in [-1, 1]
        """
        node_emb = self._last_node_embeddings  # (batch, n_nodes, hidden)
        edge_pairs = self._last_edge_pairs
        edge_feats = self._last_edge_features  # (n_edges, 4)

        batch_size = node_emb.shape[0]
        actions = []

        for i, (src_idx, dst_idx) in enumerate(edge_pairs):
            src_emb = node_emb[:, src_idx, :]  # (batch, hidden)
            dst_emb = node_emb[:, dst_idx, :]  # (batch, hidden)
            # Expand static edge features to batch: (4,) → (batch, 4)
            ef = edge_feats[i].unsqueeze(0).expand(batch_size, -1)
            edge_input = torch.cat([src_emb, dst_emb, ef], dim=-1)  # (batch, 2*hidden+4)
            action = self.edge_action_mlp(edge_input)  # (batch, 1)
            actions.append(action)

        return torch.cat(actions, dim=-1)  # (batch, n_edges)
