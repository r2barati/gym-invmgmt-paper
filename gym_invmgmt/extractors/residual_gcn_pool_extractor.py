"""
residual_gcn_pool_extractor.py — Enriched GCN with Heuristic-Anchored Residual Actions

Fixes the broken GCN-Pool by:
  1. Enriching node features: 8 per node (matching BA-MPNN) instead of 3
  2. Enriching global features: 10 instead of 2
  3. Adding heuristic action as an edge feature for residual correction
  4. Producing residual δ ∈ [-1, 1] (scaled externally by Δ_max)

Architecture:
  Node features (8-dim, grouped) → BatchNorm →
  GCN Layer 1: A_norm × X × W₁ → ReLU →
  GCN Layer 2: A_norm × H₁ × W₂ → ReLU →
  CRITIC: mean(H₂) ‖ global_feats → compress → features_dim
  ACTOR:  per-edge MLP(src_emb ‖ dst_emb ‖ edge_static ‖ heuristic_action) → δ ∈ [-1, 1]
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.network_topology import SupplyChainNetwork
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper


def _build_adjacency_from_scenario(scenario=None, config_path=None):
    """Build normalized adjacency for a given scenario."""
    kwargs = {}
    if config_path is not None:
        scenario = 'custom'
        kwargs['config_path'] = config_path
    net = SupplyChainNetwork(scenario=scenario, **kwargs)

    main_nodes = sorted(
        [n for n in net.graph.nodes()
         if n not in net.market and n not in net.rawmat]
    )
    n_main = len(main_nodes)
    node_to_idx = {node: i for i, node in enumerate(main_nodes)}

    A = np.zeros((n_main, n_main), dtype=np.float32)
    for u, v in net.graph.edges():
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0  # Undirected

    A += np.eye(n_main, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Edge data
    edge_pairs = []
    edge_static = []
    for u, v in net.reorder_links:
        edge_data = net.graph.edges[(u, v)]
        price = edge_data.get('p', 0.0)
        lead_time = float(edge_data.get('L', 0))
        pipeline_hold = edge_data.get('g', 0.0)
        is_rawmat = 1.0 if u in net.rawmat else 0.0
        edge_static.append([price, lead_time, pipeline_hold, is_rawmat])

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

    edge_static = np.array(edge_static, dtype=np.float32)
    return A_norm, main_nodes, node_to_idx, net, edge_pairs, edge_static


class ResidualGCNPoolExtractor(BaseFeaturesExtractor):
    """
    Enriched GCN with heuristic-anchored residual action generation.

    Compared to GNNPoolingExtractor:
      - 8 node features (not 3): inv_pos, lt_target, gap, on_hand, h_cost,
        capacity, is_factory, is_retail
      - 10 global features (not 2): demand_vel, norm_time, sin_time, cos_time,
        demand_hist[5], goodwill
      - Heuristic action included in edge features for residual correction
      - BatchNorm on node features before GCN
    """

    EDGE_STATIC_DIM = 4   # [price, lead_time, pipeline_hold, is_rawmat]
    EDGE_HEUR_DIM = 1     # [heuristic_action]

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 scenario: str = 'base',
                 hidden_dim: int = 64,
                 n_node_feats: int = None):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim

        # Feature schema (matches DomainFeatureWrapper V2)
        if n_node_feats is None:
            n_node_feats = DomainFeatureWrapper.V2_NODE_FEATS  # 8
        n_global_feats = DomainFeatureWrapper.V2_GLOBAL_FEATS  # 10
        self.n_node_feats = n_node_feats
        self.n_global_feats = n_global_feats

        # Build training topology
        A_norm, main_nodes, node_to_idx, net, edge_pairs, edge_static = (
            _build_adjacency_from_scenario(scenario=scenario)
        )
        self.register_buffer('A', torch.tensor(A_norm))
        self.register_buffer('edge_static_features', torch.tensor(edge_static))
        self._default_n_main = len(main_nodes)
        self._edge_pairs = edge_pairs
        self._n_edges = len(edge_pairs)
        self._scenario = scenario

        # Determine base obs dim
        n_augmented = n_node_feats * self._default_n_main + n_global_feats
        self.base_dim = observation_space.shape[0] - n_augmented

        # BatchNorm on node features
        self.node_bn = nn.BatchNorm1d(n_node_feats)

        # GCN layers (topology-invariant: same weights for any A)
        self.gcn1 = nn.Linear(n_node_feats, hidden_dim, bias=False)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Critic: mean-pooled + global → compress
        compress_input = hidden_dim + n_global_feats
        self.compress = nn.Sequential(
            nn.Linear(compress_input, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # Per-edge action MLP (residual δ output)
        # Input: src_emb + dst_emb + edge_static + heuristic_action
        edge_mlp_input = 2 * hidden_dim + self.EDGE_STATIC_DIM + self.EDGE_HEUR_DIM
        self.edge_action_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # δ ∈ [-1, 1]
        )

        # Transfer state
        self._transfer_data = None

    # Removed set_heuristic_actions as heuristic actions are now part of observation contract

    def set_transfer_topology(self, scenario=None, config_path=None):
        """Configure for zero-shot transfer to a different topology."""
        A_norm, main_nodes, node_to_idx, net, edge_pairs, edge_static = (
            _build_adjacency_from_scenario(scenario=scenario, config_path=config_path)
        )
        device = self.A.device
        self._transfer_data = {
            'A': torch.tensor(A_norm).to(device),
            'n_main': len(main_nodes),
            'edge_pairs': edge_pairs,
            'edge_static': torch.tensor(edge_static).to(device),
        }

    def clear_transfer(self):
        """Reset to native topology."""
        self._transfer_data = None

    def _get_config(self):
        """Get active topology config."""
        if self._transfer_data is not None:
            d = self._transfer_data
            return d['A'], d['n_main'], d['edge_pairs'], d['edge_static']
        return self.A, self._default_n_main, self._edge_pairs, self.edge_static_features

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        A, n_main, edge_pairs, edge_static = self._get_config()

        # Parse observation: [node_feats_grouped | global_feats | heuristic_actions]
        n_augmented = self.n_node_feats * n_main + self.n_global_feats
        
        node_feats_flat = observations[:, :self.n_node_feats * n_main]
        global_feats = observations[:, self.n_node_feats * n_main : n_augmented]
        h_acts = observations[:, n_augmented:]
        
        self._last_heuristic_actions = h_acts

        # Grouped layout: (batch, n_feats, n_main) → (batch, n_main, n_feats)
        H = node_feats_flat.view(batch_size, self.n_node_feats, n_main).transpose(1, 2)

        # BatchNorm before GCN
        H_flat = H.reshape(batch_size * n_main, self.n_node_feats)
        H_flat = self.node_bn(H_flat)
        H = H_flat.reshape(batch_size, n_main, self.n_node_feats)

        # GCN Layer 1
        AH = torch.matmul(A, H)
        H1 = torch.relu(self.gcn1(AH))

        # GCN Layer 2
        AH1 = torch.matmul(A, H1)
        H2 = torch.relu(self.gcn2(AH1))

        # Store for per-edge actions
        self._last_node_embeddings = H2
        self._last_edge_pairs = edge_pairs
        self._last_edge_static = edge_static

        # Mean pool → (batch, hidden)
        pooled = H2.mean(dim=1)

        combined = torch.cat([pooled, global_feats], dim=1)
        return self.compress(combined)

    def compute_per_edge_actions(self):
        """
        Generate per-edge residual actions δ ∈ [-1, 1].
        Includes heuristic_action as input feature to the edge MLP.
        """
        node_emb = self._last_node_embeddings
        edge_pairs = self._last_edge_pairs
        edge_static = self._last_edge_static

        batch_size = node_emb.shape[0]
        actions = []

        for i, (src_idx, dst_idx) in enumerate(edge_pairs):
            src_emb = node_emb[:, src_idx, :]
            dst_emb = node_emb[:, dst_idx, :]
            ef = edge_static[i].unsqueeze(0).expand(batch_size, -1)

            # Include heuristic action as feature
            if hasattr(self, '_last_heuristic_actions') and self._last_heuristic_actions.shape[1] > i:
                h_act = self._last_heuristic_actions[:, i:i+1]  # (batch, 1)
            else:
                h_act = torch.zeros(batch_size, 1, device=node_emb.device)

            edge_input = torch.cat([src_emb, dst_emb, ef, h_act], dim=-1)
            delta = self.edge_action_mlp(edge_input)  # (batch, 1)
            actions.append(delta)

        return torch.cat(actions, dim=-1)  # (batch, n_edges)
