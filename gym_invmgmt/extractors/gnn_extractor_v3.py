"""
gnn_extractor_v3.py — Fixed MPNN GNN Features Extractor (V3)

Fixes over V2:
  1. BatchNorm on node features BEFORE message passing (fixes scale imbalance)
  2. Grouped feature layout: GNN receives [all_inv_pos, all_lt_target, ...] not interleaved
  3. Same MPNN architecture (edge features, directed graph, attention, residuals)
"""

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.network_topology import SupplyChainNetwork
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper


class EdgeConditionedMPNNLayer(nn.Module):
    """
    Message Passing layer with edge features and attention.
    Same as V2 but operates on properly normalized inputs.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attn_linear = nn.Linear(2 * node_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.residual_proj = nn.Linear(node_dim, hidden_dim) if node_dim != hidden_dim else nn.Identity()

    def forward(self, H, edge_index, edge_feat):
        batch_size, n_nodes, node_dim = H.shape
        src, dst = edge_index

        H_src = H[:, src, :]
        H_dst = H[:, dst, :]
        E = edge_feat.unsqueeze(0).expand(batch_size, -1, -1)

        msg_input = torch.cat([H_src, H_dst, E], dim=-1)
        messages = self.message_mlp(msg_input)

        attn_input = torch.cat([H_src, H_dst], dim=-1)
        attn_scores = self.leaky_relu(self.attn_linear(attn_input))
        attn_weights = self._scatter_softmax(attn_scores.squeeze(-1), dst, n_nodes)

        weighted_msgs = messages * attn_weights.unsqueeze(-1)

        hidden_dim = messages.shape[-1]
        aggregated = torch.zeros(batch_size, n_nodes, hidden_dim, device=H.device)
        dst_expanded = dst.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, hidden_dim)
        aggregated.scatter_add_(1, dst_expanded, weighted_msgs)

        H_residual = self.residual_proj(H)
        H_new = self.layer_norm(H_residual + self.update_mlp(aggregated))
        return H_new

    def _scatter_softmax(self, scores, index, num_nodes):
        batch_size = scores.shape[0]
        max_vals = torch.full((batch_size, num_nodes), float('-inf'), device=scores.device)
        idx_exp = index.unsqueeze(0).expand(batch_size, -1)
        max_vals.scatter_reduce_(1, idx_exp, scores, reduce='amax', include_self=False)
        max_gathered = max_vals.gather(1, idx_exp)
        exp_scores = torch.exp(scores - max_gathered)
        sum_exp = torch.zeros(batch_size, num_nodes, device=scores.device)
        sum_exp.scatter_add_(1, idx_exp, exp_scores)
        sum_gathered = sum_exp.gather(1, idx_exp)
        return exp_scores / (sum_gathered + 1e-8)


class GNNFeaturesExtractorV3(BaseFeaturesExtractor):
    """
    Fixed MPNN GNN Extractor (V3).
    
    Key fixes vs V2:
    - BatchNorm1d on node features before message passing
    - Grouped feature layout (all inv_pos together, etc.)
    - Same MPNN + directed graph + attention architecture
    """
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 scenario: str = 'base',
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 n_node_feats: int = None):
        super().__init__(observation_space, features_dim)

        net = SupplyChainNetwork(scenario=scenario)
        self.main_nodes = sorted([n for n in net.graph.nodes()
                                   if n not in net.market and n not in net.rawmat])
        self.n_main = len(self.main_nodes)
        node_to_idx = {node: i for i, node in enumerate(self.main_nodes)}

        # Auto-detect or use provided node feature count
        if n_node_feats is None:
            n_node_feats = 8  # V2_NODE_FEATS, backward compatible
        n_global_feats = 10  # V2_GLOBAL_FEATS
        n_augmented = n_node_feats * self.n_main + n_global_feats
        self.base_dim = observation_space.shape[0] - n_augmented
        self.n_node_feats = n_node_feats
        self.n_global_feats = n_global_feats

        # === FIX 1: BatchNorm on node features before MPNN ===
        self.node_bn = nn.BatchNorm1d(n_node_feats)

        # Edge features
        edge_dim = 3
        self.edge_dim = edge_dim

        upstream_src, upstream_dst, upstream_feats = [], [], []
        downstream_src, downstream_dst, downstream_feats = [], [], []

        max_L = max([net.graph.edges[e].get('L', 0) for e in net.graph.edges()], default=1) or 1
        max_p = max([net.graph.edges[e].get('p', 0) for e in net.graph.edges()], default=1) or 1
        max_g = max([net.graph.edges[e].get('g', 0) for e in net.graph.edges()], default=1) or 1

        for u, v in net.graph.edges():
            if u in node_to_idx and v in node_to_idx:
                props = net.graph.edges[(u, v)]
                feat = [props.get('L', 0) / max_L, props.get('p', 0) / max_p, props.get('g', 0) / max_g]
                upstream_src.append(node_to_idx[u])
                upstream_dst.append(node_to_idx[v])
                upstream_feats.append(feat)
                downstream_src.append(node_to_idx[v])
                downstream_dst.append(node_to_idx[u])
                downstream_feats.append(feat)

        for i in range(self.n_main):
            upstream_src.append(i); upstream_dst.append(i); upstream_feats.append([0., 0., 0.])
            downstream_src.append(i); downstream_dst.append(i); downstream_feats.append([0., 0., 0.])

        self.register_buffer('up_edge_index', torch.tensor([upstream_src, upstream_dst], dtype=torch.long))
        self.register_buffer('up_edge_feat', torch.tensor(upstream_feats, dtype=torch.float32))
        self.register_buffer('down_edge_index', torch.tensor([downstream_src, downstream_dst], dtype=torch.long))
        self.register_buffer('down_edge_feat', torch.tensor(downstream_feats, dtype=torch.float32))

        # MPNN layers
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        in_dim = n_node_feats
        for _ in range(n_layers):
            self.up_layers.append(EdgeConditionedMPNNLayer(in_dim, edge_dim, hidden_dim))
            self.down_layers.append(EdgeConditionedMPNNLayer(in_dim, edge_dim, hidden_dim))
            in_dim = hidden_dim

        per_node_dim = 2 * hidden_dim * self.n_main
        global_pool_dim = 2 * hidden_dim
        combined_dim = self.base_dim + per_node_dim + global_pool_dim + n_global_feats

        self.compress = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        base_obs = observations[:, :self.base_dim]
        augmented = observations[:, self.base_dim:]

        # Grouped layout: reshape to (batch, n_main, n_node_feats)
        # Layout: [feat0_node0..feat0_nodeN, feat1_node0..feat1_nodeN, ...]
        n_node_total = self.n_node_feats * self.n_main
        node_block = augmented[:, :n_node_total]
        global_feats = augmented[:, n_node_total:]

        # Grouped: reshape as (batch, n_feats, n_main) then transpose to (batch, n_main, n_feats)
        H = node_block.view(batch_size, self.n_node_feats, self.n_main).transpose(1, 2)

        # === FIX 1: BatchNorm BEFORE message passing ===
        # BatchNorm1d expects (batch, features) — reshape, normalize, reshape back
        H_flat = H.reshape(batch_size * self.n_main, self.n_node_feats)
        H_flat = self.node_bn(H_flat)
        H = H_flat.reshape(batch_size, self.n_main, self.n_node_feats)

        # Upstream MPNN
        H_up = H
        for layer in self.up_layers:
            H_up = layer(H_up, self.up_edge_index, self.up_edge_feat)

        # Downstream MPNN
        H_down = H
        for layer in self.down_layers:
            H_down = layer(H_down, self.down_edge_index, self.down_edge_feat)

        up_flat = H_up.view(batch_size, -1)
        down_flat = H_down.view(batch_size, -1)
        up_pool = H_up.mean(dim=1)
        down_pool = H_down.mean(dim=1)

        combined = torch.cat([base_obs, up_flat, down_flat, up_pool, down_pool, global_feats], dim=1)
        return self.compress(combined)
