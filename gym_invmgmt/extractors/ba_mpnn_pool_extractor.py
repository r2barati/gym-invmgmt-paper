"""
ba_mpnn_pool_extractor.py — Topology-Invariant Bidirectional Attentive MPNN

Reuses the proven EdgeConditionedMPNNLayer from gnn_extractor_v3.py but
replaces the topology-locked flat output with mean-pooled graph embeddings,
enabling zero-shot transfer across topologies.

Architecture:
  Node features (8-dim, grouped) → BatchNorm →
  Upstream MPNN (3 layers, edge-conditioned, attentive) →
  Downstream MPNN (3 layers) →
  CRITIC: mean(H_up) ‖ mean(H_down) ‖ global_feats → compress → features_dim
  ACTOR:  per-edge MLP(src_emb ‖ dst_emb ‖ edge_static) → action ∈ [-1, 1]

Key differences from GNNFeaturesExtractorV3:
  1. Mean-pooled output instead of flat per-node concatenation (topology-invariant)
  2. Per-edge action generation via shared edge MLP
  3. Supports set_transfer_topology() for zero-shot deployment
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
    Identical to gnn_extractor_v3.py's implementation.
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
        self.residual_proj = (
            nn.Linear(node_dim, hidden_dim)
            if node_dim != hidden_dim
            else nn.Identity()
        )

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


def _build_graph_data(scenario):
    """Build edge indices and edge features for a given topology.

    Returns:
        up_edge_index, up_edge_feat, down_edge_index, down_edge_feat,
        main_nodes, node_to_idx, net, edge_pairs, edge_static_features
    """
    net = SupplyChainNetwork(scenario=scenario)
    main_nodes = sorted(
        [n for n in net.graph.nodes()
         if n not in net.market and n not in net.rawmat]
    )
    node_to_idx = {node: i for i, node in enumerate(main_nodes)}
    n_main = len(main_nodes)

    edge_dim = 3
    max_L = max([net.graph.edges[e].get('L', 0) for e in net.graph.edges()], default=1) or 1
    max_p = max([net.graph.edges[e].get('p', 0) for e in net.graph.edges()], default=1) or 1
    max_g = max([net.graph.edges[e].get('g', 0) for e in net.graph.edges()], default=1) or 1

    upstream_src, upstream_dst, upstream_feats = [], [], []
    downstream_src, downstream_dst, downstream_feats = [], [], []

    for u, v in net.graph.edges():
        if u in node_to_idx and v in node_to_idx:
            props = net.graph.edges[(u, v)]
            feat = [
                props.get('L', 0) / max_L,
                props.get('p', 0) / max_p,
                props.get('g', 0) / max_g,
            ]
            upstream_src.append(node_to_idx[u])
            upstream_dst.append(node_to_idx[v])
            upstream_feats.append(feat)
            downstream_src.append(node_to_idx[v])
            downstream_dst.append(node_to_idx[u])
            downstream_feats.append(feat)

    # Self-loops
    for i in range(n_main):
        upstream_src.append(i)
        upstream_dst.append(i)
        upstream_feats.append([0., 0., 0.])
        downstream_src.append(i)
        downstream_dst.append(i)
        downstream_feats.append([0., 0., 0.])

    up_edge_index = torch.tensor([upstream_src, upstream_dst], dtype=torch.long)
    up_edge_feat = torch.tensor(upstream_feats, dtype=torch.float32)
    down_edge_index = torch.tensor([downstream_src, downstream_dst], dtype=torch.long)
    down_edge_feat = torch.tensor(downstream_feats, dtype=torch.float32)

    # Per-edge action data: reorder link → (src_idx, dst_idx, static_feats)
    edge_pairs = []
    edge_static_features = []
    for u, v in net.reorder_links:
        edge_data = net.graph.edges[(u, v)]
        price = edge_data.get('p', 0.0)
        lead_time = float(edge_data.get('L', 0))
        pipeline_hold = edge_data.get('g', 0.0)
        is_rawmat = 1.0 if u in net.rawmat else 0.0

        edge_static_features.append([price, lead_time, pipeline_hold, is_rawmat])

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

    edge_static_features = torch.tensor(edge_static_features, dtype=torch.float32)

    return (up_edge_index, up_edge_feat, down_edge_index, down_edge_feat,
            main_nodes, node_to_idx, net, edge_pairs, edge_static_features)


class BAMPNNPoolExtractor(BaseFeaturesExtractor):
    """
    Topology-invariant Bidirectional Attentive MPNN with graph-level pooling.

    Produces a fixed-size feature vector regardless of the number of nodes,
    enabling zero-shot transfer between topologies.

    Also exposes per-node embeddings for per-edge action generation.
    """

    EDGE_STATIC_DIM = 4  # [price, lead_time, pipeline_hold, is_rawmat]

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 128,
                 scenario: str = 'base',
                 hidden_dim: int = 64,
                 n_layers: int = 3,
                 n_node_feats: int = None):
        super().__init__(observation_space, features_dim)

        self.hidden_dim = hidden_dim

        # Feature schema
        if n_node_feats is None:
            n_node_feats = DomainFeatureWrapper.V2_NODE_FEATS  # 8
        n_global_feats = DomainFeatureWrapper.V2_GLOBAL_FEATS  # 10
        self.n_node_feats = n_node_feats
        self.n_global_feats = n_global_feats

        # Build default (training) topology
        (up_ei, up_ef, down_ei, down_ef,
         main_nodes, node_to_idx, net,
         edge_pairs, edge_static) = _build_graph_data(scenario)

        self.register_buffer('up_edge_index', up_ei)
        self.register_buffer('up_edge_feat', up_ef)
        self.register_buffer('down_edge_index', down_ei)
        self.register_buffer('down_edge_feat', down_ef)
        self.register_buffer('edge_static_features', edge_static)

        self._default_n_main = len(main_nodes)
        self._edge_pairs = edge_pairs
        self._n_edges = len(edge_pairs)
        self._scenario = scenario

        # Determine base obs dim
        n_augmented = n_node_feats * self._default_n_main + n_global_feats
        self.base_dim = observation_space.shape[0] - n_augmented

        # BatchNorm on node features before MPNN
        self.node_bn = nn.BatchNorm1d(n_node_feats)

        # MPNN layers (bidirectional)
        edge_dim = 3
        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        in_dim = n_node_feats
        for _ in range(n_layers):
            self.up_layers.append(EdgeConditionedMPNNLayer(in_dim, edge_dim, hidden_dim))
            self.down_layers.append(EdgeConditionedMPNNLayer(in_dim, edge_dim, hidden_dim))
            in_dim = hidden_dim

        # TOPOLOGY-INVARIANT output: mean-pooled (not flattened)
        # Critic input: up_pool(hidden) + down_pool(hidden) + global_feats
        compress_input = 2 * hidden_dim + n_global_feats
        self.compress = nn.Sequential(
            nn.Linear(compress_input, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

        # Per-edge action MLP: concat(src_up_emb, dst_down_emb, edge_static) → scalar
        edge_mlp_input = 2 * hidden_dim + self.EDGE_STATIC_DIM
        self.edge_action_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # Transfer state
        self._transfer_data = None

    def set_transfer_topology(self, scenario=None, config_path=None):
        """Configure for zero-shot transfer to a different topology."""
        if config_path is not None:
            # Custom topology: build network from YAML, then extract graph data
            net = SupplyChainNetwork(scenario='custom', config_path=config_path)
            (up_ei, up_ef, down_ei, down_ef,
             main_nodes, node_to_idx, _, edge_pairs, edge_static) = _build_graph_data_from_net(net)
        else:
            # Standard scenario (e.g. 'serial')
            (up_ei, up_ef, down_ei, down_ef,
             main_nodes, node_to_idx, _, edge_pairs, edge_static) = _build_graph_data(scenario)

        device = self.up_edge_index.device
        self._transfer_data = {
            'up_edge_index': up_ei.to(device),
            'up_edge_feat': up_ef.to(device),
            'down_edge_index': down_ei.to(device),
            'down_edge_feat': down_ef.to(device),
            'edge_pairs': edge_pairs,
            'n_main': len(main_nodes),
            'edge_static': edge_static.to(device),
        }

    def clear_transfer(self):
        """Reset to native topology."""
        self._transfer_data = None

    def _get_config(self):
        """Get active topology config."""
        if self._transfer_data is not None:
            d = self._transfer_data
            return (d['up_edge_index'], d['up_edge_feat'],
                    d['down_edge_index'], d['down_edge_feat'],
                    d['n_main'], d['edge_pairs'], d['edge_static'])
        return (self.up_edge_index, self.up_edge_feat,
                self.down_edge_index, self.down_edge_feat,
                self._default_n_main, self._edge_pairs,
                self.edge_static_features)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing topology-invariant graph-level features.
        Also stores per-node embeddings for per-edge action generation.
        """
        batch_size = observations.shape[0]

        (up_ei, up_ef, down_ei, down_ef,
         n_main, edge_pairs, edge_static) = self._get_config()

        # Parse observation: [base_obs | node_feats_grouped | global_feats]
        n_augmented = self.n_node_feats * n_main + self.n_global_feats
        base_dim = observations.shape[1] - n_augmented

        base_obs = observations[:, :base_dim]
        augmented = observations[:, base_dim:]

        node_feats_flat = augmented[:, :self.n_node_feats * n_main]
        global_feats = augmented[:, self.n_node_feats * n_main:]

        # Grouped layout: (batch, n_feats, n_main) → (batch, n_main, n_feats)
        H = node_feats_flat.view(batch_size, self.n_node_feats, n_main).transpose(1, 2)

        # BatchNorm before MPNN
        H_flat = H.reshape(batch_size * n_main, self.n_node_feats)
        H_flat = self.node_bn(H_flat)
        H = H_flat.reshape(batch_size, n_main, self.n_node_feats)

        # Upstream MPNN
        H_up = H
        for layer in self.up_layers:
            H_up = layer(H_up, up_ei, up_ef)

        # Downstream MPNN
        H_down = H
        for layer in self.down_layers:
            H_down = layer(H_down, down_ei, down_ef)

        # Store for per-edge action generation
        self._last_H_up = H_up
        self._last_H_down = H_down
        self._last_edge_pairs = edge_pairs
        self._last_edge_static = edge_static

        # TOPOLOGY-INVARIANT: mean pool instead of flatten
        up_pool = H_up.mean(dim=1)    # (batch, hidden)
        down_pool = H_down.mean(dim=1)  # (batch, hidden)

        combined = torch.cat([up_pool, down_pool, global_feats], dim=1)
        return self.compress(combined)

    def compute_per_edge_actions(self):
        """
        Generate per-edge actions from stored node embeddings.
        Must be called AFTER forward() in the same forward pass.

        For each reorder edge (u, v):
            edge_input = [H_up[u], H_down[v], edge_static_feats]
            action = edge_MLP(edge_input) ∈ [-1, 1]
        """
        H_up = self._last_H_up
        H_down = self._last_H_down
        edge_pairs = self._last_edge_pairs
        edge_static = self._last_edge_static

        batch_size = H_up.shape[0]
        actions = []

        for i, (src_idx, dst_idx) in enumerate(edge_pairs):
            src_emb = H_up[:, src_idx, :]       # upstream embedding of supplier
            dst_emb = H_down[:, dst_idx, :]      # downstream embedding of buyer
            ef = edge_static[i].unsqueeze(0).expand(batch_size, -1)
            edge_input = torch.cat([src_emb, dst_emb, ef], dim=-1)
            action = self.edge_action_mlp(edge_input)  # (batch, 1)
            actions.append(action)

        return torch.cat(actions, dim=-1)  # (batch, n_edges)


def _build_graph_data_from_net(net):
    """Build graph data from an already-constructed SupplyChainNetwork."""
    main_nodes = sorted(
        [n for n in net.graph.nodes()
         if n not in net.market and n not in net.rawmat]
    )
    node_to_idx = {node: i for i, node in enumerate(main_nodes)}
    n_main = len(main_nodes)

    max_L = max([net.graph.edges[e].get('L', 0) for e in net.graph.edges()], default=1) or 1
    max_p = max([net.graph.edges[e].get('p', 0) for e in net.graph.edges()], default=1) or 1
    max_g = max([net.graph.edges[e].get('g', 0) for e in net.graph.edges()], default=1) or 1

    upstream_src, upstream_dst, upstream_feats = [], [], []
    downstream_src, downstream_dst, downstream_feats = [], [], []

    for u, v in net.graph.edges():
        if u in node_to_idx and v in node_to_idx:
            props = net.graph.edges[(u, v)]
            feat = [
                props.get('L', 0) / max_L,
                props.get('p', 0) / max_p,
                props.get('g', 0) / max_g,
            ]
            upstream_src.append(node_to_idx[u])
            upstream_dst.append(node_to_idx[v])
            upstream_feats.append(feat)
            downstream_src.append(node_to_idx[v])
            downstream_dst.append(node_to_idx[u])
            downstream_feats.append(feat)

    for i in range(n_main):
        upstream_src.append(i); upstream_dst.append(i); upstream_feats.append([0., 0., 0.])
        downstream_src.append(i); downstream_dst.append(i); downstream_feats.append([0., 0., 0.])

    up_ei = torch.tensor([upstream_src, upstream_dst], dtype=torch.long)
    up_ef = torch.tensor(upstream_feats, dtype=torch.float32)
    down_ei = torch.tensor([downstream_src, downstream_dst], dtype=torch.long)
    down_ef = torch.tensor(downstream_feats, dtype=torch.float32)

    edge_pairs = []
    edge_static_features = []
    for u, v in net.reorder_links:
        edge_data = net.graph.edges[(u, v)]
        price = edge_data.get('p', 0.0)
        lead_time = float(edge_data.get('L', 0))
        pipeline_hold = edge_data.get('g', 0.0)
        is_rawmat = 1.0 if u in net.rawmat else 0.0
        edge_static_features.append([price, lead_time, pipeline_hold, is_rawmat])

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

    edge_static_features = torch.tensor(edge_static_features, dtype=torch.float32)

    return (up_ei, up_ef, down_ei, down_ef,
            main_nodes, node_to_idx, net, edge_pairs, edge_static_features)
