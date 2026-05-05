"""
transformer_temporal_extractor.py — Spatio-Temporal Transformer Extractor (ST-PPO).

Extends the node-token Transformer (PPO-Transformer) with true temporal self-attention
by operating on frame-stacked observations from TemporalFrameStack.

Architecture:
  Each (node, timestep) pair becomes a separate token. With n_main=6 nodes
  and n_history=4 frames, the sequence length is 24 tokens. Self-attention
  operates over all 24 tokens simultaneously, capturing both:
    - Spatial dependencies (which node is upstream of which)
    - Temporal dependencies (how did each node's state evolve over time)

Token layout:
  [node0_t-3, node1_t-3, ..., nodeN_t-3,   # oldest frame
   node0_t-2, node1_t-2, ..., nodeN_t-2,
   node0_t-1, node1_t-1, ..., nodeN_t-1,
   node0_t,   node1_t,   ..., nodeN_t]     # current frame

References:
  - GTrXL (Parisotto et al., 2020): Gated Transformer-XL for RL
  - Decision Transformer (Chen et al., 2021): Sequence modeling for RL
"""
import gymnasium as gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.network_topology import SupplyChainNetwork


class TransformerTemporalExtractor(BaseFeaturesExtractor):
    """
    Spatio-temporal Transformer feature extractor (ST-PPO).

    Operates on frame-stacked observations from TemporalFrameStack.
    Each historical snapshot of each node becomes a separate token with
    both node-position and time-position encodings.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Must reflect the stacked observation (single_obs_dim × n_history).
    features_dim : int
        Output feature dimension.
    scenario : str
        'base' or 'serial' — determines topology and n_main.
    n_history : int
        Number of stacked frames (must match TemporalFrameStack).
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of Transformer encoder layers.
    dropout : float
        Dropout rate.
    n_node_feats : int or None
        Per-node feature count from DomainFeatureWrapper.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 scenario: str = 'base',
                 n_history: int = 4,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 n_node_feats: int = None):
        super().__init__(observation_space, features_dim)

        # Network topology
        net = SupplyChainNetwork(scenario=scenario)
        self.main_nodes = sorted([n for n in net.graph.nodes()
                                   if n not in net.market and n not in net.rawmat])
        self.n_main = len(self.main_nodes)
        self.n_history = n_history

        # Feature dimensions (per single frame)
        if n_node_feats is None:
            n_node_feats = DomainFeatureWrapper.V2_NODE_FEATS  # 8
        n_global_feats = DomainFeatureWrapper.V2_GLOBAL_FEATS  # 10
        n_augmented = n_node_feats * self.n_main + n_global_feats

        # Compute single-frame obs dim from stacked obs
        total_stacked_dim = observation_space.shape[0]
        single_obs_dim = total_stacked_dim // n_history
        self.single_obs_dim = single_obs_dim
        self.base_dim = single_obs_dim - n_augmented
        self.n_node_feats = n_node_feats
        self.n_global_feats = n_global_feats

        # Total per-token features = node feats + global feats (broadcast)
        token_dim = n_node_feats + n_global_feats

        # Input projection: token_dim → d_model
        self.input_proj = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learnable node position encoding (one per node, shared across time)
        self.node_pos_encoding = nn.Parameter(
            torch.randn(1, self.n_main, d_model) * 0.02
        )

        # Learnable temporal position encoding (one per history slot)
        self.time_pos_encoding = nn.Parameter(
            torch.randn(1, n_history, d_model) * 0.02
        )

        # Learnable node-type embedding
        # Types: factory=0, distributor=1, retailer=2
        self.type_embedding = nn.Embedding(3, d_model)
        node_types = []
        for n in self.main_nodes:
            if n in net.factory:
                node_types.append(0)
            elif n in net.retail:
                node_types.append(2)
            else:
                node_types.append(1)
        self.register_buffer('node_types', torch.tensor(node_types, dtype=torch.long))

        # Transformer encoder — operates on (n_history × n_main) tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output: use only the CURRENT frame's node representations + base obs
        # This keeps the output dimension identical to PPO-Transformer
        self.output_net = nn.Sequential(
            nn.Linear(d_model * self.n_main + self.base_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
            nn.GELU(),
        )

    def _parse_single_frame(self, frame: torch.Tensor):
        """Parse a single frame into (base_obs, node_tokens, global_feats).

        Parameters
        ----------
        frame : torch.Tensor, shape (batch, single_obs_dim)

        Returns
        -------
        base_obs : (batch, base_dim)
        node_feats : (batch, n_main, n_node_feats)
        global_feats : (batch, n_global_feats)
        """
        base_obs = frame[:, :self.base_dim]
        augmented = frame[:, self.base_dim:]

        node_feats_flat = augmented[:, :self.n_node_feats * self.n_main]
        # Grouped layout: [all_feat0, all_feat1, ...] → (batch, n_feats, n_main)
        node_feats = node_feats_flat.view(-1, self.n_node_feats, self.n_main)
        node_feats = node_feats.permute(0, 2, 1)  # (batch, n_main, n_node_feats)

        global_feats = augmented[:, self.n_node_feats * self.n_main:]

        return base_obs, node_feats, global_feats

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split stacked observation into individual frames
        # observations shape: (batch, single_obs_dim * n_history)
        frames = observations.view(batch_size, self.n_history, self.single_obs_dim)

        all_tokens = []
        base_obs_current = None

        for t in range(self.n_history):
            frame = frames[:, t, :]  # (batch, single_obs_dim)
            base_obs, node_feats, global_feats = self._parse_single_frame(frame)

            # Keep the base_obs from the CURRENT (last) frame
            if t == self.n_history - 1:
                base_obs_current = base_obs

            # Broadcast global features to all nodes
            global_broadcast = global_feats.unsqueeze(1).expand(-1, self.n_main, -1)

            # Build tokens for this time step: (batch, n_main, token_dim)
            tokens = torch.cat([node_feats, global_broadcast], dim=-1)

            # Project to d_model
            tokens = self.input_proj(tokens)  # (batch, n_main, d_model)

            # Add node position encoding (shared across time steps)
            tokens = tokens + self.node_pos_encoding

            # Add type embedding
            tokens = tokens + self.type_embedding(self.node_types).unsqueeze(0)

            # Add temporal position encoding (broadcast to all nodes)
            time_enc = self.time_pos_encoding[:, t, :].unsqueeze(1)  # (1, 1, d_model)
            tokens = tokens + time_enc

            all_tokens.append(tokens)

        # Concatenate all time steps: (batch, n_history * n_main, d_model)
        all_tokens = torch.cat(all_tokens, dim=1)

        # Transformer self-attention over all spatio-temporal tokens
        all_tokens = self.transformer(all_tokens)

        # Extract ONLY the current frame's node representations (last n_main tokens)
        current_tokens = all_tokens[:, -self.n_main:, :]  # (batch, n_main, d_model)
        pooled = current_tokens.reshape(batch_size, -1)   # (batch, n_main * d_model)

        # Combine with current frame's base observation
        combined = torch.cat([pooled, base_obs_current], dim=-1)

        return self.output_net(combined)
