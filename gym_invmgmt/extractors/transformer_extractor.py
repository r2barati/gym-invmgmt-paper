"""
transformer_extractor.py — Transformer-based Feature Extractor for PPO-Transformer.

Replaces GNN's spatial message passing with Transformer self-attention.

Key innovation:
  - Treats each supply chain node as a "token" in a sequence
  - Uses positional encoding to maintain node identity
  - Self-attention captures spatial dependencies across the supply chain graph
  - Contextual State Encoding: node features include demand history as a
    per-node temporal context, allowing the Transformer to learn seasonal
    patterns, trend detection, and shock recovery

Architecture:
  Node features → Linear projection → Positional encoding →
  N × Transformer Encoder Layers → Pool → FC → features_dim
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import math

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.network_topology import SupplyChainNetwork


class TransformerFeaturesExtractor(BaseFeaturesExtractor):
    """
    Transformer-based feature extractor (PPO-Transformer / Node-Token).
    
    Treats each node as a token with contextual state encoding:
      - Per-node features: inv_pos, lt_target, gap, on_hand, h_cost, capacity, is_factory, is_retail
      - Global features (broadcast to all tokens): demand_vel, time features, demand history
      - Structural encoding: learned node-type embeddings
    """
    
    def __init__(self, observation_space: gym.spaces.Box,
                 features_dim: int = 256,
                 scenario: str = 'base',
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
        
        # Feature dimensions
        if n_node_feats is None:
            n_node_feats = DomainFeatureWrapper.V2_NODE_FEATS  # 8
        n_global_feats = DomainFeatureWrapper.V2_GLOBAL_FEATS  # 10
        n_augmented = n_node_feats * self.n_main + n_global_feats
        self.base_dim = observation_space.shape[0] - n_augmented
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
        
        # Learnable positional encoding (one per node)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_main, d_model) * 0.02)
        
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
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output: pool node representations + base obs → features_dim
        self.output_net = nn.Sequential(
            nn.Linear(d_model * self.n_main + self.base_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
            nn.Linear(features_dim, features_dim),
            nn.GELU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split observation: [base_obs | node_feats_grouped | global_feats]
        base_obs = observations[:, :self.base_dim]
        augmented = observations[:, self.base_dim:]
        
        # Extract grouped node features: shape (batch, n_main, n_node_feats)
        node_feats_flat = augmented[:, :self.n_node_feats * self.n_main]
        node_feats = node_feats_flat.view(batch_size, self.n_node_feats, self.n_main)
        node_feats = node_feats.permute(0, 2, 1)  # (batch, n_main, n_node_feats)
        
        # Extract global features: shape (batch, n_global_feats)
        global_feats = augmented[:, self.n_node_feats * self.n_main:]
        
        # Broadcast global features to all nodes: (batch, n_main, n_global_feats)
        global_broadcast = global_feats.unsqueeze(1).expand(-1, self.n_main, -1)
        
        # Build tokens: (batch, n_main, token_dim)
        tokens = torch.cat([node_feats, global_broadcast], dim=-1)
        
        # Project to d_model
        tokens = self.input_proj(tokens)  # (batch, n_main, d_model)
        
        # Add positional encoding + type embedding
        tokens = tokens + self.pos_encoding
        tokens = tokens + self.type_embedding(self.node_types).unsqueeze(0)
        
        # Transformer self-attention
        tokens = self.transformer(tokens)  # (batch, n_main, d_model)
        
        # Flatten node representations + concat base obs
        pooled = tokens.reshape(batch_size, -1)  # (batch, n_main * d_model)
        combined = torch.cat([pooled, base_obs], dim=-1)
        
        return self.output_net(combined)
