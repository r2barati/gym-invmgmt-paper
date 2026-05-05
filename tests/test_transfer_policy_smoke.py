import pytest
import torch
import numpy as np
from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.extractors.transfer_per_edge_policy import TransferPerEdgePolicy
from gym_invmgmt.extractors.residual_gcn_pool_extractor import ResidualGCNPoolExtractor

def test_transfer_per_edge_policy_dimension_handling():
    """
    Smoke test to ensure the TransferPerEdgePolicy correctly handles forward passes
    and dimension padding/trimming when the environment topology changes.
    """
    # 1. Training Environment (base topology)
    env_train_core = CoreEnv(scenario="base", num_periods=5)
    env_train = DomainFeatureWrapper(env_train_core, enhanced=True, grouped=True)
    obs_train, _ = env_train.reset()
    
    # 2. Instantiate policy configured for the training environment
    features_dim = 128
    policy = TransferPerEdgePolicy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        lr_schedule=lambda _: 1e-4,
        extractor_class=ResidualGCNPoolExtractor,
        extractor_kwargs={
            "features_dim": features_dim,
            "scenario": "base",
            "hidden_dim": 64,
        }
    )
    
    # 3. Test forward pass on training topology
    obs_train_tensor = torch.as_tensor(obs_train, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        actions_train, values, log_probs = policy.forward(obs_train_tensor, deterministic=True)
    
    assert actions_train.shape[1] == env_train.action_space.shape[0]
    
    # 4. Transfer Environment (serial topology)
    # The serial topology has a different number of nodes and edges
    env_transfer_core = CoreEnv(scenario="serial", num_periods=5)
    env_transfer = DomainFeatureWrapper(env_transfer_core, enhanced=True, grouped=True)
    obs_transfer, _ = env_transfer.reset()
    
    # Must update the extractor's reference to the new environment
    policy.features_extractor.set_transfer_topology(scenario="serial")
    
    obs_transfer_tensor = torch.as_tensor(obs_transfer, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        actions_transfer, values, log_probs = policy.forward(obs_transfer_tensor, deterministic=True)
        
    # The policy is designed to return an action vector matching the training topology's shape
    # (pad or trim) when forward() is called by PPO, but when used via predict() for evaluation, 
    # it is supposed to return the actions matching the CURRENT topology.
    eval_actions, _ = policy.predict(obs_transfer, deterministic=True)
    
    # eval_actions should match the transfer topology's action space shape!
    assert eval_actions.shape[0] == env_transfer.action_space.shape[0]
    
    # Ensure it doesn't crash on env_transfer step
    env_transfer.step(eval_actions)
