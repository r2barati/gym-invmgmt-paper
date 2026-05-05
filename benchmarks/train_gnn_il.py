#!/usr/bin/env python3
"""
train_gnn_il.py — Train GNN-IL (Behavioral Cloning -> PPO Finetuning)
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

import gymnasium as gym
from gym_invmgmt import CoreEnv
from gymnasium.wrappers import RescaleAction
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from agents.oracle import OracleAgent
from gym_invmgmt.extractors.gnn_extractor_v3 import GNNFeaturesExtractorV3

MODELS_DIR = os.path.join(PROJECT_ROOT, 'data', 'models')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'training_logs')
os.makedirs(MODELS_DIR, exist_ok=True)

def collect_oracle_data(vec_env, num_steps=50000):
    print(f"Collecting {num_steps} steps of Oracle data...")
    
    obs_list = []
    action_list = []
    
    vec_obs = vec_env.reset()
    current_period = 0
    
    # We train on a non-randomized scenario to ensure perfect Oracle consistency
    base_env = vec_env.envs[0].unwrapped
    oracle = OracleAgent(base_env)
    
    for i in range(num_steps):
        action = oracle.get_action(None, current_period)
        
        # Scale to [-1, 1]
        action_low = base_env.action_space.low
        action_high = base_env.action_space.high
        scaled_action = 2.0 * (action - action_low) / (action_high - action_low) - 1.0
        scaled_action = np.clip(scaled_action, -1.0, 1.0)
        
        # Save vec_obs (array) and scaled action
        obs_list.append(vec_obs[0])
        action_list.append(scaled_action)
        
        # Step env
        vec_obs, rewards, dones, infos = vec_env.step([scaled_action])
        
        current_period += 1
        if dones[0]:
            # No domain randomization, but re-init Oracle just to be safe
            base_env = vec_env.envs[0].unwrapped
            oracle = OracleAgent(base_env)
            current_period = 0
            
        if (i+1) % 10000 == 0:
            print(f"Collected {i+1} steps")
            
    return np.array(obs_list), np.array(action_list)

def train_gnn_il(topology='network'):
    print(f"\n--- Training GNN-IL for topology: {topology} ---")
    
    def make_env():
        # EXACTLY one canonical wrapped training env
        # Non-randomized Oracle-consistent scenario
        base_env = CoreEnv(scenario=topology, demand_config={'type': 'stationary', 'base_mu': 20}, num_periods=30, backlog=True)
        feat_env = DomainFeatureWrapper(base_env, is_blind=False, enhanced=True, grouped=True)
        return RescaleAction(feat_env, min_action=-1.0, max_action=1.0)
        
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
    
    # 1. Collect BC Data
    obs_array, action_array = collect_oracle_data(vec_env, num_steps=50000)
    
    # 2. Init PPO Model with GNN Extractor
    policy_kwargs = dict(
        features_extractor_class=GNNFeaturesExtractorV3,
        features_extractor_kwargs=dict(features_dim=128, scenario=topology),
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )
    
    # Use MlpPolicy directly (since obs is a flat array)
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs, 
                n_steps=2048, batch_size=64, learning_rate=3e-4, device='cpu')
                
    # 3. Behavioral Cloning (Supervised Learning on policy)
    print("Starting Behavioral Cloning Phase...")
    optimizer = optim.Adam(model.policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    obs_tensors = torch.tensor(obs_array, dtype=torch.float32)
    act_tensors = torch.tensor(action_array, dtype=torch.float32)
    # Invariant: obs_array was collected through the same VecNormalize object
    # that PPO fine-tuning continues to use, so these tensors are already in the
    # policy's normalized observation space.
    
    dataset = torch.utils.data.TensorDataset(obs_tensors, act_tensors)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    model.policy.train()
    for epoch in range(10):
        total_loss = 0
        for batch_obs, batch_act in loader:
            optimizer.zero_grad()
            features = model.policy.extract_features(batch_obs)
            latent_pi = model.policy.mlp_extractor.forward_actor(features)
            mean_actions = model.policy.action_net(latent_pi)
            
            loss = loss_fn(mean_actions, batch_act)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10 BC Loss: {total_loss/len(loader):.4f}")
        
    # 4. PPO Finetuning
    print("Starting PPO Finetuning Phase...")
    suffix = 'serial' if topology == 'serial' else ''
    model_name = f'ppo_gnn_il_serial' if topology == 'serial' else 'ppo_gnn_il'
    eval_callback = EvalCallback(vec_env, best_model_save_path=MODELS_DIR,
                                 log_path=LOGS_DIR, eval_freq=10000,
                                 deterministic=True, render=False)
    
    model.learn(total_timesteps=150000, callback=eval_callback)
    
    # Save exactly matching the registry
    os.rename(os.path.join(MODELS_DIR, 'best_model.zip'), os.path.join(MODELS_DIR, f'{model_name}_best.zip'))
    
    vecnorm_name = 'vec_normalize_gnn_il_serial.pkl' if topology == 'serial' else 'vec_normalize_gnn_il.pkl'
    vec_env.save(os.path.join(MODELS_DIR, vecnorm_name))
    print(f"Saved {model_name}_best.zip and {vecnorm_name}")

if __name__ == '__main__':
    train_gnn_il('network')
    train_gnn_il('serial')
