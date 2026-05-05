import pytest
import re
import os
import pandas as pd

from benchmarks.run_benchmarks import ALL_AGENT_IDS

def test_roster_consistency_counts():
    """Check ALL_AGENT_IDS matches the 29 count claimed in README."""
    assert len(ALL_AGENT_IDS) == 29, f"Expected 29 agents in ALL_AGENT_IDS, found {len(ALL_AGENT_IDS)}"

def test_readme_claims():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        pytest.skip("README.md not found")
        
    with open(readme_path, "r") as f:
        content = f.read()
        
    assert "29 registered IDs total" in content or "29 configurations" in content or "28 non-Oracle" in content, \
        "README does not explicitly state the 29 agent count as required."

def test_csv_prefixes_match_roster():
    csv_path = "results/benchmark_final_merged.csv"
    if not os.path.exists(csv_path):
        csv_path = "../results/benchmark_final_merged.csv"
        
    if not os.path.exists(csv_path):
        pytest.skip("CSV not found")
        
    df = pd.read_csv(csv_path)
    
    # Extract all agent prefixes from CSV columns (e.g., "Oracle" from "Oracle_Profit")
    csv_agents = set()
    for col in df.columns:
        if "_Profit" in col:
            prefix = col.split("_Profit")[0]
            # Handle the fact that CSV might use underscores instead of hyphens
            # e.g. PPO_GNN vs PPO-GNN
            csv_agents.add(prefix)
            
    # We won't strictly enforce a 1:1 match because ST-PPO-B and GNN-IL might be
    # missing from an input CSV, but we CAN enforce that every CSV agent is in the roster.
    
    normalized_roster = {a.replace('-', '_') for a in ALL_AGENT_IDS}
    
    for csv_agent in csv_agents:
        # Ignore supplementary LLM agents if they are in the CSV
        if csv_agent in ["LLM_ZS", "LLM_ZS_Direct", "LLM_Policy_C", "LLM_InvAgent_C", "LLM_InvAgent_D"]:
            continue
            
        # Special mapping rules for published CSV column names
        if csv_agent == "sS": 
            continue # Maps to (s,S)
        if csv_agent == "sS_I":
            continue # Maps to (s,S)-I
        if csv_agent == "ExpSmooth_I":
            continue # Maps to ExpSmoothing-I
        if csv_agent == "GNN_V3":
            continue # Maps to PPO-GNN
        if csv_agent == "GNN_V3_B":
            continue # Maps to PPO-GNN-B
            
        assert csv_agent in normalized_roster, f"Agent {csv_agent} found in CSV but not in ALL_AGENT_IDS!"
