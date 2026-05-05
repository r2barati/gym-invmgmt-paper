import pytest
import os
import csv
import pandas as pd

from benchmarks.run_benchmarks import ALL_AGENT_IDS

def get_csv_path():
    candidates = [
        "results/benchmark_final_merged.csv",
        "../results/benchmark_final_merged.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def test_canonical_csv_exists():
    assert get_csv_path() is not None, "Canonical benchmark_final_merged.csv is missing!"

def test_csv_agent_columns_exist():
    csv_path = get_csv_path()
    if not csv_path:
        pytest.skip("CSV missing")
        
    df = pd.read_csv(csv_path)
    
    # Every agent in ALL_AGENT_IDS must have a _Profit column in the CSV.
    # PPO-MLP-raw is base-topology-only, so it will have NaNs on serial rows,
    # but the column itself must exist.
    missing_agents = []
    
    for agent in ALL_AGENT_IDS:
        # Published column-name mappings (hyphens → underscores, special chars)
        header_base = agent.replace('-', '_')
        if agent == "(s,S)": header_base = "sS"
        if agent == "(s,S)-I": header_base = "sS_I"
        if agent == "ExpSmoothing-I": header_base = "ExpSmooth_I"
        if agent == "PPO-GNN": header_base = "GNN_V3"
        if agent == "PPO-GNN-B": header_base = "GNN_V3_B"
        
        if f"{header_base}_Profit" not in df.columns and f"{agent}_Profit" not in df.columns:
            missing_agents.append(agent)
            
    assert not missing_agents, f"CSV is missing _Profit columns for registered agents: {missing_agents}"

def test_csv_row_counts_and_nans():
    csv_path = get_csv_path()
    if not csv_path:
        pytest.skip("CSV missing")
        
    df = pd.read_csv(csv_path)
    
    # Expected total rows: 26.
    assert len(df) == 26, f"Expected 26 rows in canonical CSV, found {len(df)}"
    
    # 22 main grid rows (excluding MARL)
    main_grid = df[df.get('MARL', False) != True]
    assert len(main_grid) == 22, f"Expected 22 main grid rows, found {len(main_grid)}"
    
    # Ensure no entirely NaN rows for Profit values (ignoring the scenario metadata columns)
    profit_cols = [c for c in df.columns if "_Profit" in c]
    for i, row in df.iterrows():
        # If all profit columns are NaN, this is a bad row
        if row[profit_cols].isna().all():
            pytest.fail(f"Row {i} is entirely NaN for all profit columns.")
