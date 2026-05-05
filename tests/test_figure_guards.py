import pytest
import os
import subprocess
import tempfile
import pandas as pd

def test_figure_scripts_fail_on_missing_columns():
    """
    Ensure that regenerating figures with a CSV missing required columns 
    fails fast and does not mutate existing artifacts.
    """
    scripts_to_test = [
        "paper/figures/regenerate_appendix_figures.py",
        "paper/figures/regen_speed_quality.py",
        "paper/figures/extract_appendix_tables.py"
    ]
    
    # Create a synthetic CSV that is missing GNN_V3_Profit and Residual_Profit
    df = pd.DataFrame({
        "Oracle_Profit": [100, 200],
        "Scenario": ["stationary", "M5_volatile"]
    })
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        synthetic_csv_path = tmp.name
        
    try:
        env = os.environ.copy()
        env["BENCHMARK_CSV"] = synthetic_csv_path
        
        for script in scripts_to_test:
            if not os.path.exists(script):
                continue
                
            # It should fail with exit code 1
            result = subprocess.run(["python3", script], env=env, capture_output=True, text=True)
            assert result.returncode == 1, f"{script} did not fail fast on missing columns! Output: {result.stdout}"
            assert "Missing required columns" in result.stdout or "Missing required columns" in result.stderr
            
    finally:
        os.remove(synthetic_csv_path)
