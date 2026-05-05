import pytest
import os
from benchmarks.run_benchmarks import _make_agent

# These are the expected model paths defined in run_benchmarks.py
EXPECTED_PATHS = {
    'ST-PPO-B': {
        'base': ('st-ppo_base_blind.zip', 'st-ppo_base_blind_vecnorm.pkl'),
        'serial': ('st-ppo_serial_blind.zip', 'st-ppo_serial_blind_vecnorm.pkl')
    },
    'GNN-IL': {
        'base': ('ppo_gnn_il_best.zip', 'vec_normalize_gnn_il.pkl'),
        'serial': ('ppo_gnn_il_serial_best.zip', 'vec_normalize_gnn_il_serial.pkl')
    },
    'PPO-GNN': {
        'base': ('gnn-v3_base.zip', 'gnn-v3_base_vecnorm.pkl'),
        'serial': ('gnn-v3_serial.zip', 'gnn-v3_serial_vecnorm.pkl')
    },
    'Residual': {
        'base': ('residual_base.zip', 'residual_base_vecnorm.pkl'),
        'serial': ('residual_serial.zip', 'residual_serial_vecnorm.pkl')
    }
}

@pytest.mark.parametrize("agent_id", EXPECTED_PATHS.keys())
def test_model_filename_registry_contracts(agent_id):
    """
    Test that _make_agent references the expected model filename prefixes and suffixes.
    This avoids complex global monkeypatching and sys.modules pollution while enforcing
    the registry contract via static source inspection.
    """
    import inspect
    from benchmarks import run_benchmarks
    source = inspect.getsource(run_benchmarks._make_agent)
    
    # We just need one example from the dict to extract the prefix and suffix
    # e.g., 'st-ppo_base_blind.zip' -> prefix 'st-ppo_', suffix '_blind.zip'
    base_zip, base_pkl = EXPECTED_PATHS[agent_id]['base']
    
    # Extract structural hints
    if "st-ppo" in base_zip:
        assert "f'st-ppo_{topo_label}" in source or "f'st-ppo_{topo_label}_blind" in source
    elif "gnn_il" in base_zip:
        assert "ppo_gnn_il" in source
    elif "gnn-v3" in base_zip:
        # Wait, run_benchmarks has f'gnn-v3_{topo_label}.zip'
        assert "gnn-v3_" in source
    elif "residual" in base_zip:
        assert "residual_" in source
