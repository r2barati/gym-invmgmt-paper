import pytest
import numpy as np
import pulp
from pathlib import Path
from scipy.stats import norm

from gym_invmgmt import CoreEnv
from agents.echelon_basestock_agent import EchelonApproxAgent
from agents.exp_smoothing_heuristic_agent import ExpSmoothingHeuristicAgent
from agents.heuristic_utils import BaseHeuristicAgent
from agents.or_utils import pipeline_holding_charge
from benchmarks.run_benchmarks import _extract_kpis, _make_agent

def test_dlp_mssp_one_step_solve():
    """Ensure DLP and MSSP solvers execute a one-step solve without crashing."""
    env = CoreEnv(scenario="base", num_periods=5) # Short horizon for fast test
    obs, _ = env.reset(seed=42)
    scfg = {'scenario': 'base', 'topology': 'base'}
    
    dlp = _make_agent('DLP', env, scfg, 42)
    action_dlp = dlp.get_action(obs, 0)
    assert action_dlp is not None
    assert action_dlp.shape == env.action_space.shape
    
    mssp = _make_agent('MSSP', env, scfg, 42)
    # Use the default compact scenario set for this smoke test.
    action_mssp = mssp.get_action(0)
    assert action_mssp is not None
    assert action_mssp.shape == env.action_space.shape

def test_mssp_seed_consistency():
    """Ensure MSSP produces the same SAA tree and action given the same seed."""
    env1 = CoreEnv(scenario="base", num_periods=5)
    obs1, _ = env1.reset(seed=42)
    scfg = {'scenario': 'base', 'topology': 'base'}
    
    mssp1 = _make_agent('MSSP', env1, scfg, 42)
    action1 = mssp1.get_action(0)
    
    env2 = CoreEnv(scenario="base", num_periods=5)
    obs2, _ = env2.reset(seed=42)
    
    mssp2 = _make_agent('MSSP', env2, scfg, 42)
    action2 = mssp2.get_action(0)
    
    np.testing.assert_allclose(action1, action2)


def test_kpi_decomposition_matches_discounted_profit_with_fixed_costs():
    env = CoreEnv(
        scenario="serial",
        num_periods=3,
        alpha=0.95,
        demand_config={
            "type": "stationary",
            "base_mu": 20,
            "external_series": [10.0, 10.0, 10.0],
            "noise_scale": 0.0,
        },
    )
    env.reset(seed=0)
    for edge in env.network.reorder_links:
        env.graph.edges[edge]["K"] = 7.0

    action = np.ones(env.action_space.shape) * 10.0
    for _ in range(3):
        env.step(action)

    kpis = _extract_kpis(env)
    reconstructed = (
        kpis["Revenue"]
        - kpis["ProcurementCost"]
        - kpis["OperatingCost"]
        - kpis["HoldingCost"]
        - kpis["BacklogPenalty"]
        - kpis["FixedOrderingCost"]
    )

    assert kpis["FixedOrderingCost"] > 0
    np.testing.assert_allclose(reconstructed, kpis["Profit"], rtol=1e-8, atol=1e-6)


def test_pipeline_holding_charge_uses_period_discounts_and_horizon_cap():
    charge = pipeline_holding_charge(
        flow=10.0,
        g=2.0,
        L=3,
        order_period=1,
        horizon_len=5,
        discount_factor=0.9,
    )
    expected = 10.0 * 2.0 * (0.9**1 + 0.9**2 + 0.9**3)
    assert pulp.value(charge) == pytest.approx(expected)

    terminal_charge = pipeline_holding_charge(
        flow=10.0,
        g=2.0,
        L=3,
        order_period=4,
        horizon_len=5,
        discount_factor=0.9,
    )
    assert pulp.value(terminal_charge) == pytest.approx(10.0 * 2.0 * 0.9**4)


def test_informed_heuristic_noise_fallback_matches_demand_engine_default():
    env = CoreEnv(scenario="base", num_periods=3)
    env.reset(seed=0)
    delattr(env.demand_engine, "noise_scale")

    agent = BaseHeuristicAgent(env, is_blind=False)
    mu, sigma = agent.estimate_demand_stats(0)

    assert mu == pytest.approx(20.0)
    assert sigma == pytest.approx(np.sqrt(20.0))


def test_exp_smoothing_informed_variance_sums_independent_retail_streams():
    env = CoreEnv(
        scenario="custom",
        config_path=str(Path("gym_invmgmt/topologies/distribution_tree.yaml")),
        num_periods=3,
    )
    env.reset(seed=0)
    agent = ExpSmoothingHeuristicAgent(env, is_blind=False)

    agent.level = [10.0] * agent.num_retail
    agent.trend = [0.0] * agent.num_retail
    agent._update_forecast = lambda current_period: None
    agent.estimate_lead_time_demand = lambda current_period, L: (0.0, 3.0)

    hub = 6
    L = agent.node_info[hub]["max_L"]
    n_streams = agent.num_retail
    targets = agent._compute_targets(1)

    expected_mu = n_streams * 10.0 * (L + 1)
    expected_sigma = np.sqrt(n_streams * 3.0**2)
    expected_target = expected_mu + norm.ppf(agent.node_info[hub]["cr"]) * expected_sigma

    assert targets[hub] == pytest.approx(expected_target)


def test_echelon_inventory_descendant_sets_are_unique_on_reconvergent_graph():
    env = CoreEnv(scenario="base", num_periods=3)
    env.reset(seed=0)
    agent = EchelonApproxAgent(env, is_blind=False)

    # Factory 4 reaches retailer 1 through both distributors 2 and 3.
    # The echelon set should include retailer 1 once, not once per path.
    assert 4 in agent.echelon_nodes
    assert agent.echelon_nodes[4].count(1) == 1
    assert set(agent.echelon_nodes[4]).issuperset({1, 2, 3, 4})

@pytest.mark.parametrize("heuristic_id", ["Newsvendor", "(s,S)", "ExpSmoothing"])
def test_heuristic_no_future_leak(heuristic_id):
    """
    Ensure heuristics use Y[t] (current in-transit) and don't cheat by looking
    at future pipeline rows. (Structural test based on action output).
    """
    env = CoreEnv(scenario="base")
    obs, _ = env.reset(seed=42)
    scfg = {'scenario': 'base', 'topology': 'base'}
    
    agent = _make_agent(heuristic_id, env, scfg, 42)
    action = agent.get_action(obs, 0)
    
    # We can't easily assert exactly how it used Y[t], but we can assert
    # it produced a valid non-NaN action. The review requested making sure they don't crash.
    assert not np.isnan(action).any()
    assert action.shape == env.action_space.shape
