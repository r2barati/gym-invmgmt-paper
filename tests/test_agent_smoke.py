import pytest
import os
import numpy as np

from gym_invmgmt import CoreEnv
from benchmarks.run_benchmarks import ALL_AGENT_IDS, _make_agent

@pytest.mark.parametrize("agent_id", ALL_AGENT_IDS)
@pytest.mark.parametrize("topology", ["base", "serial"])
def test_agent_instantiation(agent_id, topology):
    """
    Instantiation smoke test: verify every registered agent can be loaded
    from its trained weights on both topologies without import or shape errors.
    This does NOT step the environment or validate action bounds.
    """
    env = CoreEnv(scenario=topology, num_periods=30, backlog=True)
    scfg = {'scenario': topology, 'topology': topology, 'network': topology, 'demand_config': {'type': 'stationary', 'base_mu': 20}, 'backlog': True}
    
    try:
        agent = _make_agent(agent_id, env, scfg, 42)
    except Exception as e:
        if "No module named" in str(e) or "not found" in str(e):
            pytest.skip(f"Agent {agent_id} failed to load dependencies or weights: {e}")
        else:
            raise e
            
    if agent is None:
        pytest.skip(f"Agent {agent_id} for {topology} returned None (missing weights or topology-specific exclusion).")
        
    assert agent is not None


@pytest.mark.parametrize("agent_id", ALL_AGENT_IDS)
@pytest.mark.parametrize("topology", ["base", "serial"])
def test_agent_one_step_forward(agent_id, topology):
    """
    Forward pass smoke test: verifies that the agent can successfully execute
    its get_action() method and that the returned action is valid for env.step().
    """
    env = CoreEnv(scenario=topology, num_periods=30, backlog=True)
    # Include all required config keys to match run_benchmarks
    scfg = {'scenario': topology, 'topology': topology, 'network': topology, 'demand_config': {'type': 'stationary', 'base_mu': 20}, 'backlog': True}
    
    try:
        agent = _make_agent(agent_id, env, scfg, 42)
    except Exception as e:
        if "No module named" in str(e) or "not found" in str(e):
            pytest.skip(f"Agent {agent_id} failed to load dependencies or weights: {e}")
        else:
            raise e

    if agent is None:
        pytest.skip(f"Agent {agent_id} for {topology} returned None (missing weights or topology-specific exclusion).")

    obs, _ = env.reset(seed=42)
    
    # Action dispatch matching production logic
    if agent.__class__.__name__ == "RollingHorizonMSSPAgent":
        action = agent.get_action(0)
    else:
        action = agent.get_action(obs, 0)
        
    assert action is not None, f"Agent {agent_id} returned None action"
    
    action_arr = np.asarray(action)
    assert np.all(np.isfinite(action_arr)), f"Agent {agent_id} returned non-finite action: {action_arr}"
    assert action_arr.shape == env.action_space.shape, f"Agent {agent_id} returned shape {action_arr.shape}, expected {env.action_space.shape}"
    
    # Tolerance for floating point bounds
    tol = 1e-5
    assert np.all(action_arr >= env.action_space.low - tol), f"Agent {agent_id} action {action_arr} below lower bound {env.action_space.low}"
    assert np.all(action_arr <= env.action_space.high + tol), f"Agent {agent_id} action {action_arr} above upper bound {env.action_space.high}"
    
    # Verify environment doesn't crash on step
    env.step(action_arr)


@pytest.mark.llm
@pytest.mark.parametrize("agent_id", ["LLM-ZS", "LLM-Policy-C"])
@pytest.mark.parametrize("topology", ["base", "serial"])
def test_llm_agent_one_step_forward(agent_id, topology):
    """
    Separate smoke test for LLM agents to keep CI fast.
    """
    if os.environ.get("GYM_INVMGMT_RUN_LLM_TESTS") != "1":
        pytest.skip("Set GYM_INVMGMT_RUN_LLM_TESTS=1 to run local LLM smoke tests.")

    env = CoreEnv(scenario=topology, num_periods=30, backlog=True)
    scfg = {'scenario': topology, 'topology': topology, 'network': topology, 'demand_config': {'type': 'stationary', 'base_mu': 20}, 'backlog': True}
    
    try:
        agent = _make_agent(agent_id, env, scfg, 42)
    except Exception as e:
        pytest.skip(f"LLM agent {agent_id} failed to load: {e}")

    if agent is None:
        pytest.skip(f"Agent {agent_id} returned None.")

    obs, _ = env.reset(seed=42)
    
    action = agent.get_action(obs, 0)
    
    assert action is not None
    action_arr = np.asarray(action)
    assert np.all(np.isfinite(action_arr))
    assert action_arr.shape == env.action_space.shape
    env.step(action_arr)
