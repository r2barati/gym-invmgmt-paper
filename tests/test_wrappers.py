import pytest
import numpy as np
from gym_invmgmt import CoreEnv
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.wrappers.domain_randomization import DomainRandomizationWrapper
from gym_invmgmt.wrappers.graph_only_wrapper import GraphOnlyWrapper
from gym_invmgmt.wrappers.multi_agent import MultiAgentWrapper
from gym_invmgmt.wrappers.per_link_wrapper import PerLinkFeatureWrapper
from gym_invmgmt.wrappers.residual_action import ProportionalResidualWrapper
from gym_invmgmt.wrappers.residual_graph_wrapper import ResidualGraphWrapper
from gym_invmgmt.wrappers.temporal_frame_stack import TemporalFrameStack

def test_domain_feature_wrapper_v1_v2():
    """Test that DomainFeatureWrapper yields correctly shaped flat arrays."""
    core_env = CoreEnv(scenario="base")
    n_main = len([n for n in core_env.network.graph.nodes() 
                  if n not in core_env.network.market and n not in core_env.network.rawmat])
    base_dim = core_env.observation_space.shape[0]
    
    # V1 (enhanced=False)
    env_v1 = DomainFeatureWrapper(core_env, enhanced=False)
    obs_v1, _ = env_v1.reset()
    expected_dim_v1 = base_dim + (3 * n_main + 2)
    assert obs_v1.shape == (expected_dim_v1,)
    assert env_v1.observation_space.shape == (expected_dim_v1,)
    
    # V2 (enhanced=True, grouped=False)
    env_v2 = DomainFeatureWrapper(core_env, enhanced=True, grouped=False)
    obs_v2, _ = env_v2.reset()
    expected_dim_v2 = base_dim + (8 * n_main + 10)
    assert obs_v2.shape == (expected_dim_v2,)
    assert env_v2.observation_space.shape == (expected_dim_v2,)

def test_blind_wrapper_shapes():
    """Test that is_blind=True produces identical shapes as is_blind=False."""
    core_env = CoreEnv(scenario="base")
    
    env_sighted = DomainFeatureWrapper(core_env, is_blind=False)
    env_blind = DomainFeatureWrapper(core_env, is_blind=True)
    
    obs_s, _ = env_sighted.reset()
    obs_b, _ = env_blind.reset()
    
    assert obs_s.shape == obs_b.shape
    assert env_sighted.observation_space.shape == env_blind.observation_space.shape

def test_temporal_framestack():
    """Test that TemporalFrameStack yields n_history * obs_dim."""
    core_env = CoreEnv(scenario="base")
    base_dim = core_env.observation_space.shape[0]
    n_history = 3
    
    env = TemporalFrameStack(core_env, n_history=n_history)
    obs, _ = env.reset()
    
    expected_dim = base_dim * n_history
    assert obs.shape == (expected_dim,)
    assert env.observation_space.shape == (expected_dim,)


def test_domain_randomization_sets_demand_engine_parameters():
    """Public randomization wrapper should not silently fall back to defaults."""
    core_env = CoreEnv(scenario="base", num_periods=5)
    env = DomainRandomizationWrapper(
        core_env,
        effect_options=("seasonal", "shock"),
        effect_prob=1.0,
        goodwill_options=(False,),
        backlog_options=(True,),
        noise_scale_range=(1.0, 1.0),
        external_series_prob=0.0,
        seasonal_amp=0.3,
        seasonal_freq=0.3,
        shock_time=2,
        shock_mag=1.7,
    )

    obs, _ = env.reset(seed=123)
    engine = env.unwrapped.demand_engine

    assert obs.shape == env.observation_space.shape
    assert engine.effects == ["seasonal", "shock"]
    assert engine.use_goodwill is False
    assert engine.noise_scale == pytest.approx(1.0)
    assert engine.external_series is None
    assert engine.seasonal_amp == pytest.approx(0.3)
    assert engine.seasonal_freq == pytest.approx(0.3)
    assert engine.shock_time == 2
    assert engine.shock_mag == pytest.approx(1.7)
    assert env.unwrapped.backlog is True


def test_domain_randomization_allows_stationary_only_mode():
    """demand_types=['stationary'] should not add non-stationary effects."""
    env = DomainRandomizationWrapper(
        CoreEnv(scenario="base", num_periods=5),
        demand_types=("stationary",),
        effect_prob=1.0,
        goodwill_options=(False,),
        external_series_prob=0.0,
    )

    env.reset(seed=321)

    assert env.unwrapped.demand_engine.effects == []
    assert env.unwrapped.demand_engine.type == "stationary"


def test_per_link_terminal_observation_uses_final_state():
    """Per-link residual observations should not collapse to zero at truncation."""
    core_env = CoreEnv(
        scenario="serial",
        num_periods=3,
        demand_config={"type": "stationary", "base_mu": 5},
    )
    env = PerLinkFeatureWrapper(core_env)
    obs, _ = env.reset(seed=123)

    for _ in range(core_env.num_periods):
        previous_obs = obs.copy()
        obs, _, _, truncated, _ = env.step(np.zeros(env.action_space.shape))

    assert truncated
    assert core_env.period == core_env.num_periods
    assert obs.shape == env.observation_space.shape
    assert not np.allclose(obs, 0.0)
    assert not np.allclose(obs, previous_obs)

    features = obs.reshape(env.n_links, env.FEATURES_PER_LINK)
    heuristic_action_idx = 6
    norm_time_idx = 9
    np.testing.assert_allclose(features[:, heuristic_action_idx], 0.0)
    np.testing.assert_allclose(features[:, norm_time_idx], 1.0)


def test_graph_only_lost_sales_does_not_subtract_prior_unfulfilled():
    """Graph-only transfer features should not treat lost sales as backlog."""
    env = GraphOnlyWrapper(CoreEnv(scenario="serial", num_periods=3, backlog=False))
    env.reset(seed=123)
    core = env.unwrapped
    core.period = 1

    retail = next(n for n in env.main_nodes if n in core.network.retail)
    retail_pos = env.main_nodes.index(retail)
    retail_idx = core.network.node_map[retail]
    core.X[1, retail_idx] = 20.0
    core.Y[1, :] = 0.0
    core.U[0, :] = 7.0

    obs = env._compute_graph_obs()
    assert obs[retail_pos] == pytest.approx(20.0)


def test_multi_agent_terminal_and_lost_sales_contracts():
    """Multi-agent observations should preserve terminal state and lost-sales semantics."""
    core_env = CoreEnv(
        scenario="serial",
        num_periods=3,
        backlog=False,
        demand_config={"type": "stationary", "base_mu": 5},
    )
    env = MultiAgentWrapper(core_env)
    obs, _ = env.reset(seed=123)

    for _ in range(core_env.num_periods):
        obs, _, _, truncated, _ = env.step(np.zeros(env.action_space.shape))

    assert truncated
    assert core_env.period == core_env.num_periods
    assert obs.shape == env.observation_space.shape
    assert not np.allclose(obs, 0.0)

    core_env.period = 1
    core_env.U[0, :] = 9.0
    retail = next(n for n in env.agent_nodes if n in core_env.network.retail)
    retail_agent_pos = env.agent_nodes.index(retail)
    local_obs = env._build_local_obs()
    backlog_feature_idx = retail_agent_pos * 5 + 2
    assert local_obs[backlog_feature_idx] == pytest.approx(0.0)


def test_residual_graph_observation_space_matches_heuristic_features():
    """Residual graph wrapper should declare the heuristic-action feature tail."""
    class OnesHeuristic:
        def get_action(self, obs, t):
            return np.ones(len(core_env.network.reorder_links))

    core_env = CoreEnv(scenario="base", num_periods=3)
    env = ResidualGraphWrapper(core_env, heuristic_agent=OnesHeuristic())
    obs, _ = env.reset(seed=123)

    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_residual_graph_and_proportional_actions_are_clipped():
    """Residual wrappers should not pass requests beyond CoreEnv action bounds."""
    class HugeHeuristic:
        def __init__(self, core):
            self.core = core

        def get_action(self, obs, t):
            return self.core.action_space.high * 10.0

    core_env = CoreEnv(scenario="base", num_periods=3)
    graph_env = ResidualGraphWrapper(core_env, heuristic_agent=HugeHeuristic(core_env))
    graph_env.reset(seed=123)
    graph_env.step(np.ones(graph_env.action_space.shape) * 1e6)
    assert np.all(core_env.action_log[0] <= core_env.action_space.high)

    core_env = CoreEnv(scenario="base", num_periods=3)
    prop_env = ProportionalResidualWrapper(core_env, heuristic_agent=HugeHeuristic(core_env))
    prop_env.reset(seed=123)
    prop_env.step(np.ones(prop_env.action_space.shape) * 1e6)
    assert np.all(core_env.action_log[0] <= core_env.action_space.high)


def test_per_link_passes_current_state_to_heuristic():
    """Per-link wrapper should pass env.state, not _update_state()'s None return."""
    class CaptureHeuristic:
        def __init__(self):
            self.last_obs = None

        def get_action(self, obs, t):
            self.last_obs = obs
            return np.zeros(len(core_env.network.reorder_links))

    core_env = CoreEnv(scenario="base", num_periods=3)
    heuristic = CaptureHeuristic()
    env = PerLinkFeatureWrapper(core_env, heuristic_agent=heuristic)
    env.reset(seed=123)

    assert heuristic.last_obs is core_env.state
    assert heuristic.last_obs is not None
