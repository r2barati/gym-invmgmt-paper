"""Contracts for benchmarks/evaluate_custom.py.

The evaluator compares a new agent against the cached per-seed baselines
without re-running them, which is only valid if its episode loop reproduces
the canonical runner exactly.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks import evaluate_custom as ec

rb = ec.rb
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_POLICY = f"{PROJECT_ROOT / 'benchmarks' / 'example_custom_agent.py'}:MovingAverageBaseStock"


def _scenario(block, network='base'):
    return next(s for s in rb.build_all_scenarios()
                if s['block'] == block and s['network'] == network)


def _env(network='base'):
    return rb.CoreEnv(**rb._env_kwargs(_scenario('A_Core', network)))


@pytest.mark.parametrize('block', ['A_Core', 'B_PaperReplication', 'C_MARL', 'D_WalmartM5'])
def test_episode_runner_reproduces_published_baseline(block):
    cache = pd.read_csv(rb._cache_path('Newsvendor')).set_index(['ScenarioKey', 'Seed'])
    scfg = _scenario(block)
    for seed in rb.CANONICAL_SEEDS[:2]:
        kpis = ec.run_episode(lambda env, s, sd: rb._make_agent('Newsvendor', env, s, sd), scfg, seed)
        ref = cache.loc[(rb.scenario_id(scfg), seed)]
        for col in ('Profit', 'FillRate', 'AvgInv'):
            assert kpis[col] == pytest.approx(ref[col], rel=1e-9, abs=1e-6)


@pytest.mark.parametrize('network', ['base', 'serial'])
def test_obs_slices_match_env_state(network):
    env = _env(network)
    view = ec.AgentView(env, 'blind')
    obs, _ = env.reset(seed=0)
    for _ in range(4):
        obs, *_ = env.step(env.action_space.sample())
    s = view.obs_slices
    net = env.network
    assert s['extra'].stop == env.observation_space.shape[0]
    np.testing.assert_allclose(obs[s['demand']], env.D[env.period - 1])
    np.testing.assert_allclose(obs[s['backlog']], env.U[env.period - 1])
    np.testing.assert_allclose(obs[s['inventory']],
                               env.X[env.period, [net.node_map[n] for n in net.main_nodes]])
    assert sum(sl.stop - sl.start for sl in view.pipeline_slices.values()) == \
        s['pipeline'].stop - s['pipeline'].start


def test_blind_view_hides_demand_information():
    env = _env()
    blind = ec.AgentView(env, 'blind')
    informed = ec.AgentView(env, 'informed')

    assert not hasattr(blind, 'demand_mean')
    for e in env.network.retail_links:
        assert 'dist_param' not in blind.graph.edges[e]
        assert 'dist_param' in informed.graph.edges[e]  # original graph untouched
    assert informed.demand_mean(7) == pytest.approx(env.demand_engine.get_current_mu(7))


def test_demand_history_contains_only_past_periods():
    env = _env()
    view = ec.AgentView(env, 'blind')
    env.reset(seed=0)
    assert view.demand_history().shape[0] == 0
    for t in range(3):
        env.step(env.action_space.sample())
    hist = view.demand_history()
    assert hist.shape == (3, len(env.network.retail_links))
    np.testing.assert_allclose(hist, env.D[:3])


def test_wrong_action_shape_is_rejected():
    class BadAgent:
        def __init__(self, view, seed):
            pass

        def get_action(self, obs, t):
            return np.zeros(1)

    with pytest.raises(ValueError, match='shape'):
        ec.run_episode(lambda env, s, sd: BadAgent(None, sd), _scenario('A_Core'), 42)


def test_cli_end_to_end(tmp_path):
    ec.main(['--policy', EXAMPLE_POLICY, '--tier', 'blind', '--blocks', 'B_PaperReplication',
             '--seeds', '2', '--baselines', 'Newsvendor', 'Oracle', '--out-dir', str(tmp_path)])

    episodes = pd.read_csv(tmp_path / 'episodes.csv')
    comparison = pd.read_csv(tmp_path / 'comparison.csv')
    summary = pd.read_csv(tmp_path / 'summary.csv')
    info = json.loads((tmp_path / 'run_info.json').read_text())

    assert len(episodes) == 4 * 2
    assert len(comparison) == 4 * 2
    assert set(summary['Agent']) == {'MovingAverageBaseStock', 'Newsvendor', 'Oracle'}
    assert comparison['Pairs'].eq(2).all()
    assert info['tier'] == 'blind' and info['seeds'] == rb.CANONICAL_SEEDS[:2]
    assert info['policy_file_sha256']
