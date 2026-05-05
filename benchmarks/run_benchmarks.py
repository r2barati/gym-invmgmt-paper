#!/usr/bin/env python3
"""
run_benchmarks.py — Unified Benchmark CLI

Run one agent at a time across all scenarios and seeds, with caching and resumability.

Usage:
  python run_benchmarks.py --list
  python run_benchmarks.py --agent Oracle --seeds 10
  python run_benchmarks.py --agent PPO-MLP --seeds 10 --train --timesteps 100000
  python run_benchmarks.py --agent SAC --seeds 10 --train --timesteps 200000
  python run_benchmarks.py --agent DAgger-B --seeds 10
  python run_benchmarks.py --agent DAgger-G --seeds 10
  python run_benchmarks.py --agent ALL --seeds 10
  python run_benchmarks.py --merge-only
  python run_benchmarks.py --agent Oracle --seeds 5 --blocks A_Core B_PaperReplication
  python run_benchmarks.py --agent LLM --seeds 3 --include-llm
"""
import os
import sys
import time
import argparse
import itertools
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

for p in [PROJECT_ROOT,
          os.path.join(PROJECT_ROOT, 'agents')]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from gym_invmgmt import CoreEnv

_CHECKPOINT_COMPAT_READY = False


def _ensure_checkpoint_compat():
    """Install module aliases needed by checkpoint compatibility paths."""
    global _CHECKPOINT_COMPAT_READY
    if _CHECKPOINT_COMPAT_READY:
        return
    try:
        import agents.checkpoint_compat  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != 'agents':
            raise
        import checkpoint_compat  # noqa: F401
    _CHECKPOINT_COMPAT_READY = True


def _model_fingerprint(agent):
    """Return a short hash identifying the loaded model weights."""
    import hashlib
    model = getattr(agent, 'model', None)
    if model is None:
        return 'no-model'
    try:
        params = model.policy.state_dict()
        h = hashlib.md5()
        for k in sorted(params.keys()):
            h.update(params[k].cpu().numpy().tobytes())
        return h.hexdigest()[:12]
    except Exception:
        return 'hash-fail'

# ---------------------------------------------------------------------------
# Canonical seeds & directories
# ---------------------------------------------------------------------------
CANONICAL_SEEDS = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]
NUM_PERIODS = 30

RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
CACHE_DIR = os.path.join(RESULTS_DIR, 'cache_v2')
MODELS_DIR = os.path.join(RESULTS_DIR, 'rl_models')
MODELS = os.path.join(PROJECT_ROOT, 'data', 'models')  # Generalist model checkpoints
LEGACY_MODELS = os.path.join(PROJECT_ROOT, 'data', 'models')   # Old per-scenario models

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# ---------------------------------------------------------------------------
# Scenario grid builders (mirrors run_matrix_eval.py)
# ---------------------------------------------------------------------------
TOPOLOGIES = ['base', 'serial']
DEMAND_COMBOS = {
    'trend+seasonal':       {'type': 'combined_chaos', 'effects': ['trend', 'seasonal']},
    'trend+seasonal+shock': {'type': 'combined_chaos', 'effects': ['trend', 'seasonal', 'shock']},
}
GOODWILL_FLAGS = [False, True]
FULFILLMENT = [True, False]


def _build_core_grid():
    scenarios = []
    for topo, (dlabel, dbase), gw, bl in itertools.product(
            TOPOLOGIES, DEMAND_COMBOS.items(), GOODWILL_FLAGS, FULFILLMENT):
        dcfg = dict(dbase, base_mu=20, use_goodwill=gw)
        scenarios.append(dict(block='A_Core', network=topo, demand_label=dlabel,
                              demand_config=dcfg, use_goodwill=gw, backlog=bl, marl=False))
    return scenarios


def _build_paper_replication():
    scenarios = []
    for topo, bl in itertools.product(TOPOLOGIES, FULFILLMENT):
        dcfg = {'type': 'stationary', 'base_mu': 20, 'use_goodwill': False}
        scenarios.append(dict(block='B_PaperReplication', network=topo, demand_label='stationary',
                              demand_config=dcfg, use_goodwill=False, backlog=bl, marl=False))
    return scenarios


def _build_marl():
    scenarios = []
    for topo, dlabel in itertools.product(TOPOLOGIES, DEMAND_COMBOS.keys()):
        dcfg = dict(DEMAND_COMBOS[dlabel], base_mu=20, use_goodwill=False)
        scenarios.append(dict(block='C_MARL', network=topo, demand_label=dlabel,
                              demand_config=dcfg, use_goodwill=False, backlog=True, marl=True))
    return scenarios


def _build_m5():
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'scripts', 'eval'))
    from m5_demand_utils import get_m5_demand_windows

    windows = get_m5_demand_windows(num_periods=NUM_PERIODS, base_mu=20.0, write_metadata=False)
    vol = windows.get('volatile', list(windows.values())[0])

    scenarios = []
    for topo in TOPOLOGIES:
        dcfg = {'type': 'stationary', 'base_mu': float(np.mean(vol)),
                'use_goodwill': False, 'external_series': vol,
                'noise_scale': 0.0}
        scenarios.append(dict(block='D_WalmartM5', network=topo, demand_label='M5_volatile',
                              demand_config=dcfg, use_goodwill=False, backlog=True, marl=False))
    return scenarios


def build_all_scenarios():
    # Note: the topology-transfer ladder in Appendix M is a separate stress
    # test, not part of this canonical 26-scenario benchmark builder.
    return _build_core_grid() + _build_paper_replication() + _build_marl() + _build_m5()


def scenario_id(scfg):
    return f"{scfg['block']}|{scfg['network']}|{scfg['demand_label']}|GW:{scfg['use_goodwill']}|BL:{scfg['backlog']}|MARL:{scfg['marl']}"


# ---------------------------------------------------------------------------
# KPI extraction
# ---------------------------------------------------------------------------
def _extract_kpis(env):
    """Extract KPIs from a completed episode.

    Fill rate uses the retail sales matrix S (matching paper Eq. 7),
    NOT 1 - U/D which silently returns 1.0 in lost-sales mode.
    Avg inventory scoped to main nodes only (excludes raw materials).
    Cost decomposition mirrors CoreEnv._STEP() components (SR, PC, HC, UP, OC, FK),
    including discounting by alpha**t.
    """
    T = env.period
    total_profit = float(np.sum(env.P[:T, :]))
    total_d = float(np.sum(env.D[:T, :]))

    # Fill rate: actual retail sales from S matrix (paper: FR = Σ S_t / Σ D_t)
    retail_sales = 0.0
    for (j, k) in env.network.retail_links:
        net_idx = env.network.network_map[(j, k)]
        retail_sales += float(np.sum(env.S[:T, net_idx]))
    fill_rate = min(1.0, retail_sales / total_d) if total_d > 0 else 1.0

    # Unfulfilled: cumulative across all periods (paper: U_total = Σ_t U_t)
    total_u = float(np.sum(env.U[:T, :]))

    # Avg inventory: main nodes only, excludes raw materials with infinite supply
    main_idx = [env.network.node_map[n] for n in env.network.main_nodes]
    avg_inv = float(np.mean(env.X[1:T+1, main_idx])) if T > 0 else 0.0

    # --- Cost decomposition (mirrors CoreEnv._STEP L369-411) ---
    total_revenue = 0.0
    total_procurement = 0.0
    total_holding = 0.0
    total_backlog_penalty = 0.0
    total_operating = 0.0
    total_fixed_ordering = 0.0

    net = env.network
    for t_step in range(T):
        discount = env.alpha ** t_step
        for j in net.main_nodes:
            j_idx = net.node_map[j]
            # Revenue (SR)
            for succ_node, net_idx in net.succ_network_indices[j]:
                succ_id = net.idx_to_node[succ_node]
                p = net.graph.edges[(j, succ_id)]['p']
                total_revenue += discount * p * env.S[t_step, net_idx]
            # Procurement (PC)
            for pred_node, reorder_idx, L in net.pred_reorder_indices[j]:
                pred_id = net.idx_to_node[pred_node]
                p = net.graph.edges[(pred_id, j)]['p']
                total_procurement += discount * p * env.R[t_step, reorder_idx]
            # Holding (HC) — on-hand + pipeline
            if j not in net.rawmat:
                h = net.graph.nodes[j]['h']
                total_holding += discount * h * env.X[t_step + 1, j_idx]
                for pred_node, reorder_idx, L in net.pred_reorder_indices[j]:
                    pred_id = net.idx_to_node[pred_node]
                    g = net.graph.edges[(pred_id, j)]['g']
                    total_holding += discount * g * env.Y[t_step + 1, reorder_idx]
            # Operating (OC) — factory only
            if j in net.factory:
                node_props = net.graph.nodes[j]
                sales_vol = sum(env.S[t_step, ni] for _, ni in net.succ_network_indices[j])
                total_operating += discount * (node_props['o'] / node_props['v']) * sales_vol
            # Backlog penalty (UP) — retail only
            if j in net.retail:
                for k in net.graph.successors(j):
                    if (j, k) in net.retail_map:
                        r_idx = net.retail_map[(j, k)]
                        b = net.graph.edges[(j, k)]['b']
                        total_backlog_penalty += discount * b * env.U[t_step, r_idx]
            # Fixed ordering cost (FK)
            for pred_node, reorder_idx, L in net.pred_reorder_indices[j]:
                pred_id = net.idx_to_node[pred_node]
                k_cost = net.graph.edges[(pred_id, j)].get('K', 0.0)
                if k_cost > 0.0 and env.R[t_step, reorder_idx] > 0.0:
                    total_fixed_ordering += discount * k_cost

    decomp_profit = (
        total_revenue
        - total_procurement
        - total_operating
        - total_holding
        - total_backlog_penalty
        - total_fixed_ordering
    )
    if not np.isclose(decomp_profit, total_profit, rtol=1e-8, atol=1e-6):
        raise AssertionError(
            "KPI decomposition mismatch: "
            f"reconstructed={decomp_profit:.6f}, env_profit={total_profit:.6f}"
        )

    # Bullwhip ratio: Var(orders) / Var(demand)
    if T > 1:
        demand_per_period = np.sum(env.D[:T, :], axis=1)
        demand_var = float(np.var(demand_per_period))
        if hasattr(env, 'action_log') and env.action_log is not None:
            order_per_period = np.sum(env.action_log[:T, :], axis=1)
            order_var = float(np.var(order_per_period))
        else:
            order_var = float(np.var(np.sum(env.R[:T, :], axis=1)))
        bullwhip = order_var / demand_var if demand_var > 1e-8 else 1.0
    else:
        bullwhip = 1.0

    # Service Level: fraction of periods with zero stockout at retail
    periods_no_stockout = 0
    for t_step in range(T):
        period_fulfilled = True
        for (j, k) in env.network.retail_links:
            r_idx = env.network.retail_map.get((j, k))
            if r_idx is not None and env.U[t_step, r_idx] > 0:
                period_fulfilled = False
                break
        if period_fulfilled:
            periods_no_stockout += 1
    service_level = periods_no_stockout / T if T > 0 else 1.0

    return {
        'Profit': total_profit,
        'FillRate': fill_rate,
        'ServiceLevel': service_level,
        'AvgInv': avg_inv,
        'Unfulfilled': total_u,
        'Revenue': float(total_revenue),
        'HoldingCost': float(total_holding),
        'BacklogPenalty': float(total_backlog_penalty),
        'ProcurementCost': float(total_procurement),
        'OperatingCost': float(total_operating),
        'FixedOrderingCost': float(total_fixed_ordering),
        'BullwhipRatio': bullwhip,
    }



# ---------------------------------------------------------------------------
# Cache manager
# ---------------------------------------------------------------------------
def _cache_path(agent_id):
    safe = agent_id.replace(' ', '_').replace('(', '').replace(')', '')
    return os.path.join(CACHE_DIR, f'bench_{safe}.csv')


def _agent_prefix(agent_id):
    name_overrides = {
        'PPO-GNN': 'GNN_V3',
        'PPO-GNN-B': 'GNN_V3_B',
        'ExpSmoothing-I': 'ExpSmooth_I',
    }
    if agent_id in name_overrides:
        return name_overrides[agent_id]
    return agent_id.replace('-', '_').replace('(', '').replace(')', '').replace(',', '')


def _load_completed(agent_id):
    path = _cache_path(agent_id)
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path)
        # Smart Cache Invalidation: Drop failed or NaN rows so they are re-attempted
        if 'Failed' in df.columns:
            df = df[df['Failed'] != True]
        if 'Profit' in df.columns:
            df = df.dropna(subset=['Profit'])
        return set(zip(df['ScenarioKey'], df['Seed']))
    except Exception:
        return set()


def _append_result(agent_id, row):
    path = _cache_path(agent_id)
    new_df = pd.DataFrame([row])
    if not os.path.exists(path):
        new_df.to_csv(path, index=False)
        return

    old_df = pd.read_csv(path)
    columns = list(old_df.columns)
    columns.extend(c for c in new_df.columns if c not in columns)
    old_df = old_df.reindex(columns=columns)
    new_df = new_df.reindex(columns=columns)
    pd.concat([old_df, new_df], ignore_index=True).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Agent factory — returns (agent, needs_feature_wrapper) for a given scenario
# ---------------------------------------------------------------------------
def _env_kwargs(scfg):
    return dict(scenario='network' if scfg['network'] == 'base' else 'serial',
                demand_config=scfg['demand_config'], num_periods=NUM_PERIODS,
                backlog=scfg['backlog'])


def _make_agent(agent_id, env, scfg, seed):
    """Instantiate an agent for the given env and scenario config."""
    if agent_id in {
        'PPO-MLP', 'PPO-GNN', 'Residual', 'SAC',
        'PPO-MLP-B', 'PPO-GNN-B', 'Residual-B', 'SAC-B',
        'DAgger-G', 'DAgger-B', 'GNN-IL',
        'PPO-MLP-v1', 'PPO-MLP-raw',
    }:
        _ensure_checkpoint_compat()

    if agent_id == 'Oracle':
        from oracle import OracleAgent
        return OracleAgent(env, seed=seed)

    elif agent_id == 'MSSP':
        from mssp_agent import RollingHorizonMSSPAgent
        import networkx as nx
        G = env.graph
        cp = 0
        for src in list(env.network.factory) + list(env.network.rawmat):
            for dst in env.network.retail:
                try:
                    for path in nx.all_simple_paths(G, src, dst):
                        lt = sum(int(G.edges[path[i], path[i+1]].get('L', 0)) for i in range(len(path)-1))
                        cp = max(cp, lt)
                except nx.NetworkXError:
                    pass
        saa_seed = int(np.random.SeedSequence([seed, 0x4D535350]).generate_state(1)[0])
        return RollingHorizonMSSPAgent(env, planning_horizon=max(10, cp + 5), is_blind=True, seed=saa_seed)

    elif agent_id == 'MSSP-I':
        from mssp_agent import RollingHorizonMSSPAgent
        import networkx as nx
        G = env.graph
        cp = 0
        for src in list(env.network.factory) + list(env.network.rawmat):
            for dst in env.network.retail:
                try:
                    for path in nx.all_simple_paths(G, src, dst):
                        lt = sum(int(G.edges[path[i], path[i+1]].get('L', 0)) for i in range(len(path)-1))
                        cp = max(cp, lt)
                except nx.NetworkXError:
                    pass
        saa_seed = int(np.random.SeedSequence([seed, 0x4D535350]).generate_state(1)[0])
        return RollingHorizonMSSPAgent(env, planning_horizon=max(10, cp + 5), is_blind=False, seed=saa_seed)

    elif agent_id == 'DLP':
        from dlp_agent import DLPAgent
        return DLPAgent(env, is_blind=True)

    elif agent_id == 'DLP-I':
        from dlp_agent import DLPAgent
        return DLPAgent(env, is_blind=False)

    elif agent_id == 'Newsvendor':
        from newsvendor_heuristic_agent import NewsvendorHeuristicAgent
        return NewsvendorHeuristicAgent(env, is_blind=True)

    elif agent_id == 'Newsvendor-I':
        from newsvendor_heuristic_agent import NewsvendorHeuristicAgent
        return NewsvendorHeuristicAgent(env, is_blind=False)

    elif agent_id == '(s,S)':
        from ss_policy_heuristic_agent import SSPolicyHeuristicAgent
        return SSPolicyHeuristicAgent(env, is_blind=True)

    elif agent_id == '(s,S)-I':
        from ss_policy_heuristic_agent import SSPolicyHeuristicAgent
        return SSPolicyHeuristicAgent(env, is_blind=False)

    elif agent_id == 'ExpSmoothing':
        from exp_smoothing_heuristic_agent import ExpSmoothingHeuristicAgent
        return ExpSmoothingHeuristicAgent(env, is_blind=True)

    elif agent_id == 'ExpSmoothing-I':
        from exp_smoothing_heuristic_agent import ExpSmoothingHeuristicAgent
        return ExpSmoothingHeuristicAgent(env, is_blind=False)

    elif agent_id == 'EchelonApprox':
        from echelon_basestock_agent import EchelonApproxAgent
        return EchelonApproxAgent(env, is_blind=True)

    elif agent_id == 'EchelonApprox-I':
        from echelon_basestock_agent import EchelonApproxAgent
        return EchelonApproxAgent(env, is_blind=False)

    elif agent_id == 'PPO-MLP':
        from ppo_baseline_agent import RLAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'ppo-mlp_{topo_label}.zip')
        sp = os.path.join(MODELS, f'ppo-mlp_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return RLAgent(env, model_path=mp, stats_path=sp, obs_level='v2', verbose=False)

    elif agent_id == 'PPO-MLP-B':
        from ppo_baseline_agent import RLAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'ppo-mlp_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'ppo-mlp_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return RLAgent(env, model_path=mp, stats_path=sp, obs_level='v2', is_blind=True, verbose=False)

    elif agent_id == 'PPO-MLP-v1':
        from ppo_baseline_agent import RLAgent
        # v1 = DomainFeatureWrapper(enhanced=False) — basic 3 features/node
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'ppo-mlp_{topo_label}_obsv1.zip')
        sp = os.path.join(MODELS, f'ppo-mlp_{topo_label}_obsv1_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return RLAgent(env, model_path=mp, stats_path=sp, obs_level='v1', verbose=False)

    elif agent_id == 'PPO-MLP-raw':
        from ppo_baseline_agent import RLAgent
        # raw = no feature wrapper, CoreEnv raw obs only (71d)
        mp = os.path.join(MODELS, 'ppo-mlp_base_obsraw.zip')
        sp = os.path.join(MODELS, 'ppo-mlp_base_obsraw_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp) or scfg['network'] != 'base':
            return None
        return RLAgent(env, model_path=mp, stats_path=sp, obs_level='raw', verbose=False)

    elif agent_id == 'PPO-GNN':
        from ppo_gnn_agent import PPOGNNAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'gnn-v3_{topo_label}.zip')
        sp = os.path.join(MODELS, f'gnn-v3_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return PPOGNNAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'PPO-GNN-B':
        from ppo_gnn_agent import PPOGNNAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'gnn-v3_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'gnn-v3_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return PPOGNNAgent(env, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'PPO-Transformer':
        from ppo_transformer_agent import PPOTransformerAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'ppo-transformer_{topo_label}.zip')
        sp = os.path.join(MODELS, f'ppo-transformer_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return PPOTransformerAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'PPO-Transformer-B':
        from ppo_transformer_agent import PPOTransformerAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'ppo-transformer_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'ppo-transformer_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return PPOTransformerAgent(env, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'ST-PPO':
        from st_ppo_agent import STPPOAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'st-ppo_{topo_label}.zip')
        sp = os.path.join(MODELS, f'st-ppo_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return STPPOAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'ST-PPO-B':
        from st_ppo_agent import STPPOAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'st-ppo_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'st-ppo_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return STPPOAgent(env, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'Residual':
        from ppo_residual_agent import ResidualRLAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'residual_{topo_label}.zip')
        sp = os.path.join(MODELS, f'residual_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return ResidualRLAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'Residual-B':
        from ppo_residual_agent import ResidualRLAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'residual_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'residual_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return ResidualRLAgent(env, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'SAC':
        from sac_eval_agent import SACAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'sac_{topo_label}.zip')
        sp = os.path.join(MODELS, f'sac_{topo_label}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return SACAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'SAC-B':
        from sac_eval_agent import SACAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        mp = os.path.join(MODELS, f'sac_{topo_label}_blind.zip')
        sp = os.path.join(MODELS, f'sac_{topo_label}_blind_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return SACAgent(env, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'GNN-IL':
        from gnn_il_agent import GNNILAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        # GNN-IL = DAgger generalist (GNN arch, Oracle-imitation trained)
        if topo_label == 'base':
            mp = os.path.join(MODELS, 'ppo_gnn_il_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_il.pkl')
        else:
            mp = os.path.join(MODELS, 'ppo_gnn_il_serial_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_il_serial.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return GNNILAgent(env, model_path=mp, stats_path=sp)

    elif agent_id == 'DAgger-B':
        # DAgger-B = DAgger Blind (information-restricted specialist)
        from dagger_agent import DAggerAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        if topo_label == 'base':
            mp = os.path.join(MODELS, 'ppo_gnn_dagger_base_blind_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_dagger_base_blind.pkl')
        else:
            mp = os.path.join(MODELS, 'ppo_gnn_dagger_serial_blind_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_dagger_serial_blind.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return DAggerAgent(env, specialist=False, model_path=mp, stats_path=sp, is_blind=True)

    elif agent_id == 'DAgger-G':
        from dagger_agent import DAggerAgent
        topo_label = 'base' if scfg['network'] == 'base' else 'serial'
        # Topology-aware model selection
        if topo_label == 'base':
            mp = os.path.join(MODELS, 'ppo_gnn_dagger_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_dagger.pkl')
        else:
            mp = os.path.join(MODELS, 'ppo_gnn_dagger_serial_best.zip')
            sp = os.path.join(MODELS, 'vec_normalize_gnn_dagger_serial.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
        return DAggerAgent(env, specialist=False, model_path=mp, stats_path=sp)

    elif agent_id in ('LLM', 'LLM-ZS', 'LLM-ZS-Direct'):
        from llm_agent import LLMZeroShotAgent
        return LLMZeroShotAgent(env, backend='local', verbose=False)

    elif agent_id == 'LLM-Policy-C':
        from llm_policy_agent import LLMPolicyAgent
        return LLMPolicyAgent(env, backend='local', verbose=False)

    elif agent_id == 'LLM-InvAgent-C':
        from llm_invagent_central_agent import CentralInvAgent
        return CentralInvAgent(env, backend='local', verbose=False)

    elif agent_id.startswith('PPO-Specialist-'):
        # Specialist PPO-MLP: trained on a single scenario, obs_level=v2
        from ppo_baseline_agent import RLAgent
        scenario_key = agent_id.replace('PPO-Specialist-', '')
        # Only evaluate on base topology
        if scfg['network'] != 'base':
            return None
        mp = os.path.join(MODELS, f'ppo-mlp_base_specialist_{scenario_key}.zip')
        sp = os.path.join(MODELS, f'ppo-mlp_base_specialist_{scenario_key}_vecnorm.pkl')
        if not os.path.exists(mp) or not os.path.exists(sp):
            return None
            
        # Infer obs_level from VecNormalize dimension for v1 specialist checkpoints
        import pickle
        with open(sp, 'rb') as f:
            vec_env = pickle.load(f)
        obs_dim = vec_env.obs_rms.mean.shape[0] if hasattr(vec_env, 'obs_rms') else 129
        obs_level = 'v1' if obs_dim == 91 else 'v2'
        
        return RLAgent(env, model_path=mp, stats_path=sp, obs_level=obs_level, verbose=False)

    return None


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------
def _run_episode(agent_id, env, agent, scfg, seed, plan_time=0.0):
    """Run one full episode and return KPIs + latency."""
    from mssp_agent import RollingHorizonMSSPAgent

    obs, _ = env.reset(seed=seed)
    done = False
    t = 0
    t0 = time.perf_counter()

    total_solve_time = 0.0
    failed_solves = 0
    last_status = 1

    while not done:
        if isinstance(agent, RollingHorizonMSSPAgent):
            action = agent.get_action(t)
        else:
            action = agent.get_action(obs, t)

        if hasattr(agent, 'last_solve_time'):
            total_solve_time += agent.last_solve_time
            if agent.last_solve_status != 1:
                failed_solves += 1
                last_status = agent.last_solve_status

        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        t += 1

    # Include offline planning time (agent init) in total latency
    replay_time = time.perf_counter() - t0
    latency = plan_time + replay_time
    
    kpis = _extract_kpis(env)
    kpis['PlanTime'] = plan_time
    kpis['ReplayTime'] = replay_time
    kpis['Latency'] = latency
    
    # OR Solver Diagnostics
    if hasattr(agent, 'last_solve_time'):
        kpis['SolveTime'] = total_solve_time
        kpis['SolveStatus'] = last_status
        kpis['FailedSolves'] = failed_solves

    # LLM Reliability Diagnostics
    if hasattr(agent, 'parse_failure_rate'):
        kpis['ParseFailureRate'] = agent.parse_failure_rate
        kpis['TotalQueries'] = getattr(agent, 'total_queries', 0)
        kpis['ParseFailures'] = getattr(agent, 'parse_failures', 0)
        
    return kpis


# ---------------------------------------------------------------------------
# Evaluate one agent across all scenarios × seeds
# ---------------------------------------------------------------------------
# 1 Theoretical Upper Bound
ORACLE_ID = ['Oracle']

# 22 Primary Agent Configurations
PRIMARY_AGENT_IDS = [
    'MSSP', 'MSSP-I', 'DLP', 'DLP-I',
    # Heuristics (blind — no demand parameter access)
    'Newsvendor', '(s,S)', 'ExpSmoothing',
    # Heuristics (informed — have demand parameter access)
    'Newsvendor-I', '(s,S)-I', 'ExpSmoothing-I',
    # Echelon base-stock (Clark-Scarf-inspired)
    'EchelonApprox', 'EchelonApprox-I',
    # RL informed (have demand parameter access via DomainFeatureWrapper)
    'PPO-MLP', 'PPO-GNN', 'PPO-Transformer', 'ST-PPO', 'Residual', 'SAC',
    # RL blind (PPO-MLP has blind version trained)
    'PPO-MLP-B',
    # DAgger / IL
    'GNN-IL', 'DAgger-G', 'DAgger-B',
]

# 6 Ablation Configurations
ABLATION_AGENT_IDS = [
    # Blind RL variants with trained models
    'PPO-GNN-B', 'ST-PPO-B', 'Residual-B', 'SAC-B',
    # PPO-MLP observation ablation (base topology only)
    'PPO-MLP-v1', 'PPO-MLP-raw',
]

# Total: 29 IDs (1 Oracle + 22 Primary + 6 Ablations)
ALL_AGENT_IDS = ORACLE_ID + PRIMARY_AGENT_IDS + ABLATION_AGENT_IDS


def eval_one_agent(agent_id, scenarios, seeds, force=False):
    """Evaluate a single agent across all scenarios × seeds with caching."""
    if force:
        path = _cache_path(agent_id)
        if os.path.exists(path):
            scenario_keys = {scenario_id(s) for s in scenarios}
            try:
                old_df = pd.read_csv(path)
                keep_df = old_df[~old_df['ScenarioKey'].isin(scenario_keys)]
                if keep_df.empty:
                    os.remove(path)
                    print(f"  Cleared cache for {agent_id}")
                else:
                    keep_df.to_csv(path, index=False)
                    print(f"  Cleared {len(old_df) - len(keep_df)} cached row(s) "
                          f"for {agent_id} in selected scenario block(s)")
            except Exception:
                os.remove(path)
                print(f"  Cleared cache for {agent_id}")

    completed = _load_completed(agent_id)
    total = len(scenarios) * len(seeds)
    done_count = 0
    skip_count = 0
    fail_count = 0

    print(f"\n{'='*70}")
    print(f"  Evaluating: {agent_id}")
    print(f"  Scenarios: {len(scenarios)} | Seeds: {len(seeds)} | Total runs: {total}")
    print(f"  Cache: {len(completed)} already completed")
    print(f"{'='*70}")

    for si, scfg in enumerate(scenarios):
        sid = scenario_id(scfg)
        topo = scfg['network'].upper()
        bl = 'BL' if scfg['backlog'] else 'LS'
        marl = ' [MARL]' if scfg['marl'] else ''

        for seed in seeds:
            key = (sid, seed)
            if key in completed:
                skip_count += 1
                continue

            # Create fresh env for each run
            ekw = _env_kwargs(scfg)
            try:
                env = CoreEnv(**ekw)
            except Exception as e:
                print(f"  ERROR Env creation failed: {e}")
                fail_count += 1
                continue

            # Create agent and measure offline planning time
            try:
                t_plan_start = time.perf_counter()
                agent = _make_agent(agent_id, env, scfg, seed)
                plan_time = time.perf_counter() - t_plan_start
            except Exception as e:
                print(f"  ERROR Agent init failed for {sid} seed={seed}: {e}")
                fail_count += 1
                continue

            if agent is None:
                # Model file not found — log warning and skip
                print(f"  WARNING Model for {agent_id} not found on {sid}. Skipping...")
                fail_count += 1
                continue

            # Store model fingerprint in every cache row for checkpoint provenance
            # so future merges can prove which artifact produced each result.
            fp = _model_fingerprint(agent)
            if done_count == 0:
                print(f"  Model fingerprint: {fp}")

            # Run episode
            try:
                kpis = _run_episode(agent_id, env, agent, scfg, seed, plan_time=plan_time)
            except Exception as e:
                import traceback
                print(f"  ERROR Episode FAILED for {sid} seed={seed}: {e}")
                traceback.print_exc()
                kpis = dict(Profit=np.nan, FillRate=np.nan, AvgInv=np.nan,
                            Unfulfilled=np.nan, Latency=np.nan, PlanTime=np.nan, ReplayTime=np.nan,
                            Failed=True, FailReason=str(e))
                fail_count += 1

            model_path = getattr(agent, 'model_path', '')
            stats_path = getattr(agent, 'stats_path', '')
            row = dict(ScenarioKey=sid, Block=scfg['block'], Network=scfg['network'],
                       Demand=scfg['demand_label'], Goodwill=scfg['use_goodwill'],
                       Backlog=scfg['backlog'], MARL=scfg['marl'], Seed=seed,
                       ModelFingerprint=fp, ModelPath=model_path, StatsPath=stats_path,
                       **kpis)
            _append_result(agent_id, row)
            done_count += 1

            # Progress
            progress = done_count + skip_count + fail_count
            if done_count % 5 == 1 or done_count == 1:
                print(f"  [{progress}/{total}] {scfg['block'][:6]:6s} {topo:6s} "
                      f"{scfg['demand_label']:20s} {bl} seed={seed:5d} "
                      f"→ Profit={kpis['Profit']:>8.1f} Fill={kpis['FillRate']:.3f} "
                      f"({kpis['Latency']:.2f}s){marl}")

    print(f"\n  OK {agent_id} complete: {done_count} new + {skip_count} cached + {fail_count} failed")
    if fail_count > 0:
        print(f"  WARNING: {fail_count} episode(s) FAILED for {agent_id}. "
              f"Check output CSV for Failed=True rows. Results may be unreliable.")


# ---------------------------------------------------------------------------
# Merge all caches into final CSV
# ---------------------------------------------------------------------------

# KPI suffixes to emit for each agent
_KPI_SUFFIXES = ['Profit', 'Profit_Std', 'FillRate', 'AvgInv', 'Unfulfilled']


def merge_all(include_llm=False):
    """Merge per-agent caches into a pivoted paper-ready CSV.

    Output format matches the explicit agent taxonomy from the codebase:
      - No ``_Mean`` suffix on KPI columns
      - Precise parity matching via strict seed intersections
    """
    print(f"\n{'='*70}")
    print(f"  Merging all agent caches")
    print(f"{'='*70}")

    all_dfs = {}
    agent_list = ALL_AGENT_IDS + (['LLM-ZS-Direct', 'LLM-InvAgent-C', 'LLM-Policy-C'] if include_llm else [])
    for agent_id in agent_list:
        path = _cache_path(agent_id)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                all_dfs[agent_id] = df
                print(f"  OK {agent_id}: {len(df)} rows")
            except Exception as e:
                print(f"  ERROR {agent_id}: {e}")

    if not all_dfs:
        print("  No caches found. Run some evaluations first.")
        return

    # Build merged: group by scenario, aggregate across seeds
    merge_keys = ['ScenarioKey', 'Block', 'Network', 'Demand', 'Goodwill', 'Backlog', 'MARL']
    merged_rows = []

    # Get all unique scenarios from any agent
    all_scenarios = set()
    for df in all_dfs.values():
        for _, r in df.iterrows():
            all_scenarios.add(tuple(r[k] for k in merge_keys))

    # --- Strict Parity Intersection ---
    # To guarantee fair comparisons, we only evaluate on seeds that succeeded for ALL agents
    valid_seeds_per_scenario = {}
    for scen_tuple in all_scenarios:
        all_tested_seeds = set()
        for df in all_dfs.values():
            mask = True
            for k, v in zip(merge_keys, scen_tuple):
                mask = mask & (df[k] == v)
            all_tested_seeds.update(df[mask]['Seed'].unique())
        
        valid_seeds = set(all_tested_seeds)
        for agent_id, df in all_dfs.items():
            mask = True
            for k, v in zip(merge_keys, scen_tuple):
                mask = mask & (df[k] == v)
            agent_df = df[mask]
            
            # If the agent didn't run this scenario at all (e.g., base-only ablations),
            # we do not enforce its empty seed set against the valid seeds.
            if agent_df.empty:
                continue
                
            # Drop failed or NaN seeds
            if 'Failed' in agent_df.columns:
                failed_seeds = agent_df[agent_df['Failed'] == True]['Seed'].values
                valid_seeds.difference_update(failed_seeds)
            if 'Profit' in agent_df.columns:
                nan_seeds = agent_df[agent_df['Profit'].isna()]['Seed'].values
                valid_seeds.difference_update(nan_seeds)
            
            # Drop seeds the agent didn't even run
            ran_seeds = set(agent_df['Seed'].values)
            valid_seeds.intersection_update(ran_seeds)
            
        valid_seeds_per_scenario[scen_tuple] = valid_seeds
        
        dropped = len(all_tested_seeds) - len(valid_seeds)
        if dropped > 0:
            print(f"  WARNING [Parity] Dropped {dropped} seeds for {scen_tuple[0]} (Valid N={len(valid_seeds)})")
    # ----------------------------------

    for scen_tuple in sorted(all_scenarios):
        row = dict(zip(merge_keys, scen_tuple))

        for agent_id, df in all_dfs.items():
            mask = True
            for k, v in zip(merge_keys, scen_tuple):
                mask = mask & (df[k] == v)
            # Filter agent data to only include valid_seeds
            mask = mask & (df['Seed'].isin(valid_seeds_per_scenario[scen_tuple]))
            agent_df = df[mask]

            prefix = _agent_prefix(agent_id)
            if len(agent_df) > 0:
                profits = agent_df['Profit'].values
                # Primary columns (no _Mean suffix; matches the published merged CSV schema)
                row[f'{prefix}_Profit'] = profits.mean()
                row[f'{prefix}_Profit_Std'] = profits.std()
                row[f'{prefix}_FillRate'] = agent_df['FillRate'].mean()
                row[f'{prefix}_AvgInv'] = agent_df['AvgInv'].mean()
                row[f'{prefix}_Unfulfilled'] = agent_df['Unfulfilled'].mean()

                # CVaR (5%): mean profit of the worst 5% of outcomes
                n_tail = max(1, int(0.05 * len(profits)))
                part_idx = min(n_tail, len(profits) - 1)
                row[f'{prefix}_CVaR5'] = float(np.mean(np.partition(profits, part_idx)[:n_tail]))

                # Cost decomposition
                for cost_col in ['Revenue', 'HoldingCost', 'BacklogPenalty',
                                 'ProcurementCost', 'OperatingCost', 'FixedOrderingCost',
                                 'BullwhipRatio', 'ServiceLevel']:
                    if cost_col in agent_df.columns:
                        row[f'{prefix}_{cost_col}'] = agent_df[cost_col].mean()

                # Time columns
                latency = agent_df['Latency'].mean()
                row[f'Time_{prefix}'] = latency
                row[f'Time_Sec_{prefix}'] = latency
                
                for tcol in ['PlanTime', 'ReplayTime', 'SolveTime']:
                    if tcol in agent_df.columns:
                        row[f'{prefix}_{tcol}'] = agent_df[tcol].mean()

                for acol in ['ModelFingerprint', 'ModelPath', 'StatsPath']:
                    if acol in agent_df.columns:
                        vals = [v for v in agent_df[acol].dropna().astype(str).unique() if v]
                        if vals:
                            row[f'{prefix}_{acol}'] = ';'.join(sorted(vals))

        merged_rows.append(row)

    df_final = pd.DataFrame(merged_rows)

    # Compute derived metrics if Oracle exists
    if 'Oracle_Profit' in df_final.columns:
        for agent_id in all_dfs:
            prefix = _agent_prefix(agent_id)
            col = f'{prefix}_Profit'
            if col in df_final.columns:
                df_final[f'{prefix}_OptGap_Pct'] = (
                    (df_final['Oracle_Profit'] - df_final[col])
                    / df_final['Oracle_Profit'].abs() * 100
                ).round(2)

        # Empirical OR diagnostics. These are benchmark gaps and VSS analogues
        # for the shipped rolling-horizon approximations, not theoretical
        # EVPI/VPI quantities.
        if 'MSSP_Profit' in df_final.columns:
            df_final['OracleGap_MSSP'] = df_final['Oracle_Profit'] - df_final['MSSP_Profit']
        if 'MSSP_I_Profit' in df_final.columns:
            df_final['OracleGap_MSSP_I'] = (
                df_final['Oracle_Profit'] - df_final['MSSP_I_Profit']
            )
        if 'MSSP_Profit' in df_final.columns and 'DLP_Profit' in df_final.columns:
            df_final['VSS_Blind'] = df_final['MSSP_Profit'] - df_final['DLP_Profit']
        if 'MSSP_I_Profit' in df_final.columns and 'DLP_I_Profit' in df_final.columns:
            df_final['VSS_Informed'] = (
                df_final['MSSP_I_Profit'] - df_final['DLP_I_Profit']
            )

    # Save to primary location
    out_path = os.path.join(RESULTS_DIR, 'benchmark_final_merged.csv')
    df_final.to_csv(out_path, index=False)
    print(f"\n  Merged CSV: {out_path} ({len(df_final)} scenarios × {len(all_dfs)} agents)")

    # --- Duplicate-output scanner: detect identical profit columns ---
    profit_cols = [c for c in df_final.columns if c.endswith('_Profit') and '_Profit_Std' not in c]
    duplicate_output_found = False
    for i, col_a in enumerate(profit_cols):
        for col_b in profit_cols[i+1:]:
            vals_a = df_final[col_a].dropna()
            vals_b = df_final[col_b].dropna()
            common = vals_a.index.intersection(vals_b.index)
            if len(common) < 5:
                continue
            matches = (vals_a[common] - vals_b[common]).abs() < 0.01
            pct_identical = matches.mean() * 100
            if pct_identical > 80:
                duplicate_output_found = True
                print(f"  DUPLICATE-OUTPUT WARNING: {col_a} ≈ {col_b} "
                      f"({pct_identical:.0f}% identical across {len(common)} scenarios)")
    if not duplicate_output_found:
        print(f"  Duplicate-output scan passed: no duplicate agent columns detected")

    # Print summary table
    profit_cols = [c for c in df_final.columns if c.endswith('_Profit') and not c.endswith('_Profit_Std')]
    if profit_cols:
        print(f"\n  {'Scenario':<50} ", end='')
        for c in profit_cols:
            name = c.replace('_Profit', '')[:8]
            print(f"{name:>10}", end='')
        print()
        print("  " + "-" * (50 + 10 * len(profit_cols)))
        for _, r in df_final.head(10).iterrows():
            label = f"{r['Network']:6s} {r['Demand']:20s} BL={r['Backlog']}"
            print(f"  {label:<50} ", end='')
            for c in profit_cols:
                v = r[c]
                print(f"{'NaN' if pd.isna(v) else f'{v:.0f}':>10}", end='')
            print()
        if len(df_final) > 10:
            print(f"  ... ({len(df_final) - 10} more rows)")


# ---------------------------------------------------------------------------
# List agents
# ---------------------------------------------------------------------------
def list_agents(include_llm=False):
    ids = ALL_AGENT_IDS + (['LLM-ZS-Direct', 'LLM-InvAgent-C', 'LLM-Policy-C'] if include_llm else [])
    print(f"\n{'='*70}")
    print(f"  Registered Agents ({len(ids)})")
    print(f"{'='*70}")
    print(f"  {'ID':<15} {'Paradigm':<15} {'Cache':<10} {'Rows':<8}")
    print(f"  {'-'*48}")

    paradigms = {
        'Oracle': 'Math Opt', 'MSSP': 'Math Opt', 'DLP': 'Math Opt',
        'Newsvendor': 'Heuristic', '(s,S)': 'Heuristic', 'ExpSmoothing': 'Heuristic',
        'PPO-MLP': 'RL (PPO)', 'PPO-GNN': 'RL (PPO)',
        'PPO-Transformer': 'RL (PPO)', 'ST-PPO': 'RL (PPO)',
        'Residual': 'RL (PPO)', 'SAC': 'RL (SAC)',
        'GNN-IL': 'Imitation', 'DAgger-B': 'Imitation', 'DAgger-G': 'Imitation',
        'LLM': 'Foundation', 'LLM-ZS-Direct': 'Foundation', 'LLM-InvAgent-C': 'Foundation', 'LLM-Policy-C': 'Foundation',
    }

    for aid in ids:
        path = _cache_path(aid)
        exists = os.path.exists(path)
        rows = 0
        if exists:
            try:
                rows = len(pd.read_csv(path))
            except Exception:
                pass
        status = f"{'OK' if exists else '—'}"
        print(f"  {aid:<15} {paradigms.get(aid, '?'):<15} {status:<10} {rows if rows else '':<8}")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Unified Benchmark Runner')
    parser.add_argument('--agent', type=str, default=None,
                        help='Agent ID to evaluate (or ALL). Use --list to see options.')
    parser.add_argument('--seeds', type=int, default=10,
                        help='Number of seeds (default: 10)')
    parser.add_argument('--blocks', nargs='*', default=None,
                        help='Scenario blocks to include (e.g. A_Core B_PaperReplication)')
    parser.add_argument('--force', action='store_true',
                        help='Clear cache for this agent before running')
    parser.add_argument('--list', action='store_true', help='List all registered agents')
    parser.add_argument('--merge-only', action='store_true', help='Merge caches without evaluating')
    parser.add_argument('--include-llm', action='store_true',
                        help='Include LLM agents (LLM-ZS-Direct, LLM-InvAgent-C, LLM-Policy-C; slow, requires GGUF)')
    parser.add_argument('--train', action='store_true',
                        help='Train generalist models before evaluating')
    parser.add_argument('--timesteps', type=int, default=200000,
                        help='Training timesteps for --train (default: 200000)')
    args = parser.parse_args()

    if args.list:
        list_agents(include_llm=args.include_llm)
        return

    if args.merge_only:
        merge_all(include_llm=args.include_llm)
        return

    if args.agent is None:
        parser.print_help()
        return

    # --- Training phase (if requested) ---
    ARCH_MAP = {
        'PPO-MLP': 'ppo-mlp',
        'PPO-GNN': 'ppo-gnn',
        'PPO-Transformer': 'ppo-transformer',
        'ST-PPO': 'st-ppo',
        'SAC': 'sac',
    }
    if args.train:
        if args.agent == 'ALL':
            archs_to_train = list(ARCH_MAP.values())
        elif args.agent in ARCH_MAP:
            archs_to_train = [ARCH_MAP[args.agent]]
        else:
            archs_to_train = []
            print(f"  WARNING {args.agent} does not require training.")

        if archs_to_train:
            from train_generalist import train
            for arch in archs_to_train:
                for topo, topo_label in [('network', 'base'), ('serial', 'serial')]:
                    model_file = os.path.join(MODELS, f'{arch}_{topo_label}.zip')
                    if os.path.exists(model_file) and not args.force:
                        print(f"  SKIP {arch}/{topo_label} already trained at {model_file}")
                    else:
                        print(f"\n  Training {arch}/{topo_label} ({args.timesteps:,} steps)...")
                        train(arch, total_timesteps=args.timesteps, topology=topo)

    # Build scenarios
    all_scenarios = build_all_scenarios()
    if args.blocks:
        all_scenarios = [s for s in all_scenarios if s['block'] in args.blocks]
    print(f"\n  Scenario grid: {len(all_scenarios)} scenarios")

    # Resolve seeds
    if args.seeds <= len(CANONICAL_SEEDS):
        seeds = CANONICAL_SEEDS[:args.seeds]
    else:
        seeds = CANONICAL_SEEDS + list(range(10000, 10000 + args.seeds - len(CANONICAL_SEEDS)))
    print(f"  Seeds: {seeds}")

    # Resolve agents
    if args.agent == 'ALL':
        agent_ids = list(ALL_AGENT_IDS)
        if args.include_llm:
            agent_ids.extend(['LLM-ZS-Direct', 'LLM-InvAgent-C', 'LLM-Policy-C'])
    else:
        agent_ids = [args.agent]

    # Run each agent
    for agent_id in agent_ids:
        if agent_id in ('LLM', 'LLM-ZS-Direct', 'LLM-InvAgent-C', 'LLM-Policy-C') and not args.include_llm and args.agent not in ('LLM', 'LLM-ZS-Direct', 'LLM-InvAgent-C', 'LLM-Policy-C'):
            continue
        eval_one_agent(agent_id, all_scenarios, seeds, force=args.force)

    # Auto-merge after completion
    if len(agent_ids) > 1 or args.agent == 'ALL':
        merge_all(include_llm=args.include_llm)

    print(f"\n{'#'*70}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'#'*70}")


if __name__ == '__main__':
    main()
