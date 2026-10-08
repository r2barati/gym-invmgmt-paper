"""
evaluate_custom.py — Evaluate your own policy on the canonical gym-invmgmt benchmark.

Runs ONLY your agent on the paper's scenarios and canonical seeds, then compares
it seed-by-seed against the published per-seed baseline results in
results/cache_v2/. Baselines are not re-run, so no solver, checkpoint, or GPU
is needed.

Usage:
  python benchmarks/evaluate_custom.py --policy path/to/my_agent.py:MyAgent --tier blind
  python benchmarks/evaluate_custom.py --policy my_pkg.agents:MyAgent --tier informed --blocks A_Core
  python benchmarks/evaluate_custom.py --policy benchmarks/example_custom_agent.py:MovingAverageBaseStock --tier blind

Agent protocol (same call convention as the shipped baselines):

  class MyAgent:
      def __init__(self, view, seed): ...
      def get_action(self, obs, t): ...   # array shaped like view.action_space

`obs` is the raw CoreEnv observation; `view.obs_slices` and
`view.pipeline_slices` describe its layout. `view` exposes only what the
declared --tier allows:

  blind     Observation/action spaces, horizon, backlog flag, discount, the
            static network (topology, costs, lead times, capacities — demand
            distribution parameters removed), and realized demand for past
            periods. Matches the blind baselines (MSSP, DLP, Newsvendor, (s,S),
            ExpSmoothing, EchelonApprox, and the -B RL agents).
  informed  Everything in blind, plus demand_mean(t) — the demand engine's
            expected demand for any period — and demand_noise_scale. Matches
            the -I baselines and the informed RL agents.

The view limits accidental information leakage; it is not a sandbox. Results
submitted for comparison are reviewed for compliance with the declared tier.

Outputs (default results/custom/<name>/):
  episodes.csv     One row per (scenario, seed), same KPI columns as the cache.
  comparison.csv   Per baseline x scenario: paired profit difference and 95% CI.
  summary.csv      Per baseline: mean profit, optimality gap, wins/ties/losses.
  run_info.json    Policy, tier, seeds, git commit, and package versions.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Scenario grid, canonical seeds, KPI extraction, and cache layout all come
# from the canonical runner so this script cannot drift from the paper protocol.
import run_benchmarks as rb  # noqa: E402

TIERS = ('blind', 'informed')

# Information tier of each published baseline (see _make_agent in run_benchmarks.py).
BASELINE_TIERS = {
    'Oracle': 'upper bound',
    'MSSP': 'blind', 'MSSP-I': 'informed',
    'DLP': 'blind', 'DLP-I': 'informed',
    'Newsvendor': 'blind', 'Newsvendor-I': 'informed',
    '(s,S)': 'blind', '(s,S)-I': 'informed',
    'ExpSmoothing': 'blind', 'ExpSmoothing-I': 'informed',
    'EchelonApprox': 'blind', 'EchelonApprox-I': 'informed',
    'PPO-MLP': 'informed', 'PPO-GNN': 'informed', 'PPO-Transformer': 'informed',
    'ST-PPO': 'informed', 'Residual': 'informed', 'SAC': 'informed',
    'GNN-IL': 'informed', 'DAgger-G': 'informed', 'PPO-MLP-v1': 'informed',
    'PPO-MLP-B': 'blind', 'PPO-GNN-B': 'blind', 'ST-PPO-B': 'blind',
    'Residual-B': 'blind', 'SAC-B': 'blind', 'DAgger-B': 'blind',
    'PPO-MLP-raw': 'blind',  # raw observation only, no demand features
    'LLM-Policy-C': 'blind',
}

# Demand distribution parameters stored on retail edges; hidden from blind agents.
_DEMAND_EDGE_ATTRS = ('demand_dist', 'dist_param')


# ---------------------------------------------------------------------------
# What a custom agent is allowed to see
# ---------------------------------------------------------------------------
class AgentView:
    """Read-only view of an environment, restricted by information tier."""

    def __init__(self, env, tier):
        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        self.tier = tier
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_periods = env.num_periods
        self.backlog = env.backlog
        self.alpha = env.alpha

        net = env.network
        graph = env.graph.copy()
        if tier == 'blind':
            for e in net.retail_links:
                for k in _DEMAND_EDGE_ATTRS:
                    graph.edges[e].pop(k, None)
        self.graph = graph
        self.network = SimpleNamespace(
            main_nodes=list(net.main_nodes), rawmat=list(net.rawmat),
            factory=list(net.factory), distrib=list(net.distrib),
            retail=list(net.retail), market=list(net.market),
            reorder_links=list(net.reorder_links), retail_links=list(net.retail_links),
            lead_times=dict(net.lead_times), node_map=dict(net.node_map),
        )

        # Observation layout, mirroring CoreEnv._update_state():
        # [demand(n_retail) | backlog(n_retail) | inventory(n_main) | pipeline | (t, goodwill)]
        n_retail, n_main = len(net.retail_links), len(net.main_nodes)
        demand = slice(0, n_retail)
        backlog = slice(n_retail, 2 * n_retail)
        inventory = slice(2 * n_retail, 2 * n_retail + n_main)
        self.pipeline_slices = {}  # reorder link -> slice; element k arrives in k periods
        i = p0 = inventory.stop
        for e in net.reorder_links:
            L = net.lead_times[e]
            if L > 0:
                self.pipeline_slices[e] = slice(i, i + L)
                i += L
        self.obs_slices = dict(demand=demand, backlog=backlog, inventory=inventory,
                               pipeline=slice(p0, i), extra=slice(i, i + env.extra_features_dim))

        self._demand_history = lambda: np.array(env.D[:env.period], copy=True)
        if tier == 'informed':
            engine = env.demand_engine
            self.demand_mean = lambda t: float(engine.get_current_mu(t))
            self.demand_noise_scale = float(getattr(engine, 'noise_scale', 1.0))

    def demand_history(self):
        """Realized retail demand for periods 0..t-1, shape (t, n_retail_links)."""
        return self._demand_history()


# ---------------------------------------------------------------------------
# Loading and running the custom policy
# ---------------------------------------------------------------------------
def load_policy_class(spec):
    """Resolve 'path/to/file.py:ClassName' or 'package.module:ClassName'."""
    target, _, cls_name = spec.rpartition(':')
    if not target or not cls_name:
        raise SystemExit("--policy must look like path/to/file.py:ClassName "
                         "or package.module:ClassName")
    if target.endswith('.py') or os.sep in target:
        path = os.path.abspath(target)
        if not os.path.exists(path):
            raise SystemExit(f"Policy file not found: {path}")
        sys.path.insert(0, os.path.dirname(path))  # allow sibling imports
        mod_spec = importlib.util.spec_from_file_location(
            os.path.splitext(os.path.basename(path))[0], path)
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[mod_spec.name] = module
        mod_spec.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        module = importlib.import_module(target)
    try:
        return getattr(module, cls_name)
    except AttributeError:
        raise SystemExit(f"{cls_name!r} not found in {target}")


def run_episode(make_agent, scfg, seed):
    """Run one episode in the canonical order: build env, build agent, reset, roll out.

    make_agent(env, scfg, seed) returns an object with get_action(obs, t).
    """
    env = rb.CoreEnv(**rb._env_kwargs(scfg))
    t_plan = time.perf_counter()
    agent = make_agent(env, scfg, seed)
    plan_time = time.perf_counter() - t_plan

    obs, _ = env.reset(seed=seed)
    done = False
    t = 0
    t0 = time.perf_counter()
    while not done:
        action = np.asarray(agent.get_action(obs, t), dtype=np.float64)
        if action.shape != env.action_space.shape:
            raise ValueError(f"get_action returned shape {action.shape}, "
                             f"expected {env.action_space.shape}")
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
        t += 1
    replay_time = time.perf_counter() - t0

    kpis = rb._extract_kpis(env)
    kpis.update(PlanTime=plan_time, ReplayTime=replay_time, Latency=plan_time + replay_time)
    return kpis


def evaluate(policy_cls, tier, scenarios, seeds, name):
    """Evaluate the policy on scenarios x seeds; return one row per episode."""
    def make_agent(env, scfg, seed):
        return policy_cls(AgentView(env, tier), seed)

    rows = []
    for si, scfg in enumerate(scenarios, 1):
        profits = []
        for seed in seeds:
            kpis = run_episode(make_agent, scfg, seed)
            profits.append(kpis['Profit'])
            rows.append(dict(Agent=name, Tier=tier, ScenarioKey=rb.scenario_id(scfg),
                             Block=scfg['block'], Network=scfg['network'],
                             Demand=scfg['demand_label'], Goodwill=scfg['use_goodwill'],
                             Backlog=scfg['backlog'], MARL=scfg['marl'], Seed=seed, **kpis))
        print(f"  [{si:2d}/{len(scenarios)}] {rb.scenario_id(scfg):<62} "
              f"mean profit {np.mean(profits):>9.1f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired comparison against the published per-seed results
# ---------------------------------------------------------------------------
def load_baseline(agent_id):
    path = rb._cache_path(agent_id)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if 'Failed' in df.columns:
        df = df[df['Failed'] != True]  # noqa: E712 — column may hold NaN
    return df.dropna(subset=['Profit'])


def compare(custom, baseline_ids):
    """Return (per-scenario comparison, per-baseline summary) DataFrames."""
    oracle = load_baseline('Oracle')
    oracle_mean = None
    if oracle is not None:
        o = custom[['ScenarioKey', 'Seed']].merge(oracle[['ScenarioKey', 'Seed', 'Profit']])
        oracle_mean = o.groupby('ScenarioKey')['Profit'].mean()

    def opt_gap(scenario_means):
        # Paper definition: (Oracle - agent) / |Oracle| x 100, averaged over scenarios
        if oracle_mean is None:
            return np.nan
        o = oracle_mean.reindex(scenario_means.index)
        return float(((o - scenario_means) / o.abs() * 100).mean())

    comp_rows, summary_rows = [], []
    custom_name = custom['Agent'].iloc[0]
    custom_means = custom.groupby('ScenarioKey')['Profit'].mean()
    summary_rows.append(dict(Agent=custom_name, Tier=custom['Tier'].iloc[0],
                             Scenarios=len(custom_means), MeanProfit=custom_means.mean(),
                             MeanFillRate=custom['FillRate'].mean(),
                             MeanOptGap_Pct=opt_gap(custom_means)))

    for aid in baseline_ids:
        base = load_baseline(aid)
        if base is None:
            continue
        m = custom.merge(base[['ScenarioKey', 'Seed', 'Profit', 'FillRate']],
                         on=['ScenarioKey', 'Seed'], suffixes=('', '_Base'))
        if m.empty:
            continue
        per_scen = []
        for sid, g in m.groupby('ScenarioKey', sort=False):
            d = (g['Profit'] - g['Profit_Base']).to_numpy()
            n = len(d)
            half = (stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            lo, hi = d.mean() - half, d.mean() + half
            outcome = 'win' if lo > 0 else ('loss' if hi < 0 else 'tie')
            per_scen.append(d.mean())
            comp_rows.append(dict(Baseline=aid, BaselineTier=BASELINE_TIERS.get(aid, '?'),
                                  ScenarioKey=sid, Block=g['Block'].iloc[0], Pairs=n,
                                  Profit=g['Profit'].mean(), BaselineProfit=g['Profit_Base'].mean(),
                                  ProfitDiff=d.mean(), ProfitDiff_CI95_Low=lo,
                                  ProfitDiff_CI95_High=hi, Outcome=outcome,
                                  FillRate=g['FillRate'].mean(),
                                  BaselineFillRate=g['FillRate_Base'].mean()))

        scen = pd.DataFrame([r for r in comp_rows if r['Baseline'] == aid])
        per_scen = np.asarray(per_scen)
        p = np.nan
        if len(per_scen) >= 6 and np.any(per_scen != 0):
            p = float(stats.wilcoxon(per_scen).pvalue)
        base_means = m.groupby('ScenarioKey')['Profit_Base'].mean()
        summary_rows.append(dict(
            Agent=aid, Tier=BASELINE_TIERS.get(aid, '?'), Scenarios=len(scen),
            MeanProfit=base_means.mean(), MeanFillRate=m['FillRate_Base'].mean(),
            MeanOptGap_Pct=opt_gap(base_means),
            CustomMinusBaseline=float(per_scen.mean()),
            Wins=int((scen['Outcome'] == 'win').sum()),
            Ties=int((scen['Outcome'] == 'tie').sum()),
            Losses=int((scen['Outcome'] == 'loss').sum()),
            Wilcoxon_p=p))

    summary = pd.DataFrame(summary_rows).sort_values('MeanProfit', ascending=False)
    for col in ('Wins', 'Ties', 'Losses'):
        summary[col] = summary[col].astype('Int64')
    return pd.DataFrame(comp_rows), summary.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git_commit():
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=rb.PROJECT_ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                        cwd=rb.PROJECT_ROOT, stderr=subprocess.DEVNULL,
                                        text=True).strip()
        return sha + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


def _file_sha256(policy_cls):
    try:
        path = sys.modules[policy_cls.__module__].__file__
        with open(path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return None


def run_info(args, policy_cls, name, scenarios, seeds):
    import gymnasium

    import gym_invmgmt
    return dict(
        name=name, policy=args.policy, policy_file_sha256=_file_sha256(policy_cls),
        tier=args.tier, seeds=list(seeds), blocks=sorted({s['block'] for s in scenarios}),
        scenarios=len(scenarios), repo_commit=_git_commit(),
        python=platform.python_version(), numpy=np.__version__,
        gymnasium=gymnasium.__version__, gym_invmgmt=gym_invmgmt.__version__,
        created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Evaluate a custom policy against the published gym-invmgmt baselines')
    parser.add_argument('--policy', required=True,
                        help='path/to/file.py:ClassName or package.module:ClassName')
    parser.add_argument('--tier', required=True, choices=TIERS,
                        help='Information your agent uses (see module docstring)')
    parser.add_argument('--name', default=None, help='Label for results (default: class name)')
    parser.add_argument('--seeds', type=int, default=len(rb.CANONICAL_SEEDS),
                        help=f'Number of canonical seeds (1-{len(rb.CANONICAL_SEEDS)}, default: all)')
    parser.add_argument('--blocks', nargs='*', default=None,
                        help='Scenario blocks to include (e.g. A_Core B_PaperReplication)')
    parser.add_argument('--baselines', nargs='*', default=None,
                        help='Baseline IDs to compare against (default: all with cached results)')
    parser.add_argument('--out-dir', default=None,
                        help='Output directory (default: results/custom/<name>)')
    args = parser.parse_args(argv)

    if not 1 <= args.seeds <= len(rb.CANONICAL_SEEDS):
        parser.error(f'--seeds must be between 1 and {len(rb.CANONICAL_SEEDS)}; '
                     'published baselines exist only for the canonical seeds')
    seeds = rb.CANONICAL_SEEDS[:args.seeds]

    scenarios = rb.build_all_scenarios()
    if args.blocks:
        scenarios = [s for s in scenarios if s['block'] in args.blocks]
        if not scenarios:
            parser.error(f'no scenarios match --blocks {args.blocks}')

    baseline_ids = args.baselines or list(BASELINE_TIERS)
    unknown = [b for b in baseline_ids if load_baseline(b) is None]
    if args.baselines and unknown:
        parser.error(f'no cached results for baseline(s): {unknown}')
    baseline_ids = [b for b in baseline_ids if b not in unknown]

    policy_cls = load_policy_class(args.policy)
    name = args.name or policy_cls.__name__
    out_dir = args.out_dir or os.path.join(rb.RESULTS_DIR, 'custom', name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nEvaluating {name} ({args.tier}) on {len(scenarios)} scenarios x {len(seeds)} seeds")
    custom = evaluate(policy_cls, args.tier, scenarios, seeds, name)
    comparison, summary = compare(custom, baseline_ids)

    custom.to_csv(os.path.join(out_dir, 'episodes.csv'), index=False)
    comparison.to_csv(os.path.join(out_dir, 'comparison.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    with open(os.path.join(out_dir, 'run_info.json'), 'w', encoding='utf-8') as f:
        json.dump(run_info(args, policy_cls, name, scenarios, seeds), f, indent=2)

    cols = ['Agent', 'Tier', 'Scenarios', 'MeanProfit', 'MeanFillRate', 'MeanOptGap_Pct',
            'CustomMinusBaseline', 'Wins', 'Ties', 'Losses', 'Wilcoxon_p']
    with pd.option_context('display.width', 170, 'display.max_rows', 100,
                           'display.float_format', '{:,.3f}'.format):
        print("\nSummary (paired on matched scenario x seed; Oracle is a "
              "perfect-information upper bound, not a deployable method):\n")
        print(summary[cols].to_string(index=False, na_rep='-'))
    print(f"\nWins/Ties/Losses count scenarios where the 95% CI of "
          f"({name} - baseline) profit is above/overlapping/below zero.")
    print("Baselines covering fewer scenarios are averaged over those scenarios only; "
          "CustomMinusBaseline is always paired.")
    print(f"Results written to {out_dir}")


if __name__ == '__main__':
    main()
