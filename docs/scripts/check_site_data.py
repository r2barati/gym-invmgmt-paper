"""Check that the project page's data agrees with the canonical benchmark results.

Run before every deployment (CI and the Pages workflow). It fails if:

- docs/data/leaderboard.json or docs/data/topologies.json differ from what the
  exporters produce from results/ and gym_invmgmt/topologies/ right now, i.e.
  the results changed and the page data was not regenerated,
- the rollouts do not cover exactly the 22 main scenarios of the canonical
  runner, a rollout file is malformed, its network layout no longer matches the
  environment, or a replayed profit disagrees with results/cache_v2 at the
  rollout seed,
- index.html references a local file that does not exist.

Needs only the base package (no PuLP, torch or checkpoints), so it runs in a
few seconds on any runner.

Usage (from the repository root):
    python docs/scripts/check_site_data.py
"""

import json
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_leaderboard  # noqa: E402
import export_topologies  # noqa: E402
from export_rollouts import LEARNED, ROOT, matches_cache, scenario_meta, slug  # noqa: E402
from run_benchmarks import _cache_path, build_all_scenarios, scenario_id  # noqa: E402

DOCS = os.path.join(ROOT, "docs")
SITE = "https://r2barati.github.io/gym-invmgmt-paper/"
errors = []


def fail(msg):
    errors.append(msg)


def load(rel):
    with open(os.path.join(DOCS, rel)) as f:
        return json.load(f)


def roundtrip(obj):
    return json.loads(json.dumps(obj))


def check_exported(rel, built, script):
    """The committed file must equal a fresh export."""
    if load(rel) != roundtrip(built):
        fail(f"{rel} is stale: run `python docs/scripts/{script}` and commit the result")


def check_leaderboard(scenarios):
    try:
        built = export_leaderboard.build()
    except AssertionError as e:  # build() checks the caches against the merged CSV
        return fail(f"results/cache_v2 does not reproduce the means in results/benchmark_final_merged.csv: {e}")
    check_exported("data/leaderboard.json", built, "export_leaderboard.py")
    keys = [s["key"] for s in load("data/leaderboard.json")["scenarios"]]
    if sorted(keys) != sorted(scenario_id(s) for s in scenarios):
        fail("leaderboard scenarios differ from build_all_scenarios() in benchmarks/run_benchmarks.py")


def check_rollout(entry, scfg, index, cache):
    key, path = entry["key"], os.path.join(DOCS, "data", "rollouts", entry["file"])
    if not os.path.exists(path):
        return fail(f"rollouts: {entry['file']} is listed in index.json but missing")
    with open(path) as f:
        data = json.load(f)
    meta, seed = data["meta"], index["seed"]
    if meta != roundtrip(scenario_meta(scfg, seed)):
        return fail(f"rollouts/{entry['file']}: network layout or settings no longer match the environment")
    agents = [r["agent"] for r in data["runs"]]
    if agents != index["agents"]:
        fail(f"rollouts/{entry['file']}: policies {agents} != index.json {index['agents']}")
    T = meta["periods"]
    width = {"X": len(meta["node_index"]), "R": len(meta["reorder_links"]), "S": len(meta["network_links"]),
             "D": len(meta["retail_links"]), "U": len(meta["retail_links"])}
    for r in data["runs"]:
        where = f"rollouts/{entry['file']} {r['agent']}"
        for m, w in width.items():
            rows = len(r[m])
            if rows != (T + 1 if m == "X" else T) or any(len(row) != w for row in r[m]):
                fail(f"{where}: {m} has shape {rows}x{len(r[m][0]) if rows else 0}, expected "
                     f"{T + 1 if m == 'X' else T}x{w}")
        if len(r["P"]) != T or abs(r["P"][-1] - r["kpis"]["profit"]) > 0.11:
            fail(f"{where}: cumulative profit does not end at the episode profit")
        c = cache[r["agent"]]
        row = c[(c["ScenarioKey"] == key) & (c["Seed"] == seed)]
        if row.empty:
            fail(f"{where}: no row in {os.path.relpath(_cache_path(r['agent']), ROOT)} for seed {seed}")
        elif not matches_cache(r["agent"], float(row["Profit"].iloc[0]), r["kpis"]["profit"]):
            fail(f"{where}: profit {r['kpis']['profit']:.1f} disagrees with the cached {float(row['Profit'].iloc[0]):.1f}")


def check_rollouts(scenarios):
    index = load("data/rollouts/index.json")
    main = {scenario_id(s): s for s in scenarios if not s["marl"]}
    listed = [e["key"] for e in index["scenarios"]]
    if sorted(listed) != sorted(main):
        missing, extra = sorted(set(main) - set(listed)), sorted(set(listed) - set(main))
        fail(f"rollouts/index.json does not cover the main scenarios (missing {missing}, unexpected {extra})")
    if index["default"] not in listed:
        fail("rollouts/index.json: default scenario is not in the list")
    if set(index["learned"]) != LEARNED:
        fail("rollouts/index.json: learned policies do not match export_rollouts.py")
    if index["seed"] not in load("data/leaderboard.json")["seeds"]:
        fail(f"rollouts/index.json: seed {index['seed']} is not a canonical benchmark seed")
    files = {e["file"] for e in index["scenarios"]}
    for e in index["scenarios"]:
        if e["file"] != slug(e["key"]) + ".json":
            fail(f"rollouts/index.json: {e['key']} points to {e['file']}")
    on_disk = {f for f in os.listdir(os.path.join(DOCS, "data", "rollouts")) if f != "index.json"}
    for f in sorted(on_disk - files):
        fail(f"rollouts/{f} is not listed in index.json")

    cache = {a: pd.read_csv(_cache_path(a)) for a in index["agents"]}
    for e in index["scenarios"]:
        if e["key"] in main:
            check_rollout(e, main[e["key"]], index, cache)
    return len(index["scenarios"]), len(index["agents"]), index["seed"]


def check_references():
    with open(os.path.join(DOCS, "index.html")) as f:
        html = f.read()
    refs = set(re.findall(r'(?:href|src|poster)="(?![a-z]+:|#|//)([^"?#]+)', html))
    refs |= set(re.findall(r'url\("?(?!#|data:)([^")?#]+)', html))
    refs |= set(re.findall(r'"(data/[^"]+\.json)"', html))
    refs |= {p for p in re.findall(re.escape(SITE) + r'([^"\s?#]*)', html) if p}
    for ref in sorted(refs):
        if not os.path.exists(os.path.join(DOCS, ref)):
            fail(f"index.html references {ref}, which does not exist in docs/")
    return len(refs)


def main():
    scenarios = build_all_scenarios()
    check_leaderboard(scenarios)
    check_exported("data/topologies.json", export_topologies.build(), "export_topologies.py")
    n_scn, n_agents, seed = check_rollouts(scenarios)
    n_refs = check_references()
    if errors:
        print(f"{len(errors)} problem(s) with the project page data:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"leaderboard.json and topologies.json match a fresh export; {n_scn} rollout files x {n_agents} "
          f"policies agree with results/cache_v2 at seed {seed}; {n_refs} local references resolve.")


if __name__ == "__main__":
    main()
