"""Export the shipped network topologies (built-in presets and YAML library) for the project page.

Usage (from the repository root):
    python docs/scripts/export_topologies.py --out docs/data/topologies.json
"""

import argparse
import glob
import json
import os
import sys

import networkx as nx
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from gym_invmgmt import CoreEnv  # noqa: E402

ORDER = ["raw_materials", "manufacturer", "distributor", "retailer"]


def describe(env, name, source, blurb):
    net, g = env.network, env.graph
    role = {int(n): k for k, ids in net.levels.items() if k in ORDER for n in ids}
    for m in net.market:
        role[int(m)] = "market"
    # Column = longest path from a source, so assembly stages get their own column.
    depth = {}
    for n in nx.topological_sort(g):
        depth[int(n)] = max((depth[int(p)] + 1 for p in g.predecessors(n)), default=0)
    return {
        "name": name,
        "source": source,
        "blurb": blurb,
        "nodes": [{"id": int(n), "depth": depth[int(n)], "role": role.get(int(n), "other")} for n in g.nodes],
        "edges": [{"from": int(a), "to": int(b), "L": int(env.graph.edges[a, b].get("L", 0))}
                  for a, b in env.graph.edges],
        "actions": len(net.reorder_links),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "topologies.json"))
    args = ap.parse_args()

    out = [
        describe(CoreEnv(scenario="network"), "Base network", 'CoreEnv(scenario="network")',
                 "The benchmark's divergent multi-echelon network."),
        describe(CoreEnv(scenario="serial"), "Serial", 'CoreEnv(scenario="serial")',
                 "The benchmark's single-path chain."),
    ]
    skip = {"serial.yaml", "divergent.yaml"}  # identical to the two presets above
    for path in sorted(glob.glob(os.path.join(ROOT, "gym_invmgmt", "topologies", "*.yaml"))):
        if os.path.basename(path) in skip:
            continue
        with open(path) as f:
            cfg = yaml.safe_load(f)
        env = CoreEnv(scenario="custom", config_path=path)
        blurb = " ".join(str(cfg.get("description", "")).split())
        out.append(describe(env, cfg.get("name", os.path.basename(path)),
                            os.path.relpath(path, ROOT), blurb))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for t in out:
        print(t["name"], len(t["nodes"]), "nodes", t["actions"], "actions")


if __name__ == "__main__":
    main()
