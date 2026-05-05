#!/usr/bin/env python3
"""
verify_publication_ready.py

The final "red button" verification script for the gym-invmgmt repository.
Runs pytest suite, verifies artifact completeness, and generates a final manifest.
"""
import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
import hashlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, PROJECT_ROOT)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def run_pytest(quick=False, include_llm=False):
    print("=== Running Pytest Suite ===")
    cmd = ["python3", "-m", "pytest", "tests/", "-v"]
    if quick:
        cmd.extend(["-k", "not smoke and not or_heuristics"])
    if not include_llm:
        cmd.extend(["-m", "not llm"])
        
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    if result.returncode != 0:
        print("\n[ERROR] Pytest suite failed! Fix errors before publication.")
        return False
    return True

def verify_roster_and_caches(quick=False):
    print("\n=== Verifying Roster and Caches ===")
    from benchmarks.run_benchmarks import ALL_AGENT_IDS
    
    missing_agents = []
    
    # Check that ALL_AGENT_IDS matches 29
    if len(ALL_AGENT_IDS) != 29:
        print(f"[ERROR] Expected 29 ALL_AGENT_IDS, found {len(ALL_AGENT_IDS)}")
        return False
        
    if not quick:
        print("Validating model zip existences...")
        from tests.test_training_registry_contracts import EXPECTED_PATHS
        for agent, paths in EXPECTED_PATHS.items():
            for top, (z_file, p_file) in paths.items():
                if not os.path.exists(os.path.join(PROJECT_ROOT, "data", "models", z_file)):
                    missing_agents.append(f"{agent} ({top} model)")
                    
    if missing_agents:
        if not quick:
            print(f"[ERROR] Missing trained models for: {missing_agents}")
            print("Cannot pass full verification without all required artifacts.")
            return False
        else:
            print(f"[WARNING] Missing trained models for: {missing_agents}")
            
    return True

def verify_canonical_csv(require_llm=False):
    print("\n=== Verifying Canonical CSV ===")
    import pandas as pd
    from benchmarks.run_benchmarks import ALL_AGENT_IDS
    
    csv_path = os.path.join(PROJECT_ROOT, "results", "benchmark_final_merged.csv")
    if not os.path.exists(csv_path):
        print("[ERROR] Canonical merged CSV not found!")
        return False
        
    df = pd.read_csv(csv_path)
    if len(df) != 26:
        print(f"[ERROR] Canonical CSV has {len(df)} rows, expected 26.")
        return False
    
    # Verify every registered agent has a _Profit column
    LEGACY_MAP = {
        "(s,S)": "sS", "(s,S)-I": "sS_I",
        "ExpSmoothing-I": "ExpSmooth_I",
        "PPO-GNN": "GNN_V3", "PPO-GNN-B": "GNN_V3_B",
    }
    missing = []
    for agent in ALL_AGENT_IDS:
        header = LEGACY_MAP.get(agent, agent.replace('-', '_'))
        if f"{header}_Profit" not in df.columns:
            missing.append(agent)
    
    if missing:
        print(f"[ERROR] CSV missing _Profit columns for: {missing}")
        return False
        
    if require_llm:
        llm_col = "LLM_Policy_C_Profit"
        if llm_col not in df.columns:
            print(f"[ERROR] Full-with-LLM mode requires {llm_col}.")
            return False
        if df[llm_col].notna().sum() != 26:
            print(
                f"[ERROR] {llm_col} is incomplete: "
                f"{df[llm_col].notna().sum()} / 26 scenario rows."
            )
            return False

    print(f"Canonical CSV OK. {len(df)} scenarios, {len(df.columns)} columns, all {len(ALL_AGENT_IDS)} agents present.")
    return True

def generate_manifest():
    print("\n=== Generating Publication Manifest ===")
    from benchmarks.run_benchmarks import ALL_AGENT_IDS, _cache_path
    
    csv_path = Path(PROJECT_ROOT) / "results" / "benchmark_final_merged.csv"
    optional_llm = {}
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            for col in ["LLM_Policy_C_Profit", "LLM_ZS_Direct_Profit", "LLM_InvAgent_C_Profit"]:
                optional_llm[col] = {
                    "present": col in df.columns,
                    "nonnull_scenarios": int(df[col].notna().sum()) if col in df.columns else 0,
                }
        except Exception as exc:
            optional_llm["error"] = str(exc)

    model_artifacts = {}
    try:
        from tests.test_training_registry_contracts import EXPECTED_PATHS
        for agent, paths in EXPECTED_PATHS.items():
            model_artifacts[agent] = {}
            for topo, (z_file, p_file) in paths.items():
                entries = {}
                for label, file_name in [("model", z_file), ("vecnormalize", p_file)]:
                    path = Path(PROJECT_ROOT) / "data" / "models" / file_name
                    entries[label] = {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else None,
                        "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None,
                        "sha256": _sha256(path) if path.exists() else None,
                    }
                model_artifacts[agent][topo] = entries
    except Exception as exc:
        model_artifacts["error"] = str(exc)

    cache_files = {}
    for agent in ALL_AGENT_IDS:
        cache = Path(_cache_path(agent))
        cache_files[agent] = {
            "path": str(cache.relative_to(PROJECT_ROOT)),
            "exists": cache.exists(),
            "bytes": cache.stat().st_size if cache.exists() else None,
            "mtime": datetime.fromtimestamp(cache.stat().st_mtime).isoformat() if cache.exists() else None,
            "sha256": _sha256(cache) if cache.exists() else None,
        }

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "agent_roster_count": len(ALL_AGENT_IDS),
        "agents": ALL_AGENT_IDS,
        "scenarios_count": 26,
        "canonical_csv_path": "results/benchmark_final_merged.csv",
        "optional_llm_columns": optional_llm,
        "model_artifacts": model_artifacts,
        "cache_files": cache_files,
        "notes": "Verified automatically by verify_publication_ready.py"
    }
    
    # Try to get commit hash
    try:
        commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode('ascii').strip()
        manifest["commit_hash"] = commit_hash
    except Exception:
        manifest["commit_hash"] = "unknown"
        
    out_path = os.path.join(PROJECT_ROOT, "results", "manifest.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Skip slow OR/Agent smoke tests")
    parser.add_argument("--full", action="store_true", help="Run full non-LLM suite")
    parser.add_argument("--with-llm", action="store_true", help="Include LLM tests and require LLM-Policy-C in canonical CSV")
    args = parser.parse_args()
    
    if not run_pytest(quick=args.quick, include_llm=args.with_llm):
        sys.exit(1)
        
    if not verify_roster_and_caches(quick=args.quick):
        sys.exit(1)
        
    if not args.quick:
        if not verify_canonical_csv(require_llm=args.with_llm):
            sys.exit(1)
            
        generate_manifest()

    if args.quick:
        print("\nQuick verification passed. Run --full before final release.")
    elif args.with_llm:
        print("\nFull verification with LLM checks passed. The repository is ready.")
    else:
        print("\nFull non-LLM verification passed. Run --with-llm if LLMs are a main-table requirement.")

if __name__ == "__main__":
    main()
