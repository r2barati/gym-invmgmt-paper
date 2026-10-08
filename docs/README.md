# Project page

Static site for <https://r2barati.github.io/gym-invmgmt-paper/>, deployed from this
folder by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`.

| Path | What it is |
| --- | --- |
| `index.html` | The page: rollout player, leaderboard with confidence intervals and row details, robustness and reliability charts, environment reference, quick start, citation. No build step. |
| `data/leaderboard.json` | Per-scenario means from `results/benchmark_final_merged.csv`, per-seed profits and cost breakdowns from `results/cache_v2/`, and one-line method descriptions. |
| `data/rollouts/` | Per-period state of eight policies (optimization, heuristic and learned) on each of the 22 main scenarios, at benchmark seed 42. One file per scenario plus `index.json`. |
| `data/topologies.json` | The two benchmark networks and the YAML networks in `gym_invmgmt/topologies/`. |
| `media/teaser.mp4`, `media/teaser-poster.png` | Teaser clip of the player for the README, slides and social posts. |
| `fonts/` | Computer Modern Unicode (Serif and Typewriter) web fonts, self-hosted under the SIL Open Font License in `fonts/OFL.txt`. |
| `scripts/` | Generators for everything in `data/` and `media/`. |

Every number on the page is computed in the browser from these files, so regenerating
them after a new benchmark run updates the whole page.

## Regenerate

From the repository root:

```bash
pip install -e ".[or]"
python docs/scripts/export_leaderboard.py
python docs/scripts/export_rollouts.py      # about 5 minutes; MSSP-I re-solves every period
python docs/scripts/export_topologies.py

# Teaser video (needs Node, Playwright with Chromium, and ffmpeg)
(cd docs && python3 -m http.server 8765) &
node docs/scripts/record_teaser.mjs http://localhost:8765/ docs/media
```

`export_rollouts.py` runs the Oracle, which needs `pulp<3` (PuLP 3 removed
`LpVariable.dicts`). The learned policies (PPO-Transformer, PPO-MLP, DAgger-G)
also need `torch`, `stable-baselines3` and the checkpoints from
`download_weights.sh`; without them the script skips those policies.

`.github/workflows/page-data.yml` does all of this on a GitHub runner, checks
every replay against `results/cache_v2/` (`--verify`), and commits the data back
to the branch. It runs on pushes that change the exporter, or manually from the
Actions tab.

## Preview locally

```bash
cd docs && python3 -m http.server 8765
```

Then open <http://localhost:8765/>. Opening `index.html` from disk will not load the
JSON data because browsers block `fetch` on `file://` URLs.

## One-time setup

In the repository settings, open **Pages** and set **Source** to **GitHub Actions**.
The workflow cannot turn Pages on by itself.
