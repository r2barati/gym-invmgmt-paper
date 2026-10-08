# Project page

Static site for <https://r2barati.github.io/gym-invmgmt-paper/>, deployed from this
folder by `.github/workflows/pages.yml` on every push to `main` that touches `docs/`.

| Path | What it is |
| --- | --- |
| `index.html` | The page: animated rollout player, leaderboard, quick start, citation. No build step. |
| `data/leaderboard.json` | Compact export of `results/benchmark_final_merged.csv`. |
| `data/rollouts.json` | Per-period state of four weight-free policies on one benchmark scenario (seed 0). |
| `media/teaser.mp4`, `media/teaser-poster.png` | Teaser clip of the player for the README, slides and social posts. |
| `scripts/` | Generators for everything in `data/` and `media/`. |

## Regenerate

From the repository root:

```bash
pip install -e ".[or]"
python docs/scripts/export_leaderboard.py
python docs/scripts/export_rollouts.py

# Teaser video (needs Node, Playwright with Chromium, and ffmpeg)
(cd docs && python3 -m http.server 8765) &
node docs/scripts/record_teaser.mjs http://localhost:8765/ docs/media
```

`export_rollouts.py` runs the Oracle, which needs `pulp<3` (PuLP 3 removed
`LpVariable.dicts`).

## Preview locally

```bash
cd docs && python3 -m http.server 8765
```

Then open <http://localhost:8765/>. Opening `index.html` from disk will not load the
JSON data because browsers block `fetch` on `file://` URLs.

## One-time setup

In the repository settings, open **Pages** and set **Source** to **GitHub Actions**.
The workflow cannot turn Pages on by itself.
