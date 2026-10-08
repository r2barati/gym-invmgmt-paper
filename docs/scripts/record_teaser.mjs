// Render the simulation console to an MP4 teaser (README, slides, social posts).
//
// Usage, from the repository root with the site served locally:
//   (cd docs && python3 -m http.server 8765) &
//   node docs/scripts/record_teaser.mjs http://localhost:8765/ docs/media PPO-Transformer
//
// The optional third argument names the highlighted policy (default: PPO-Transformer,
// falling back to the first policy if that one is not in the data).
// Needs Playwright (Chromium) and ffmpeg on PATH. Frames are captured by seeking the
// player, not by screen recording, so the output is deterministic and drop-free.
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
let chromium;
try { ({ chromium } = require("playwright")); }
catch { ({ chromium } = require(process.env.PLAYWRIGHT_MODULE || "/opt/node22/lib/node_modules/playwright")); }

const url = process.argv[2] || "http://localhost:8765/";
const outDir = process.argv[3] || "docs/media";
const agent = process.argv[4] || "PPO-Transformer";
const FPS = 24, FRAMES_PER_PERIOD = 14, HOLD_SEC = 1.5;

const frames = mkdtempSync(join(tmpdir(), "teaser-"));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1180, height: 1200 }, colorScheme: "light", deviceScaleFactor: 1.5, reducedMotion: "reduce" });
await page.goto(url, { waitUntil: "networkidle" });
await page.waitForFunction(() => window.gimPlayer);
if (!(await page.evaluate((a) => window.gimPlayer.select(a), agent))) console.warn(`${agent} not in the data; using the first policy`);
await page.addStyleTag({ content: ".topbar{display:none!important}" });
const sim = await page.$("#sim");
const T = await page.evaluate(() => window.gimPlayer.periods);
const total = T * FRAMES_PER_PERIOD + Math.round(HOLD_SEC * FPS);
for (let f = 0; f < total; f++) {
  await page.evaluate((t) => window.gimPlayer.seek(t), Math.min(T, f / FRAMES_PER_PERIOD));
  await sim.screenshot({ path: join(frames, `f${String(f).padStart(4, "0")}.png`) });
}
await page.evaluate((t) => window.gimPlayer.seek(t), 14);
await sim.screenshot({ path: join(outDir, "teaser-poster.png") });
await browser.close();

execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-framerate", String(FPS), "-i", join(frames, "f%04d.png"),
  "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24",
  "-preset", "slow", "-movflags", "+faststart", "-an", join(outDir, "teaser.mp4")], { stdio: "inherit" });
rmSync(frames, { recursive: true, force: true });
console.log(`wrote ${outDir}/teaser.mp4 (${total} frames at ${FPS} fps) and teaser-poster.png`);
