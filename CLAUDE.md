# X-Sample

Local-first AI music sample editor: capture or import audio, analyze it, separate stems, replace vocals/lyrics, transform genre, generate instruments, control fidelity, and edit in a multitrack timeline. Everything runs on the user's machine by default.

Status: **v0.1.0 released (Phase 0 + Phase 1). v0.2.0 (musical analysis) in progress**: DSP analyzers, analysis job/cache/corrections/API/UI built; optional Beat This! small beat/downbeat model (ADR-0012, ONNX, CPU + verified DirectML); real-audio accuracy still below the acceptance targets (`docs/EVALUATION.md`). No generative models. See `docs/ROADMAP.md` for milestones (requirements 13 intro/outro and 14 YouTube publishing are planned) and `docs/DECISIONS.md` for ADRs (0001–0012 accepted).

## Target hardware (reference machine)

- Windows 11 Pro (build 26200), AMD Ryzen 7 3700X (8C/16T), 16 GB RAM
- **AMD Radeon RX 5700 XT, 8 GB VRAM, RDNA1 (gfx1010)**, driver 32.0.21045.16010
- No NVIDIA GPU. CUDA is unavailable and must never be assumed.

## Architecture

```
engine/                     Python 3.12 package `xsample` (uv-managed, uv.lock committed)
  src/xsample/
    config.py               EngineSettings (port, token, budgets, data dir)
    cli.py                  `xsample serve`, `xsample export-openapi`
    netguard.py             blocks non-loopback sockets (tests, workers)
    api/                    FastAPI app; security.py = token/Host/Origin/CORS/headers (ADR-0008)
    jobs/                   kinds.py (allow-listed job kinds + param schemas + estimates),
                            manager.py (queue, budgets, worker supervision, cancel, crash, timeout)
    workers/                main.py (worker entry), tasks.py (task implementations),
                            winjob.py (Job Object: memory limit, kill-on-close),
                            cancel_event.py (named-event cancellation)
    audio/                  ffmpeg.py (probe/decode), media.py (import), peaks.py, render.py (preview/export),
                            capture.py (PyAudioWPatch mic + WASAPI loopback), meter.py
    analysis/               dsp.py (streaming features), rhythm.py, harmony.py, structure.py,
                            schema.py (versioned sidecar), run.py (worker orchestration),
                            effective.py (detected + corrections) — ADR-0011,
                            beat_model.py (Beat This! log-mel/chunking/postprocessing,
                            pinned ONNX exports, BeatTracker) — ADR-0012
    project/                layout.py, model.py (manifest schema), store.py (atomic save/recovery/lock),
                            ops.py (edit operations), service.py (open project + job results), view.py, verify.py
    runtime/                adapters.py (CPU / DirectML / Vulkan adapters), probe.py + probe_cache.py,
                            inference_check.py, smoke_model.py (generated ONNX model),
                            registry.py (model registry, SHA-256), win_gpu.py (registry/DXGI/PDH),
                            vulkan.py (loader-level detection)
  tests/                    pytest; GPU tests marked `gpu`
ui/                         React 19 + TypeScript 5.9 + Vite 7
  src/api/                  client.ts, auth.ts, schema.d.ts (GENERATED — `npm run gen:api`)
  src/components/           Dashboard (Project/System tabs), SystemPanel, BackendsPanel, JobsPanel, ConnectScreen
  src/workspace/            Workspace, Timeline (canvas waveforms + beat grid), AnalysisPanel, Recorder,
                            SamplesPanel, playback (Web Audio)
  tests/e2e/                Playwright (network-isolated fixtures)
scripts/                    start.ps1 (built UI served by engine), dev.ps1 (engine + Vite)
docs/                       ROADMAP.md, DECISIONS.md
```

Rules:
- **The API process never imports onnxruntime or touches the GPU for inference.** All inference, detection and probes run in worker processes (`python -m xsample.workers.main`), one process per job, each in a Windows Job Object (memory limit, killed if the engine dies).
- Feature modules never call ONNX Runtime directly; they go through `runtime/adapters.py`.
- **A backend being available never implies a model runs on it.** Only a passing probe (numerical match vs NumPy/CPU, verified placement) sets `claims_model_support`. Vulkan currently has no runner and claims nothing.
- New job kinds must be added to `jobs/kinds.py` with a strict (`extra="forbid"`) params model and VRAM/RAM estimates. Only `public=True` kinds are accepted by `POST /api/jobs`.
- Long operations must report progress and check `ctx.cancelled()` between bounded chunks.
- The UI holds no audio processing logic; the engine is the source of truth.
- Non-destructive editing (ADR-0006, ADR-0009): source audio is never modified; positions are integer samples; the manifest is only changed through `ProjectService.edit`.
- Audio job kinds (`import_audio`, `export_audio`, `record_audio`, `list_audio_devices`, `verify_project`, `analyze_audio`) are `public=False`; only the project routes submit them. Finished jobs are applied to the project in `EngineState._on_job_finished`.
- Analysis results are sidecars in `analysis/results/<source>/<cache_key>.json` (key = media PCM hash + analyzer version + schema version + params + beat model identity when installed); bump `ANALYZER_VERSION` when results can change. User corrections live in the manifest (`analysis`), never in the cache. Never report unavailable recognition as a result; scores are not calibrated probabilities.
- Beat model files are verified against `beat_model.PINNED` (the trust root), never against `registry.json`; the worker hashes the bytes it loads. DirectML is reported only after a strict, profiled run of the first chunk and a CPU parity check; otherwise the piece runs on CPU and the result says why. Clips shorter than one 1500-frame chunk use the dynamic export on CPU (never zero-pad to 1500). Torch is only for `tools/export_beat_this.py` in an isolated temporary venv.
- Evaluation datasets live in `%LOCALAPPDATA%\X-Sample\eval-datasets` (manifests with SHA-256, licenses), never in Git. Fetch only small subsets, with approval. `engine/tools/bench_synthetic.py` and `engine/tools/eval_real_audio.py` report synthetic and real-audio accuracy separately.
- Recording only starts from an explicit user action; device listing opens no stream. WASAPI render/loopback entries must be the WASAPI host-API ones (the same name also exists under MME). Recordings never leave the machine (ADR-0010).

## AMD RX 5700 XT compatibility requirements

- **Backends:** ONNX Runtime DirectML EP (GPU), Vulkan via model-specific native runners (none yet), CPU EP (mandatory reference). No CUDA/ROCm dependencies.
- **Package:** `onnxruntime-directml` only — never install `onnxruntime` alongside it (same module name).
- **DirectML session settings:** `enable_mem_pattern=False`, sequential execution. Strict runs set `session.disable_cpu_ep_fallback=1`; non-strict runs must report fallback nodes from the profiler.
- **Verification:** ORT always lists `CPUExecutionProvider` in `get_providers()`; placement must be checked via profiling (`adapters.parse_profile`). Adapter identity is checked via PDH GPU-memory counters mapped to DXGI LUIDs (`win_gpu.py`).
- **Memory:** VRAM budget 6.5 GiB default; one GPU job at a time; per-worker RAM limit 4 GiB default. Workers set `OPENBLAS_NUM_THREADS=4` (OpenBLAS otherwise commits ~30 MB per thread).
- **Chunking:** choose chunk sizes from measured per-chunk timings; do not rely on a fixed driver timeout value.
- WMI `AdapterRAM` is capped at 4 GiB — use `win_gpu.registry_gpus()` / DXGI for VRAM.
- Known DirectML gaps found so far: `Det`, `Unique` fall back to CPU.

## Privacy rules

- **Local-first by default.** No project audio, lyrics, recordings, or metadata leaves the machine without the user's explicit approval for that specific action.
- The engine binds to `127.0.0.1` only, requires the per-launch token, and validates Host and Origin. No telemetry.
- Cloud/external services (including the Abracadabrax Background Music Generator MCP) are opt-in and never called automatically. Claude must not invoke them during development without asking.
- **Model downloads require approval.** `ModelRegistry.download()` deliberately raises; models are installed only from verified local files (size + SHA-256). Models live in `%LOCALAPPDATA%\X-Sample\models`, never in the repo.
- Respect model licenses; record restrictions in the registry and show them to the user.
- Voice conversion features must require that the user confirms rights/consent for the target voice.
- User audio is git-ignored. Only small, license-cleared fixtures under `tests/fixtures/audio/` may be committed.

## Coding standards

**Python (engine)**
- Python 3.12 via `uv` (`uv sync`, `uv run ...`). `python` on PATH is the Microsoft Store stub — always use `uv run`.
- `ruff format`, `ruff check` (rules in `pyproject.toml`), `mypy --strict` over `src/xsample`.
- Type hints on all public functions. Audio is `numpy.float32`, shape `(channels, samples)` (batched as `(1, channels, samples)` for models), sample rate passed explicitly.
- `pathlib` for paths; handle spaces and non-ASCII. Log with `logging`; never log audio content, lyrics or tokens.
- Dependencies use bounded ranges; update `uv.lock` deliberately.

**TypeScript (ui)**
- Strict TS (`noUncheckedIndexedAccess`), ESLint `strictTypeChecked`, Prettier (100 cols).
- API types come from `src/api/schema.d.ts`, regenerated with `npm run gen:api` after any API schema change. Never hand-edit it.
- No inline scripts/styles in `index.html` (the engine serves the build with a strict CSP).
- Token handling: only via `src/api/auth.ts` (URL fragment → sessionStorage).

**General**
- Comments explain *why*, not *what*. New architectural choices get an ADR in `docs/DECISIONS.md`.
- Do not commit or push without user approval.
- All Git commits must use the repository owner's configured Git identity. Do not add Claude, Anthropic, or AI-generated co-author attribution, commit trailers, or promotional footers. Verify commit metadata before pushing. Never rewrite historical commits or force-push without explicit approval.

## Commands

```powershell
# Engine (from engine/)
uv sync                              # install locked deps
uv run pytest                        # unit + integration tests (GPU tests excluded)
uv run pytest -m gpu                 # DirectML verification on the reference GPU
uv run pytest -m hardware            # opt-in: real mic + loopback recording (plays a quiet tone)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy
uv run xsample serve --open          # run engine (+ built UI) and open browser

# UI (from ui/)
npm ci
npm run check                        # typecheck + lint + format check
npm run gen:api                      # regenerate API types from the engine
npm run build
npx playwright install chromium      # once
npm run test:e2e                     # builds UI, starts an engine on :8766, runs Playwright

# Whole app
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
powershell -ExecutionPolicy Bypass -File scripts\dev.ps1     # Vite hot reload + engine
```

## Testing requirements

- **Engine unit/integration (pytest):** synthetic signals generated in tests; jobs tested through real worker processes (success, progress ordering, cooperative and forced cancellation, crash, timeout, memory limit, budgets).
- **Security tests:** every protected route rejects missing/wrong tokens; foreign Host/Origin/preflight/cross-site requests are rejected; test-only job kinds are disabled by default.
- **Backend parity:** CPU tests always run; DirectML tests (`-m gpu`) must run strict, verify placement via profiling, verify the physical adapter, compare to NumPy and CPU, and record metrics to `engine/test-reports/directml_smoke.json`. Set `XSAMPLE_EXPECT_GPU` to override the expected adapter name.
- **Long audio:** processing must be chunked; memory must not grow with duration (10-minute tests on CPU and DirectML).
- **UI:** `npm run check` must pass; Playwright e2e covers auth, CSP, dashboard, probes, jobs, cancellation, layout widths, and fails on any non-local request or console error.
- **No network in tests.** `netguard` blocks non-loopback sockets in pytest and in workers; Playwright aborts non-local requests.
- Recording tests (`-m hardware`) use real devices; never claim hardware capture passes unless they were actually run. Recorded test audio must be deleted.
- A change is done when engine pytest (+ `-m gpu` on the reference machine), ruff, mypy, `npm run check` and `npm run test:e2e` pass.

## Environment notes

- FFmpeg 9.0.2 (full build) and ffprobe are on PATH (required by import and by test fixtures).
- Node 24 / npm 11. npm's allow-scripts blocked esbuild's postinstall; it is not needed (the platform binary ships as an optional dependency).
- Not installed: SoX, Rubber Band CLI, GitHub CLI, pnpm, conda, ROCm/HIP SDK.
- MCP: Playwright is configured in `.mcp.json` and also via a plugin (duplicate tools).
