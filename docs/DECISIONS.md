# Architectural Decisions

Lightweight ADRs. Status values: **Proposed** (awaiting user approval), **Accepted**, **Superseded by ADR-XXXX**, **Rejected**.

Template:

```
## ADR-XXXX — Title
- Status:
- Date:
- Context:
- Decision:
- Consequences:
- Risks:
- Alternatives considered:
```

---

## ADR-0001 — Local-first with explicit opt-in for external services
- Status: Accepted (from project brief)
- Date: 2026-10-28
- Context: The project edits users' private audio and voices. The development environment also exposes external MCP services (e.g. a cloud music generator).
- Decision: All processing runs locally by default. No audio, lyrics or metadata is sent anywhere without explicit, per-action approval. The engine binds to 127.0.0.1. Model downloads are approval-gated and recorded with license and hash.
- Consequences: Feature quality is bounded by models that run on 8 GB VRAM / 16 GB RAM. Cloud features, if ever added, live in a separate, clearly labelled module.
- Alternatives considered: Hybrid cloud inference by default — rejected for privacy.

## ADR-0002 — Python engine + web UI over an authenticated localhost API
- Status: Accepted
- Date: 2026-10-28
- Context: The AI audio ecosystem is Python-centric. The UI needs a waveform/timeline editor and must be testable with Playwright. A server on 127.0.0.1 is still reachable by every local process and by any web page open in the user's browser (cross-site requests, DNS rebinding).
- Decision: Python package `xsample` (FastAPI + uvicorn) owns state, DSP and orchestration. TypeScript/React/Vite UI talks to it over HTTP. The API requires a per-launch bearer token and enforces Host and Origin validation (ADR-0008). The built UI is served by the engine on the same origin with a strict CSP; during development the Vite server is an explicitly allowed origin. Phase 0 delivers job updates by authenticated HTTP polling; a WebSocket channel may be added later with the same token/Origin rules.
- Consequences: Clear separation; the UI is testable in a browser and the engine headless. Two toolchains. The API passes file references and metadata, never bulk audio as JSON. Where real-time playback lives (engine vs Web Audio) is decided before Phase 8.
- Risks: Browser memory limits for long multitrack sessions (mitigated by streaming peaks/previews from the engine); token handling in the browser (kept in sessionStorage, delivered via URL fragment).
- Alternatives considered: Qt/PySide6 desktop app; Electron with Node-native audio; DAW plugin only.

## ADR-0003 — Modular inference adapters: DirectML, Vulkan, CPU
- Status: Accepted
- Date: 2026-10-28
- Context: Reference GPU is an AMD RX 5700 XT (RDNA1, gfx1010, 8 GB). CUDA is unavailable; ROCm/HIP on Windows does not officially support gfx1010. DirectML runs on DirectX 12 GPUs but its ONNX Runtime package lags upstream (1.24.x vs 1.31.x) and not every operator runs on it. There is no general-purpose Python Vulkan inference runtime; Vulkan acceleration comes from specific native runners (e.g. ggml/whisper.cpp builds).
- Decision:
  - `runtime/` exposes backend adapters: **CPU** (ONNX Runtime CPU EP — mandatory reference), **DirectML** (ONNX Runtime DirectML EP), **Vulkan** (native runner executables, model-specific).
  - **No model is assumed to support any backend.** Each model declares candidate backends; before first use on a machine the backend is **probed** with a known input and compared to the CPU/NumPy reference. Results (pass/fail, error, timing, memory) are cached keyed by model hash, backend, adapter, driver version and runtime version, and re-probed when any of these change.
  - Strict GPU runs disable CPU-EP fallback; non-strict runs report which nodes fell back.
  - Vulkan detection reports driver/device presence only and claims no model support until a specific runner is approved and probed.
  - PyTorch-only models may get an optional per-model adapter (CPU, or torch-directml if it passes probes), never a global assumption.
- Consequences: Conversion and parity work per model. Some models will be CPU-only. Switching to Windows ML or a future ROCm path is contained to new adapters.
- Risks: DirectML maintenance pace; operator gaps; driver-specific numerical differences.
- Alternatives considered: torch-directml as the main path; ROCm on Windows/WSL2 (unsupported for RDNA1); CPU only.

## ADR-0004 — Python 3.12 with locked dependencies
- Status: Accepted (supersedes the earlier 3.11 proposal)
- Date: 2026-10-28
- Context: Current numpy (2.5) and scipy (1.18) no longer publish Python 3.11 wheels. onnxruntime-directml publishes cp311–cp314 wheels. The machine default is 3.14, which some ML packages may lag.
- Decision: The engine targets Python 3.12 (`requires-python = ">=3.12,<3.13"`), managed by uv with a committed `uv.lock`. Dependencies use bounded ranges at known-stable versions, not automatic newest releases. The UI uses `package-lock.json` with versions chosen for mutual peer compatibility (e.g. TypeScript 5.9 because typescript-eslint and openapi-typescript do not yet support 6+/7).
- Consequences: Reproducible installs; the Store `python` stub is never used.
- Alternatives considered: 3.11 (losing current numpy/scipy); 3.14 (wheel availability risk); conda.

## ADR-0005 — Resource budgets, isolated workers, explicit unloading
- Status: Accepted
- Date: 2026-10-28
- Context: 8 GB VRAM (~0.5–1 GB used by the desktop; no integrated GPU) and 16 GB RAM are shared with the OS, browser and other apps. Unloading a model inside a long-lived process does not reliably return GPU memory.
- Decision:
  - All inference runs in **worker processes** (ADR-0007); a worker exits after its job, which is the explicit and reliable unload. Inside a worker, sessions are released before exit.
  - The engine enforces a configurable **VRAM budget** (default 6.5 GiB) and a **per-worker RAM limit** via a Windows Job Object. GPU jobs run one at a time.
  - Jobs declare estimated memory; the engine rejects jobs that exceed budgets instead of attempting them.
  - Long audio is processed in bounded chunks whose size is chosen from **measured** per-chunk timings (no fixed driver-timeout assumption).
  - Every job reports backend, adapter, wall time, peak worker RAM and (where Windows counters allow) dedicated GPU memory.
- Consequences: Process start-up and model load cost per job (acceptable for batch operations; a persistent per-model worker may be added later with the same isolation).
- Risks: GPU memory counters are approximate; Job Object limits apply after process start.
- Alternatives considered: Keeping all models resident (does not fit); in-process inference (unload and crash isolation problems).

## ADR-0006 — Non-destructive project storage with a sample-accurate time base
- Status: Accepted
- Date: 2026-10-28
- Context: AI edits are expensive and must be comparable (A/B, fidelity control) and reversible. Lyrics sync, vocal replacement and multitrack editing need positions that survive edits such as trimming and time-stretching.
- Decision:
  - A project is a folder: `project.json` (tracks, clips, operation graph, metadata), `sources/` (immutable originals, content-addressed by SHA-256), `analysis/` (binary sidecars such as `.npy` for pitch, beats, alignments), `cache/` (re-creatable renders and stems).
  - **Positions are integer sample counts** at a declared sample rate; no floating-point seconds in stored data.
  - Each clip has a **source-to-timeline time map** (piecewise-linear anchors) so lyric/word timings and analysis attached to source audio can be projected onto the timeline after edits.
  - Every derived artefact records **versioned processing metadata**: operation id, input hashes, model id + hash, backend, parameters, engine version. Cache entries are keyed by this metadata.
- Consequences: Cheap undo/redo and re-render; deterministic invalidation. Disk usage grows; cache cleanup required. Implementation starts in Phase 1.
- Alternatives considered: SQLite project database (possible later); destructive edits (rejected); float-second positions (rejected — drift and rounding).

## ADR-0007 — Worker isolation, cancellation and crash recovery
- Status: Accepted
- Date: 2026-10-28
- Context: GPU drivers, native libraries and models can crash, hang, leak or exhaust memory. Conflicting dependencies between future models may need separate environments.
- Decision:
  - Each job runs in a separate process (`python -m xsample.workers.main`), started with an explicit job specification on stdin.
  - Protocol: newline-delimited JSON on a dedicated stdout channel (`progress`, `result`, `error`, `cancelled`); library output is redirected to stderr, whose tail is kept for diagnostics.
  - Each worker is placed in a Windows **Job Object** with a memory limit and `KILL_ON_JOB_CLOSE`, so workers cannot outlive the engine.
  - **Cancellation** is cooperative first (`cancel` message on stdin, checked between chunks), then forced termination after a grace period.
  - **Crash recovery:** a worker exit without a result marks the job `failed` with exit code and stderr tail; the engine stays up and the next job starts a fresh worker. Wall-clock timeouts terminate stuck workers.
  - Workers inherit network blocking when the engine runs offline/tests.
  - Job history is in-memory in Phase 0; persistence comes with the project format.
- Consequences: Robustness and reliable memory release at the cost of per-job start-up time (~1 s).
- Alternatives considered: Threads (no crash isolation); multiprocessing pools (harder to kill and limit cleanly on Windows).

## ADR-0008 — Local API authentication, origin validation and security boundaries
- Status: Accepted
- Date: 2026-10-28
- Context: Binding to 127.0.0.1 does not stop a malicious web page from sending requests to the engine, nor DNS-rebinding attacks.
- Decision:
  - **Token:** a random per-launch token (≥256 bits) required as `Authorization: Bearer` on every `/api` route except `/api/health`. Compared in constant time. Delivered to the browser via the launch URL fragment (`#token=…`, never sent to servers), then kept in `sessionStorage`.
  - **Host validation:** only `127.0.0.1:<port>` and `localhost:<port>` (blocks DNS rebinding).
  - **Origin validation:** requests carrying an `Origin` must match the allow-list (engine origin, dev UI origin); `null` and unknown origins get 403 — including CORS preflights. Requests without `Origin` but with `Sec-Fetch-Site: cross-site|same-site` are rejected. Requests with neither header are treated as non-browser clients and still need the token.
  - **CORS:** only allow-listed origins, no credentials/cookies, limited methods and headers.
  - **Headers:** `nosniff`, `no-referrer`, `no-store` for API responses; strict CSP and `frame-ancestors 'none'` for the served UI.
  - **Limits:** request body size cap; job kinds are an allow-list; test-only job kinds are disabled unless explicitly enabled.
  - **Boundary:** the engine trusts the same Windows user (who could read the token from process memory or environment). It does not defend against malware running as that user.
- Consequences: Phase 0 API docs (`/docs`) are disabled by default; the OpenAPI schema is exported offline for UI type generation.
- Alternatives considered: Cookies (CSRF exposure); no auth on loopback (unsafe); OS named pipes (not browser-reachable).

## ADR-0009 — Audio pipeline and project storage details (Phase 1)
- Status: Accepted (refines ADR-0006)
- Date: 2026-10-28
- Context: ADR-0006 fixed the non-destructive principle; Phase 1 needed concrete formats, an import path and a playback strategy.
- Decision:
  - **Project folder:** `project.json` (strict pydantic schema, `format_version`, integrity validation on load, atomic save with `.bak` and backup recovery), `sources/<sha256>.<ext>` (read-only originals, content-addressed, duplicates reuse the same file), `cache/media/` (float32 WAV working copy at the project sample rate), `analysis/peaks/*.npz` (peak pyramid), `cache/{incoming,recording,tmp}`, `exports/`. A `.xsample.lock` file (pid + process start time) prevents two engines opening one project; stale locks are taken over.
  - **Project sample rate** is chosen at creation (22.05–96 kHz, default 48 kHz). Imports are resampled by FFmpeg to it; mono/stereo conversion is explicit (`keep` / `mono` / `stereo`). All positions are integer samples at that rate.
  - **Decoding:** FFmpeg (pipe, float32, 65 536-frame chunks) decodes every format; libsndfile writes/reads WAV and FLAC. Memory is bounded by the chunk size, not by duration.
  - **Edits** are operations on the manifest (`trim`, `slice` into *samples*, `move`, `delete`, tracks). Each edit is applied to a copy, validated, saved, then swapped in. Clips carry source-to-timeline anchor pairs; only unity maps are accepted in Phase 1, the helpers are general for later stretch.
  - **Waveforms:** int16 min/max pyramid (base block 64 frames, factor 4); the API returns exact per-column extrema for any visible range and falls back to raw reads when zoomed below the base block.
  - **Playback:** the engine renders a 16-bit WAV preview (max 120 s) of the requested range; the UI plays it with Web Audio. The UI mixes nothing. A real-time engine-side mixer remains the Phase 8 decision.
  - **Export:** a worker job renders from a server-built render plan to `exports/` (WAV 16/24/float, FLAC 16/24), never overwriting an existing file; results are downloaded through the API.
  - **Import upload:** the browser streams the file body to `cache/incoming/` (size-capped, extension allow-list); a worker job then hashes, stores the original, decodes and builds peaks. Failure or cancellation removes every partial file.
- Consequences: Edits cost milliseconds (manifest only). Disk use is about 1 original + 1 float32 working copy per source. Long previews are capped.
- Risks: Float32 working copies are large (about 1 GB/hour stereo at 48 kHz); a cleanup/rebuild policy is a later concern.
- Alternatives considered: Decode on the fly from originals (slow scrubbing, lossy-codec seeking); in-memory audio in the API process (violates worker isolation).

## ADR-0010 — Audio capture with PyAudioWPatch (WASAPI) in an isolated worker
- Status: Accepted
- Date: 2026-10-28
- Context: Recording needs microphones and system audio (WASAPI loopback) on Windows. The common `sounddevice` wheel does not expose loopback devices.
- Decision:
  - Use **PyAudioWPatch** (PortAudio with WASAPI loopback). Device enumeration runs in a worker and opens no stream; capture runs only after an explicit Start in a worker `record_audio` job.
  - The capture callback only enqueues blocks; a writer streams float32 WAV into `cache/recording/`. The worker reports peak/RMS/hold/clip telemetry every 80 ms through job progress data.
  - **Stop** (finish and keep) and **cancel** (discard) are separate signals (two named events). On success the engine chains an `import_audio` job (origin `recording`) so the recording becomes an ordinary immutable source.
  - For loopback, a silent WASAPI render stream on the mirrored endpoint is kept open so that Windows delivers continuous packets and pauses stay on the timeline. The render device must be the **WASAPI** entry, not the same-named MME entry.
  - Rates/channels are the device's native format; conversion happens at import.
- Consequences: A microphone blocked by Windows privacy settings produces a clear error. Recordings are never uploaded.
- Risks: Exotic devices (exclusive-mode only, >2 channels) are untested. Loopback only captures what Windows renders to that endpoint.
- Alternatives considered: sounddevice (no loopback), `soundcard` (loopback but heavier and less controllable), a browser `getUserMedia` recorder (cannot capture system audio, and keeps audio in the browser).

## Backlog (decisions not yet made)
- Desktop packaging: Tauri vs Electron vs browser-only.
- Specific models per feature (separation, transcription, voice conversion, generation) and their licenses.
- Real-time playback engine location (engine-side audio output vs Web Audio), required before Phase 8.
- Persistent per-model workers for interactive latency.
- Plugin (VST3) hosting.
