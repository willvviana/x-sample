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

## ADR-0011 — Musical analysis: DSP analyzers, versioned sidecars, corrections in the manifest
- Status: Accepted
- Date: 2026-10-29
- Context: v0.2.0 needs tempo, beats, meter, groove, key, chords, melody and sections that later features (editing grid, generation, intro/outro repair) build on. Recognition models need approval and evaluation first, and results must never be invented.
- Decision:
  - **Analyzers:** NumPy-only DSP in `xsample.analysis` at 22.05 kHz mono with three streaming framers (rhythm 2048/256, pitch 4096/256, chroma 8192/2048). Audio is decoded by FFmpeg in bounded chunks, so memory follows the number of frames, not the samples (10 min: about 216 MiB peak worker memory). The analyzers run in an `analyze_audio` worker job (`public=False`, CPU, cancellable between chunks and stages). Results are labelled `backend: "cpu"`; nothing claims GPU use.
  - Methods: spectral-flux onsets; comb-weighted autocorrelation tempo with a log-normal prior and half/double candidates; dynamic-programming beat tracking with a time-varying period; off-beat phase check from low-band energy; beat-grid completion at the edges; meter/downbeats from low-band accent and chroma change; Temperley-style key profiles (natural-minor-aware) plus tonic evidence from bass and chord roots, and a phrase-start bass tie-break between relative keys; HMM local keys; beat-synchronous chord templates (maj, min, 7, maj7, min7, no-chord) with median flooring and Viterbi smoothing; harmonic-summation melody with whitening, fundamental gating and Viterbi tracking; Foote novelty sections.
  - **Results** are a versioned sidecar `analysis/results/<source_id>/<cache_key>.json` (`schema_version`, `analyzer_id`, `analyzer_version`, parameters, backend, timings, created time). `cache_key = sha256(media PCM SHA-256, analyzer version, schema version, parameters)`. A different working copy or analyzer version gives a different key, so stale results are never shown as current. A successful run deletes older sidecars of that source. Positions are integer samples at the project rate, relative to the source.
  - **Corrections** (beat grid: half/double/fixed BPM, offset, meter, downbeat shift; key; chord spans; genre/instrument tags) live in the manifest under `analysis[source_id]` and change only through `ProjectService.edit`. The API returns detected results, corrections and an *effective* view computed by the engine. Re-running analysis never touches corrections.
  - Manifest **format 2** adds `analysis`. Format-1 manifests load unchanged and are written back as format 2. v0.1.0 refuses format 2 with its existing "newer X-Sample" message.
  - Genre, instrument and vocal recognition return an explicit `unavailable` result with the reason until a model is approved, evaluated and adopted (a later ADR). Users can add tags manually.
  - Boundary measurements for intro/outro repair (requirement 13) are recorded as raw values only: edge sample, edge level, beat offset, position in the bar, chord at the edge and whether it is the tonic.
- Consequences: Analysis is fast (10 min of audio in about 13 s on the reference CPU) and needs no download. Accuracy is good on clean synthetic material, but much lower on real recordings (see `docs/EVALUATION.md`). The v0.2.0 real-audio acceptance targets are not met by DSP alone.
- Risks: Tempo octave and 4:3 errors on dense or drumless material; beat phase on solo instruments; seventh chords under a lead line; melody below C3 is not searched. Scores are not calibrated probabilities and the UI says so.
- Alternatives considered: librosa/madmom/Essentia (extra heavy dependencies and, for Essentia models, non-commercial licenses); learned beat/key/chord models (planned through ADR-0012 after approval and evaluation).

## ADR-0012 — Beat and downbeat tracking with Beat This! small (ONNX, DSP fallback)
- Status: Accepted (subject to approval of the v0.2.0 checkpoint)
- Date: 2026-10-29
- Context: DSP beat tracking misses the v0.2.0 targets on real audio (4:3 tempo locks on funk comping, off-beat phase on solo instruments, expressive classical tempo). Beat This! (CPJKU, ISMIR 2024) is a small transformer for beats and downbeats without a DBN; code and published weights are MIT. Evaluation of the "small" checkpoint was approved.
- Decision:
  - **Model:** `small0` (2.1 M parameters, 8.45 MB checkpoint, SHA-256 `6074be2c…d47b`, downloaded once with approval from the authors' JKU cloud share listed in the beat_this README). Its training data includes GuitarSet and 15 other datasets (GTZAN held out); some training audio is copyrighted or under limited CC licenses, which the license note shown to users states.
  - **Export:** `engine/tools/export_beat_this.py` (run in an isolated temporary venv with torch 2.15 CPU and beat_this at commit `b95c8ab`; torch is not an engine dependency) writes two ONNX files: a fixed `(1, 1500, 128)` export for CPU/DirectML and a dynamic-length export for pieces shorter than one chunk. DirectML cannot run the dynamic export (a Reshape fails); zero-padding short pieces to 1500 frames changes predictions (F 0.68 vs upstream on a 14 s clip), so short pieces use the dynamic export on CPU, as upstream does. Two export-only rewrites (autocast-free SumHead sum; rotary embedding without autocast decorators and in-place cache) give bit-identical outputs. The export is deterministic; the engine pins both files' SHA-256 and size (`beat_model.PINNED`) as the trust root.
  - **Install:** `engine/tools/install_beat_this.py` copies verified files to `%LOCALAPPDATA%\X-Sample\models\beat-this-small0-{t1500,dynamic}\` and mirrors them in `registry.json`. Nothing is downloaded by the engine.
  - **Pre/postprocessing in NumPy** (`analysis/beat_model.py`): streaming log-mel equal to torchaudio's `MelSpectrogram` (22.05 kHz, n_fft 1024, hop 441, 128 Slaney mels, 30–11000 Hz, `frame_length` normalisation = 1/√n_fft, `log1p(1000·x)`), upstream chunking (1500 frames, border 6, keep-first) and the "minimal" peak-picking postprocessor. End-to-end the pipeline reproduces upstream beat_this on real recordings (beat/downbeat F 1.000 vs torch, logits within 1e-4).
  - **Backends:** inference only in the `analyze_audio` worker via `runtime/adapters.py`. The engine requests DirectML when detection found it and the clip is ≥ 30 s; the worker then requires a strict session (CPU fallback disabled), profiles the first chunk (every node on DmlExecutionProvider) and compares it with CPU (tolerance 1e-2; measured ~1e-4). Any failure, including a DirectML runtime error mid-piece, reruns the whole piece on CPU and records the reason. Results report the backend that actually ran.
  - **Use in analysis:** model beats and downbeats replace the DSP grid unedited; tempo, tempo curve, meter, groove and syncopation are derived from them; onsets, harmony, melody and structure remain DSP. If the model is not installed the DSP tracker runs; if it is installed but cannot load, the job falls back to DSP and says why.
  - **Cache:** sidecar schema 2 records `beat_tracker` (method, model id/name, both SHA-256, license, backend, notes) and `timings.beat_model_s`. The cache key includes the model identity (both pins), so installing/removing the model makes earlier results stale; it excludes the backend because DirectML output is parity-checked against CPU.
  - **Resources (RX 5700 XT / Ryzen 7 3700X):** 30 s chunk 115 ms DirectML vs 720 ms CPU; 10 min of audio 4.1 s vs 15.1 s; ~570 MiB VRAM; worker peak commit 1.06 GiB (CPU) / 1.64 GiB (DirectML, includes the CPU reference session for the parity check), flat with duration. Job estimates: 768 MiB VRAM, 1.3/1.9 GiB RAM.
- Consequences: Better beats/downbeats on held-out real audio (MusicNet beat F 0.48 → 0.62) at a small, fixed cost; analysis gains an optional GPU path that is verified per run. Results on GuitarSet are not a held-out measurement.
- Risks: The minimal postprocessor has no tempo continuity, so metrical-level errors remain (MusicNet solo piano, Slakh track 20); training-data licensing must be reviewed by the user for commercial use; the 1500-frame attention makes CPU memory ~0.8 GiB.
- Alternatives considered: `final0` (78 MB, more accurate in the paper; not approved); madmom DBN postprocessing (non-commercial model license, extra dependency); `fold*` checkpoints for a fair GuitarSet evaluation (78 MB each, not approved); keeping DSP only.

## ADR-0013 — Instrument, vocal and genre tagging with PANNs MobileNetV1 (ONNX)
- Status: Accepted (subject to approval of the v0.2.0 checkpoint)
- Date: 2026-10-29
- Context: v0.2.0 requires multi-label instrument and genre tags with confidences, per-window activity, vocal presence, editable tags and uncertain results shown as uncertain. Candidates and licenses are in `docs/MODELS.md`; PANNs MobileNetV1 was approved (weights CC BY 4.0, code MIT, 23.6 MB). Cnn14, FMA and other models/datasets were not approved.
- Decision:
  - **Model:** `MobileNetV1_mAP=0.389.pth` from Zenodo record 3987831 (CC BY 4.0, Kong et al. 2020), 23,639,473 bytes, MD5 as published by Zenodo, SHA-256 `5f1d0943…271f`. AudioSet ontology, 527 classes, 32 kHz.
  - **Export:** `engine/tools/export_panns.py` in an isolated temporary venv (torch 2.15 CPU, torchlibrosa 0.1.0 hash-pinned, librosa 0.11, model code from `audioset_tagging_cnn` commit `d2f4b8c`, inspected before running). The graph starts after the STFT/log-mel layers and ends with the sigmoid; it is MobileNetV1.forward in eval mode, checked against the complete torch model. numba is replaced by a no-op module during export (Windows Application Control blocks its DLLs; the STFT and mel weights come from the checkpoint). One dynamic-length ONNX file, deterministic, pinned in `tagger.PINNED` (SHA-256 `0dfb5d1b…7544`, 19,383,221 bytes) as the trust root; `tools/install_panns.py` installs it from a verified local file. Torch is not an engine dependency.
  - **Front end in NumPy** (`analysis/tagger.py`): streaming 32 kHz log-mel equal to torchlibrosa (Hann 1024, hop 320, centre/reflect, power, librosa Slaney mel 64 bands 50–14000 Hz, 10·log10 with amin 1e-10). The analysis job decodes the source a second time at 32 kHz (the beat/DSP pass is 22.05 kHz and would lose 11–14 kHz).
  - **Windows and scores:** 10 s windows (the AudioSet clip length) with a 5 s hop, last window aligned to the end; pieces under 10 s are one window, under 1 s are not tagged (unavailable). Each X-Sample label aggregates AudioSet classes by max (e.g. Piano = Piano + Electric piano; Singing = Singing, Choir, Male/Female/Child singing). Piece score = highest window score; activity = merged windows ≥ the uncertain threshold. Thresholds fixed before evaluation: detected ≥ 0.5, uncertain ≥ 0.2. Scores are sigmoid outputs, not calibrated probabilities, and the UI says so.
  - **Honesty rules:** each tag carries `evaluated` (precision/recall measured on labelled held-out audio with ≥ 5 pieces each way). Genres may be reported as detected only when the evidence rule holds (≥ 10 pieces each way, precision ≥ 0.8, recall ≥ 0.5 at 0.5); none does, so genres are at most uncertain. Unavailable states remain for missing model, load failure and too-short audio.
  - **Backends:** worker only, via `runtime/adapters.py`. DirectML is requested when detection found it; the worker requires a strict session, profiles the first window (all nodes on DmlExecutionProvider) and compares with CPU (tolerance 1e-3; measured 3e-7); any failure reruns the piece on CPU and records why. The sidecar reports the tagger's backend separately from the beat model's.
  - **Cache and corrections:** sidecar schema 3 (`tagger`, `genres/instruments/vocals` as `TagsOut | Unavailable`, `timings.tagger_s`). The cache key includes the tagger identity (pinned file + label mapping, thresholds, windows and evaluated/evidence sets), so changing any of them makes results stale. Manifest format 3: tag corrections may be `vocal` and may reject a detected tag (`present: false`); the effective view merges user tags, detected and uncertain tags, minus rejected ones.
  - **Resources:** ~48 MiB VRAM; 10 min of audio: inference 0.38 s DirectML / 1.0 s CPU, tagging step 1.6 s / 2.1 s; worker peak unchanged with the beat model (models run in turn). Job estimates: +300 MiB RAM, 128 MiB VRAM (not additive to the beat model's).
- Consequences: Piano, violin and guitar are recognised on real acoustic recordings with high precision; other instruments, singing and genres are shown with explicit uncertainty or "not evaluated" until labelled data exists. Attribution for CC BY 4.0 is shown with every result (license string) and in `docs/MODELS.md`.
- Risks: Low recall on synthesised multi-instrument audio (BabySlakh); no ground truth for singing or R&B; AudioSet training audio comes from YouTube (the same kind of caveat as Beat This!); 10 s windows give coarse activity regions (5 s resolution).
- Alternatives considered: PANNs Cnn14 (327 MB, more accurate; not approved), YAMNet (Apache-2.0, lower mAP, TF conversion), EfficientAT (weights license unresolved), Essentia/MTG (non-commercial), CLAP (too large).

## Backlog (decisions not yet made)
- Desktop packaging: Tauri vs Electron vs browser-only.
- Specific models per feature (separation, transcription, voice conversion, generation) and their licenses.
- Real-time playback engine location (engine-side audio output vs Web Audio), required before Phase 8.
- Persistent per-model workers for interactive latency.
- Plugin (VST3) hosting.
