# X-Sample Roadmap

Each phase ends with working, tested functionality before the next begins. Model choices are candidates only; each needs an ADR and approval before download.

## Phase 0 — Foundation ✅ released as part of v0.1.0
- Engine: Python 3.12 + uv lockfile, FastAPI on 127.0.0.1 with per-launch token, Host/Origin validation, CORS allow-list, CSP for the served UI (ADR-0002, ADR-0008).
- Job queue with progress, cooperative + forced cancellation, crash/timeout handling, isolated worker processes in Windows Job Objects with memory limits (ADR-0005, ADR-0007).
- Runtime: CPU and DirectML adapters, Vulkan detection (no model support claimed), backend probes with cached results keyed by model/backend/device/driver/runtime (ADR-0003).
- Model registry with SHA-256 verification; downloads disabled.
- UI: React/TS/Vite dashboard — system status, backends, probes, smoke jobs with progress/cancel.
- Tests: pytest (unit, integration, security, worker crash/cancel/limits, 10-minute chunked runs), DirectML GPU verification (strict, profiled, adapter-verified, numerically checked), Playwright e2e, network isolation.
- **Not in Phase 0 (moved to Phase 1):** project create/open/save, audio import, recording, waveform peaks.

## Phase 1 — Project format, audio capture and import ✅ released as v0.1.0
- **Done and tested:** create/open/save/close/reopen projects (ADR-0006, ADR-0009); import WAV/MP3/FLAC/OGG/M4A (resample, mono/stereo conversion, chunked, originals preserved); waveform peak pyramids with zoom/scroll in a canvas timeline; click-to-seek, drag-select, keyboard control; non-destructive trim, slice-to-sample, reset trim, remove clip; multiple tracks (mute/solo/gain/rename/delete); preview playback (Web Audio); export of samples and selections (WAV/FLAC) with download; microphone and WASAPI loopback recording with meters, clipping warning, stop/discard and automatic import (ADR-0010); verify/rebuild of project files; crash/cancel cleanup.
- **Verified on the reference machine:** microphone and loopback captures were recorded, saved, imported, reopened and rendered (`engine/test-reports/hardware_recording.json`, run with `uv run pytest -m hardware`). 10-minute stereo WAV/MP3 imports and a full export keep worker memory flat (`long_audio.json`).
- **Known limitations:** playback is rendered previews (max 120 s) rather than a real-time mixer; clips cannot yet be dragged on the timeline (the API supports `move`); no undo/redo yet (edits are logged in `history`); time-stretch/non-unity time maps rejected; only the default sample-rate choices; one open project per engine; drag-and-drop import not implemented (file picker only); devices needing exclusive mode or more than two channels are untested.
- **Exit met:** create/open/save a project; record and import 10-minute files with bounded memory; view waveforms; slice and export samples.

## Phase 2 — AI analysis
- Tempo, beat grid, downbeats, key/scale, chords, onsets, loudness (LUFS) — DSP-first, ML where it improves accuracy.
- Transcription and word-level lyric alignment (Whisper-class model; candidates: whisper.cpp with Vulkan runner, or ONNX on DirectML/CPU — decided by probes).
- Auto-tagging (instrument, genre, mood) for sample browsing.
- **Exit:** analysis panel with results stored as versioned sidecars; results editable by the user.

## Phase 3 — Source separation
- Stem separation: vocals, drums, bass, other (Demucs/HTDemucs-class or MDX-class via ONNX + DirectML, CPU fallback).
- Chunked inference with overlap; chunk size chosen from measured timings and VRAM budget.
- Quality presets (fast/balanced/best) mapped to model size, overlap and shifts.
- **Exit:** separate a 10-minute track on the RX 5700 XT within budget; stems appear as tracks.

## Phase 4 — Vocal and lyrics replacement
- Vocal conversion (RVC/SVC-class) with consent confirmation; PyTorch-only models get per-model adapters only if probes pass.
- Lyric replacement: align original lyrics, edit text, re-synthesize singing preserving melody and timing; word timings follow edits via the time map.
- Pitch editing and timing correction on isolated vocals.
- **Exit:** replace a phrase in a separated vocal and re-mix it convincingly.

## Phase 5 — Genre transformation
- Style/genre transfer conditioned on analysed tempo, key and structure.
- Re-arrangement: keep selected stems, regenerate others in a target genre.
- Timbre transfer for individual stems.
- **Exit:** transform a loop into a different genre while keeping key and tempo.

## Phase 6 — Instrument generation
- Generate instrument parts from prompts plus musical context (key, tempo, chords).
- MIDI-first option rendered with local instruments/soundfonts.
- Continuation and infill for regions.
- **Exit:** generate a part that is in time and in key with the project.

## Phase 7 — Fidelity control
- Per-operation fidelity controls (guidance, strength, blend); wet/dry, A/B, null-test and spectral-difference views.
- Artifact reduction; output format control (sample rate, bit depth, dither, LUFS target).
- **Exit:** every AI operation exposes fidelity controls and an A/B view.

## Phase 8 — Multitrack editing
- Timeline with tracks, clips, fades, crossfades, mute/solo, gain, pan; undo/redo as operations.
- Effects chain and bus routing; render/bounce to stems or mix.
- Decide playback location (engine-side vs Web Audio) before starting (ADR backlog).
- **Exit:** arrange separated, generated and recorded material into a mixed export.

## Cross-cutting (every phase)
- Every model: CPU path + probe-gated GPU path; strict DirectML tests with profiling.
- Privacy: no network calls without explicit approval; model downloads gated and hash-verified.
- Performance and memory budgets recorded per feature on the reference machine.
- Accessibility and keyboard-first UI; Playwright e2e coverage of each exit criterion.

## Later / optional
- Desktop packaging (Tauri or Electron).
- Persistent per-model workers for interactive latency.
- Plugin hosting (VST3); DAW export (stems + MIDI).
- Opt-in cloud generation (e.g. Background Music Generator MCP), clearly separated from local features.
