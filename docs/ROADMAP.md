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

## V1.0.0 milestones

V1 is complete when a user can, entirely on this machine: import a sample (1); see its genres, instruments, tempo, rhythm, key, chords and structure (2); separate vocals and instruments where supported (3); add complementary instruments (4); replace or remove instruments (5); transform it into another genre (6); replace or generate sung vocals (7); rewrite lyrics under word/rhyme/syllable/meaning constraints (8); align singing to melody, rhythm and phrasing (9); keep or deliberately change pulse, groove, harmony and dynamics (10); edit non-destructively on multiple tracks (11); preview, compare and export the result (12); and repair or extend samples that start or end unnaturally with fades, smoothing or generated intros/outros (13). Numbers in brackets refer to these requirements.

Milestones are built one at a time; each needs passing acceptance tests before the next starts. Model choices are candidates until an ADR is accepted and downloads are approved.

### Requirement 13 — Intelligent intro and outro
Imported audio often starts or stops mid-phrase. X-Sample must detect this and offer repairs, from simple boundary fixes to generated introductions and endings. It is delivered across milestones (see the [13] items below), not as one release.
- **Detect:** abrupt starts and stops, clicks/DC steps at the boundaries, truncated notes (energy still sustaining at the cut), incomplete phrases (cut away from a bar/phrase boundary), and missing resolution (final chord or melody note off the tonic/expected cadence). Findings carry a confidence and the evidence used; nothing is reported as a problem without a measured cause.
- **Simple repairs:** automatic fade-in/out, crossfade and transient smoothing for simple boundary problems, sample-accurate and non-destructive.
- **Generated intros/outros:** AI-generated material conditioned on the original sample. Intros establish the musical context before the sample begins; endings complete melodic phrases, resolve the harmony and let instruments decay naturally. Genre, BPM, beat alignment, key, chords, instrumentation, timbre, groove, dynamics and phrasing are preserved.
- **Controls:** adjustable duration (in bars), style, instrumentation and transition type; several alternatives per request; before/after preview; every result is a separate clip that can be removed.
- **Quality bar:** no clicks, clipping, sudden level changes, timing drift or audible seams at the join (measured, not judged by ear only).
- **Scope:** R&B, piano and violin arrangements are priority test cases; other genres and instruments are not restricted.
- **Honesty:** AI music continuation is not claimed as implemented until real local inference has been demonstrated and evaluated on the reference machine. Until then the UI offers only the DSP repairs.

### v0.2.0 — Musical intelligence (analysis) [2, 10, 13] — in progress
- Tempo (with changes), beats, downbeats, meter where reliable, subdivisions, onsets, swing/groove, sections; half/double-time candidates kept; manual beat-grid correction.
- Multi-label genre and instrument tags with confidences, per-section/per-window activity, editable tags; uncertain results shown as uncertain.
- Key (global and local), mode, chords, harmonic rhythm, predominant melody contour; correction of key, chords and tempo.
- Vocal presence and vocal activity regions (no lyric transcription claims).
- Versioned analysis sidecars (model/version, parameters, source hash, backend, timings, confidences), cached and invalidated by audio change; user corrections stored separately and preserved on re-run.
- Analysis workspace: tags, BPM/meter, beat grid on the waveform, key/chord lane, sections, vocal activity, confidence indicators, correction controls.
- Boundary information for [13], recorded in the analysis sidecar with no repair actions yet: onset/transient and level at the first and last samples, whether a note is still sustaining at the cut, the position of the start/end relative to the beat/bar grid and section boundaries, and the first/last chord against the detected key.
- **Status (2026-10-29):** built and tested — DSP analyzers (tempo/beats/downbeats/meter, groove, onsets, key and local keys, chords, harmonic rhythm, bass line, melody notes with articulation/vibrato, sections, loudness), the `analyze_audio` worker job, versioned sidecar cache, manifest corrections (format 2) with an engine-computed effective view, API, Analysis panel with beat grid on the waveform, boundary measurements (ADR-0011). Not done: genre/instrument/vocal recognition (needs approved models; shown as unavailable), DirectML parity (no model yet). Real-audio accuracy is below the targets below (`docs/EVALUATION.md`), so v0.2.0 is not complete.
- **Acceptance:** on the labelled evaluation set — tempo Acc2 ≥ 95 % and beat F-measure ≥ 0.85 on clean material; key accuracy ≥ 80 % (MIREX-weighted) and chord majmin frame accuracy ≥ 70 %; instrument and genre precision/recall reported per class with a documented threshold; 10-minute analysis with bounded memory, cancellable, crash-safe; DirectML results match CPU within tolerance and are labelled with the backend that actually ran; all v0.1.0 tests pass.

### v0.3.0 — Source separation [3, 5]
- Vocals/drums/bass/other (Demucs/HTDemucs- or MDX-class, ONNX on DirectML with CPU fallback), chunked with overlap, chunk size from measured timings within the 6.5 GiB VRAM budget.
- Stems become linked sources/tracks with provenance; "remove instrument" = mute/delete a stem; residual mix shown honestly.
- **Acceptance:** separate a 10-minute track on the RX 5700 XT within budget; SDR measured on a licensed reference set; stems re-sum to the mix within a documented error.

### v0.4.0 — Multitrack editing and mixing foundation [11, 10, 12, 13]
- Drag/move/split clips, fades/crossfades, undo/redo as operations, tempo-aware grid snapping from v0.2.0 analysis.
- Per-track gain, pan, EQ, dynamics; loudness metering (LUFS), true-peak limiting/clipping prevention; masking indicators.
- Decide and implement the real-time playback location (engine-side vs Web Audio) — ADR required.
- [13] Boundary problem detection from the v0.2.0 boundary data, plus one-click fixes: fade-in/out (selectable curves), crossfades at joins, click/DC-step and transient smoothing, with before/after preview.
- **Acceptance:** arrange ≥ 8 tracks with sample-accurate edits, undo/redo across save/reopen, offline render matches real-time playback. [13] On generated test signals with known defects, abrupt starts/stops, boundary clicks and truncated notes are detected with reported precision/recall; after automatic repair, no sample-to-sample step above a documented threshold, no clipping, and no level jump above a documented dB limit at the boundary.

### v0.5.0 — Instrument generation and replacement [4, 5, 10, 13]
- Generate parts conditioned on tempo, beat grid, key, chords, sections and a user-chosen style; MIDI-first path rendered with local instruments/soundfonts; audio-model path only after probes pass on DirectML.
- Replace an instrument: separate (v0.3.0) → generate a substitute in time and key → mix.
- [13] Evaluate local audio continuation/inpainting models (license, size, RAM/VRAM, DirectML/CPU compatibility, real inference on the reference machine) before any is adopted. Then implement generated intros and endings conditioned on the sample, with duration, style, instrumentation and transition controls, several alternatives, and inserted as separate clips. A MIDI-first path (the generation pipeline above, conditioned on the sample's grid, key and chords) is acceptable until an audio model passes evaluation.
- **Acceptance:** generated parts are on-beat (onset deviation measured against the grid) and in key/chord (pitch-class agreement measured); user can regenerate, edit and A/B. [13] Generated intros/outros: the beat grid continues across the join without drift (measured), detected key/chords stay compatible with the sample, outros end on a resolved chord with a natural decay, and the join has no click, clipping or level jump; tested on R&B, piano and violin material and at least one other genre.

### v0.6.0 — Genre transformation [6, 10]
- Re-arrange into a target genre: keep selected stems, regenerate others; preserve timing and harmony unless the user changes tempo/key/groove explicitly.
- **Acceptance:** transformed output keeps the beat grid and chord sequence within measured tolerances; target-genre tag confidence rises on the v0.2.0 tagger.

### v0.7.0 — Lyrics and singing [7, 8, 9]
- Local transcription where a model proves compatible (Whisper-class, e.g. whisper.cpp with a Vulkan/CPU runner), word/syllable/phoneme alignment, editable lyrics.
- Lyric rewriting under constraints (kept words, rhyme scheme, syllable counts, meaning), local language model only after evaluation and approval.
- Singing synthesis / voice conversion with pitch, note duration, breathing/phrasing and voice controls; target-voice rights/consent confirmation required.
- **Acceptance:** generated vocal phrases land on the intended notes and syllable timings (measured pitch and onset error); rewritten lyrics satisfy the stated constraints (checked automatically); consent flow enforced.

### v0.8.0 — Fidelity, comparison and export [12, 10, 13]
- Per-operation fidelity controls (strength, blend, wet/dry), A/B and null-test views, spectral difference, version comparison.
- Export: stems and mix, sample rate, bit depth, dither, LUFS targets, metadata.
- **Acceptance:** every AI operation exposes fidelity controls and A/B; exports verified by ffprobe and loudness measurement. [13] End-to-end tests: export a sample with a repaired or generated intro/outro and verify timing, harmonic continuity, transition quality and export integrity.

### v0.9.0 — Beta hardening
- Packaging decision (ADR), installer, first-run model manager with license display and approval, performance and memory profiling on the reference machine, crash reporting kept local, documentation.
- **Acceptance:** fresh-machine install and an end-to-end scripted workflow covering requirements 1–13.

### v1.0.0 — Release
- All milestone acceptance tests green on the reference machine; known limitations documented; no network use without per-action approval.

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
