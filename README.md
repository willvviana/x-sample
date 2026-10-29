# X-Sample

Local-first AI music sample editor for Windows, designed for AMD GPUs via DirectML. Version **0.1.0** — status: **Phase 1** (projects, audio import, recording, waveform editing, sample export) on top of the Phase 0 foundation. No AI audio models are included yet. See `CLAUDE.md`, `docs/ROADMAP.md` and `docs/DECISIONS.md`.

## What works today

- **Projects:** create, open, save, close and reopen; edits persist. Originals are stored read-only and never modified.
- **Import:** WAV, MP3, FLAC, OGG and M4A (also Opus/AAC/AIFF), resampled to the project rate, optional mono/stereo conversion.
- **Timeline:** waveforms, zoom, scroll, click to move the playhead, drag to select, keyboard control, multiple tracks (mute, solo, gain).
- **Editing (non-destructive):** trim to selection, slice into named samples, reset trim, remove clips.
- **Playback:** the engine renders previews (up to 120 s) and the browser plays them.
- **Export:** samples or selections as WAV (16/24-bit, float) or FLAC, downloadable.
- **Recording:** microphone or system audio (WASAPI loopback) with level meters, clipping warning, stop-and-save or discard. Recording happens only after you press Start and is saved into the open project; nothing leaves your computer.
- **System tab:** Phase 0 dashboard (backends, DirectML probes, smoke jobs).

## Requirements

- Windows 10/11, DirectX 12 GPU (reference: AMD Radeon RX 5700 XT)
- [uv](https://docs.astral.sh/uv/) (provides Python 3.12), Node.js 22+, FFmpeg and ffprobe on `PATH`

## Setup

```powershell
cd engine; uv sync; cd ..
cd ui; npm ci; npm run build; cd ..
```

## Run

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

The engine listens on `http://127.0.0.1:8765` only and prints a launch URL containing a one-time token (`#token=…`). The browser opens it automatically; the token changes on every start. Projects are stored in `Documents\X-Sample Projects` (override with `--projects-dir` or `XSAMPLE_PROJECTS_DIR`).

Development (Vite hot reload + engine): `powershell -ExecutionPolicy Bypass -File scripts\dev.ps1`

## Test

```powershell
cd engine
uv run pytest            # unit, integration, API, security, worker isolation, long audio
uv run pytest -m gpu     # DirectML verification on the GPU
uv run pytest -m hardware  # opt-in: records from the real microphone and loopback (see below)
uv run ruff check . ; uv run ruff format --check . ; uv run mypy
cd ..\ui
npm run check            # typecheck, lint, format
npx playwright install chromium   # first time only
npm run test:e2e
```

`-m hardware` opens your default microphone for about 3 seconds and plays a short, quiet 440 Hz tone on your default output so the loopback has something to capture. Recorded audio is deleted afterwards; only measurements are written to `engine/test-reports/hardware_recording.json`.

No other test downloads models, contacts the network or touches audio devices.

## Limitations (Phase 1)

See the Phase 1 section of `docs/ROADMAP.md`: previews instead of a live mixer, no drag-to-move clips or undo/redo yet, one project open at a time.
