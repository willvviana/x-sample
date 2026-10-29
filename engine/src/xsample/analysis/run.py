"""Runs the analyzers over one source's working copy and builds the versioned sidecar.

Runs inside a worker process. Audio is decoded and resampled by FFmpeg in bounded chunks, so
memory does not grow with the raw audio length (features are kept at frame rate only). When
the Beat This! model is installed, beats and downbeats come from it (beat_model.py) and the
DSP tracker remains the fallback. When the PANNs tagger is installed, a second streamed pass
at 32 kHz tags instruments, vocals and genres (tagger.py); otherwise they are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from xsample.analysis import beat_model, tagger
from xsample.analysis.dsp import ANALYSIS_RATE, FeatureExtractor, Features
from xsample.analysis.harmony import PITCH_CLASSES, HarmonyResult, analyze_harmony
from xsample.analysis.rhythm import RhythmResult, analyze_rhythm, apply_tracked_beats
from xsample.analysis.schema import (
    ANALYZER_VERSION,
    SCHEMA_VERSION,
    AnalysisResult,
    BassNoteOut,
    BeatTrackerOut,
    BoundariesOut,
    BoundaryEdge,
    ChordOut,
    ContourPoint,
    HarmonyOut,
    KeyOut,
    KeyRegion,
    LevelPoint,
    MelodyNoteOut,
    RhythmOut,
    SectionOut,
    StructureOut,
    TaggerOut,
    TagOut,
    TagSegment,
    TagsOut,
    TempoCandidateOut,
    TempoPoint,
    Timings,
    Unavailable,
)
from xsample.analysis.structure import StructureResult, analyze_structure

PARAMS: dict[str, float | int | str] = {"analysis_rate": ANALYSIS_RATE}
CHUNK_FRAMES = 65536
MODEL_PENDING = (
    "No tagging model is installed (PANNs MobileNetV1, docs/MODELS.md); "
    "add tags manually meanwhile."
)
TOO_SHORT = "Too short to tag (under 1 s); add tags manually."

TagFields = tuple[TagsOut | Unavailable, TagsOut | Unavailable, TagsOut | Unavailable]


class AnalysisCancelledError(Exception):
    pass


def cache_key(
    media_sha256: str, beat_model_id: str | None = None, tagger_id: str | None = None
) -> str:
    """Key of the sidecar for this audio, analyzer version and models.

    Model identities are part of the key (installing or removing a model makes older
    results stale). The backend is not: DirectML results are parity-checked against CPU.
    """
    material: list[Any] = [media_sha256, ANALYZER_VERSION, SCHEMA_VERSION, PARAMS]
    if beat_model_id is not None:
        material.append({"beat_model": beat_model_id})
    if tagger_id is not None:
        material.append({"tagger": tagger_id})
    blob = json.dumps(material, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:20]


def extract_features(
    chunks: Any,
    total_frames: int,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    mel: beat_model.LogMelStream | None = None,
    share: float = 0.7,
) -> tuple[Features, beat_model.F32 | None]:
    """DSP features and, when `mel` is given, the beat model's log-mel spectrogram."""
    fx = FeatureExtractor()
    spect: list[beat_model.F32] = []
    done = 0
    for chunk in chunks:
        if cancelled():
            raise AnalysisCancelledError
        mono = np.ascontiguousarray(chunk[:, 0] if chunk.ndim == 2 else chunk, dtype=np.float32)
        fx.push(mono)
        if mel is not None:
            spect.append(mel.push(mono))
        done += len(mono)
        progress(min(share, share * done / max(total_frames, 1)), "Reading audio")
    if mel is None:
        return fx.finish(), None
    spect.append(mel.finish())
    return fx.finish(), np.concatenate(spect)


def _key_out(tonic: int, mode: str, score: float) -> KeyOut:
    return KeyOut(tonic=tonic, mode=mode, label=f"{PITCH_CLASSES[tonic]} {mode}", score=score)  # type: ignore[arg-type]


def build_result(
    features: Features,
    rhythm: RhythmResult,
    harmony: HarmonyResult,
    structure: StructureResult,
    *,
    source_id: str,
    media_sha256: str,
    sample_rate: int,
    frames: int,
    timings: Timings,
    beat_tracker: BeatTrackerOut | None = None,
    sidecar_key: str | None = None,
    tagger_out: TaggerOut | None = None,
    tags: TagFields | None = None,
) -> AnalysisResult:
    tracker = beat_tracker or BeatTrackerOut(method="dsp")
    genres, instruments, vocals = tags or (
        Unavailable(reason=MODEL_PENDING),
        Unavailable(reason=MODEL_PENDING),
        Unavailable(reason=MODEL_PENDING),
    )
    used_gpu = tracker.backend == "directml" or (
        tagger_out is not None and tagger_out.backend == "directml"
    )

    def pos(t: float) -> int:
        return int(min(max(round(t * sample_rate), 0), frames))

    key = harmony.key_candidates[0] if harmony.key_candidates else None
    chords = [
        ChordOut(
            start=pos(c.start),
            end=pos(c.end),
            root=c.root,
            quality=c.quality,  # type: ignore[arg-type]
            bass=c.bass,
            label=c.label,
            confidence=round(c.confidence, 3),
        )
        for c in harmony.chords
    ]
    sections = [
        SectionOut(
            start=pos(s.start),
            end=pos(s.end),
            label=s.label,
            confidence=s.confidence,
            loudness_db=s.loudness_db,
        )
        for s in structure.sections
    ]
    rhythm_out = RhythmOut(
        bpm=round(rhythm.bpm, 2),
        confidence=round(rhythm.confidence, 3),
        candidates=[
            TempoCandidateOut(bpm=round(c.bpm, 2), relation=c.relation, score=round(c.score, 3))  # type: ignore[arg-type]
            for c in rhythm.candidates
        ],
        tempo_curve=[TempoPoint(position=pos(t), bpm=round(b, 2)) for t, b in rhythm.tempo_curve],
        variable_tempo=rhythm.variable_tempo,
        tempo_changes=[
            TempoPoint(position=pos(t), bpm=round(b, 2)) for t, b in rhythm.tempo_changes
        ],
        beats=[pos(t) for t in rhythm.beats],
        beat_strength=[round(float(s), 3) for s in rhythm.beat_strength],
        downbeats=[pos(t) for t in rhythm.downbeats],
        meter=rhythm.meter,
        meter_confidence=round(rhythm.meter_confidence, 3),
        subdivision=rhythm.subdivision,  # type: ignore[arg-type]
        subdivision_confidence=round(rhythm.subdivision_confidence, 3),
        swing_ratio=None if rhythm.swing_ratio is None else round(rhythm.swing_ratio, 3),
        swing_confidence=round(rhythm.swing_confidence, 3),
        syncopation=None if rhythm.syncopation is None else round(rhythm.syncopation, 3),
        microtiming_ms=None if rhythm.microtiming_ms is None else round(rhythm.microtiming_ms, 2),
        onsets=[pos(t) for t in rhythm.onsets],
        onset_strength=[round(float(s), 3) for s in rhythm.onset_strength],
        notes=rhythm.notes,
    )
    harmony_out = HarmonyOut(
        key_candidates=[
            _key_out(k.tonic, k.mode, round(k.score, 3)) for k in harmony.key_candidates
        ],
        key_confidence=round(harmony.key_confidence, 3),
        local_keys=[
            KeyRegion(
                start=pos(r.start),
                end=pos(r.end),
                key=_key_out(r.key.tonic, r.key.mode, round(r.key.score, 3)),
            )
            for r in harmony.local_keys
        ],
        chords=chords,
        harmonic_rhythm_beats=harmony.harmonic_rhythm_beats,
        changes_per_bar=harmony.changes_per_bar,
        bass_notes=[
            BassNoteOut(start=pos(a), end=pos(b), pitch_class=p) for a, b, p in harmony.bass_notes
        ],
        melody_notes=[
            MelodyNoteOut(
                start=pos(n.start),
                end=pos(n.end),
                midi=round(n.midi, 2),
                articulation=n.articulation,  # type: ignore[arg-type]
                vibrato_hz=n.vibrato_hz,
                vibrato_cents=n.vibrato_cents,
            )
            for n in harmony.melody_notes
        ],
        melody_contour=[
            ContourPoint(position=pos(t), midi=None if m is None else round(m, 2))
            for t, m in harmony.melody_contour
        ],
        voiced_ratio=round(harmony.voiced_ratio, 3),
        notes=harmony.notes,
    )
    structure_out = StructureOut(
        sections=sections,
        loudness_curve=[LevelPoint(position=pos(t), db=d) for t, d in structure.loudness_curve],
        dynamic_range_db=None
        if structure.dynamic_range_db is None
        else round(structure.dynamic_range_db, 2),
    )
    boundaries = measure_boundaries(
        features, rhythm_out, chords, sections, key, sample_rate, frames
    )
    return AnalysisResult(
        schema_version=SCHEMA_VERSION,
        backend="directml" if used_gpu else "cpu",
        source_id=source_id,
        cache_key=sidecar_key or cache_key(media_sha256),
        media_sha256=media_sha256,
        sample_rate=sample_rate,
        analysis_rate=ANALYSIS_RATE,
        frames=frames,
        params=PARAMS,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        timings=timings,
        beat_tracker=tracker,
        tagger=tagger_out,
        rhythm=rhythm_out,
        harmony=harmony_out,
        structure=structure_out,
        boundaries=boundaries,
        genres=genres,
        instruments=instruments,
        vocals=vocals,
    )


def measure_boundaries(
    features: Features,
    rhythm: RhythmOut,
    chords: list[ChordOut],
    sections: list[SectionOut],
    key: Any,
    sample_rate: int,
    frames: int,
) -> BoundariesOut:
    edges = features.edges
    rms_db = 20 * np.log10(np.maximum(features.rms.astype(np.float64), 1e-6))
    audible = rms_db[rms_db > -70]
    median_db = float(np.median(audible)) if len(audible) else -120.0
    beats = np.asarray(rhythm.beats, dtype=np.float64)
    downs = np.asarray(rhythm.downbeats, dtype=np.float64)
    period = float(np.median(np.diff(beats))) if len(beats) > 1 else None

    def chord_at(p: int) -> ChordOut | None:
        return next(
            (c for c in chords if c.start <= p < c.end or (p == frames and c.end == frames)), None
        )

    def edge(which: str, p: int) -> BoundaryEdge:
        amp, level = edges[which]
        level_db = 20 * float(np.log10(max(level, 1e-6)))
        beat_off = from_down = None
        if period and len(beats):
            i = int(np.argmin(np.abs(beats - p)))
            # Measured against the grid extended past the last detected beat: -0.5..0.5 beats.
            phase = (p - beats[i]) / period
            beat_off = round(phase - round(phase), 3)
            if len(downs):
                if which == "start":
                    later = downs[downs >= p]
                    from_down = round((later[0] - p) / period, 3) if len(later) else None
                else:
                    earlier = downs[downs <= p]
                    from_down = round((p - earlier[-1]) / period, 3) if len(earlier) else None
        probe = p if which == "start" else max(p - 1, 0)
        ch = chord_at(probe)
        sec = next((s for s in sections if s.start <= probe < s.end), None)
        tonic = None
        if ch is not None and ch.root is not None and key is not None:
            minor = ch.quality in ("min", "min7")
            tonic = ch.root == key.tonic and minor == (key.mode == "minor")
        return BoundaryEdge(
            position=p,
            edge_amplitude=round(amp, 5),
            level_dbfs=round(level_db, 2),
            level_vs_median_db=round(level_db - median_db, 2),
            beat_offset=beat_off,
            beats_from_downbeat=from_down,
            section_label=sec.label if sec else None,
            chord=ch.label if ch else None,
            chord_is_tonic=tonic,
        )

    return BoundariesOut(start=edge("start", 0), end=edge("end", frames))


def analyze_file(
    media_path: Path,
    *,
    source_id: str,
    media_sha256: str,
    sample_rate: int,
    frames: int,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
    decode: Callable[..., Any] | None = None,
    beat_model_dir: Path | None = None,
    beat_backend: str = "cpu",
    tagger_dir: Path | None = None,
    tagger_backend: str = "cpu",
) -> AnalysisResult:
    """Analyse one file. With `beat_model_dir`, beats come from the installed model; with
    `tagger_dir`, instruments, vocals and genres come from the installed tagger.

    The sidecar key reflects the request (models installed or not); `beat_tracker` and
    `tagger` in the result say what actually ran, including fallbacks and load failures.
    """
    total_analysis_frames = int(frames * ANALYSIS_RATE / sample_rate)
    t0 = time.perf_counter()
    if decode is None:
        from xsample.audio.ffmpeg import decode_f32

        decode = decode_f32
    chunks = decode(
        media_path,
        out_channels=1,
        sample_rate=ANALYSIS_RATE,
        chunk_frames=CHUNK_FRAMES,
        cancelled=cancelled,
    )
    use_model = beat_model_dir is not None
    features, spect = extract_features(
        chunks,
        total_analysis_frames,
        progress,
        cancelled,
        mel=beat_model.LogMelStream() if use_model else None,
        share=0.6 if use_model else 0.7,
    )
    t1 = time.perf_counter()
    tracker_out = BeatTrackerOut(method="dsp")
    tracked: tuple[Any, Any] | None = None
    if beat_model_dir is not None and spect is not None:
        progress(0.6, "Beats and downbeats (model)")
        tracked, tracker_out = _track_beats(
            beat_model_dir, beat_backend, spect, progress, cancelled
        )
    t_model = time.perf_counter()
    progress(0.75, "Tempo and beats")
    rhythm = analyze_rhythm(features)
    if tracked is not None:
        rhythm = apply_tracked_beats(
            rhythm, features, tracked[0], tracked[1], beat_model.DISPLAY_NAME
        )
    if cancelled():
        raise AnalysisCancelledError
    t2 = time.perf_counter()
    progress(0.85, "Key, chords and melody")
    harmony = analyze_harmony(features, rhythm.beats, rhythm.downbeats, rhythm.onsets)
    if cancelled():
        raise AnalysisCancelledError
    t3 = time.perf_counter()
    progress(0.9, "Sections")
    structure = analyze_structure(features, rhythm.beats, rhythm.downbeats)
    t4 = time.perf_counter()
    tagger_out: TaggerOut | None = None
    tags: TagFields | None = None
    if tagger_dir is not None:
        progress(0.92, "Instruments, vocals and genres")
        tagger_out, tags = _tag(
            tagger_dir, tagger_backend, media_path, decode, sample_rate, frames, progress, cancelled
        )
    t5 = time.perf_counter()
    timings = Timings(
        decode_features_s=round(t1 - t0, 3),
        beat_model_s=round(t_model - t1, 3),
        tagger_s=round(t5 - t4, 3),
        rhythm_s=round(t2 - t_model, 3),
        harmony_s=round(t3 - t2, 3),
        structure_s=round(t4 - t3, 3),
        total_s=round(t5 - t0, 3),
    )
    return build_result(
        features,
        rhythm,
        harmony,
        structure,
        source_id=source_id,
        media_sha256=media_sha256,
        sample_rate=sample_rate,
        frames=frames,
        timings=timings,
        beat_tracker=tracker_out,
        sidecar_key=cache_key(
            media_sha256,
            beat_model.identity() if use_model else None,
            tagger.identity() if tagger_dir is not None else None,
        ),
        tagger_out=tagger_out,
        tags=tags,
    )


def _tag(
    models_dir: Path,
    backend: str,
    media_path: Path,
    decode: Callable[..., Any],
    sample_rate: int,
    frames: int,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> tuple[TaggerOut, TagFields]:
    def out(used: str, notes: list[str]) -> TaggerOut:
        return TaggerOut(
            model_id=tagger.MODEL_ID,
            model_name=tagger.DISPLAY_NAME,
            model_sha256=tagger.PINNED[0],
            license=tagger.LICENSE,
            attribution=tagger.ATTRIBUTION,
            backend=used,  # type: ignore[arg-type]
            window_s=tagger.WINDOW / tagger.FPS,
            hop_s=tagger.WINDOW_HOP / tagger.FPS,
            detect_threshold=tagger.DETECT_THRESHOLD,
            uncertain_threshold=tagger.UNCERTAIN_THRESHOLD,
            notes=notes,
        )

    def unavailable(reason: str) -> TagFields:
        return Unavailable(reason=reason), Unavailable(reason=reason), Unavailable(reason=reason)

    # The tagger needs 32 kHz (content up to 14 kHz), so it reads the audio a second time.
    mel = tagger.LogMelStream()
    parts = []
    total = max(1, int(frames * tagger.SAMPLE_RATE / sample_rate))
    done = 0
    for chunk in decode(
        media_path,
        out_channels=1,
        sample_rate=tagger.SAMPLE_RATE,
        chunk_frames=CHUNK_FRAMES,
        cancelled=cancelled,
    ):
        if cancelled():
            raise AnalysisCancelledError
        mono = np.ascontiguousarray(chunk[:, 0] if chunk.ndim == 2 else chunk, dtype=np.float32)
        parts.append(mel.push(mono))
        done += len(mono)
        progress(0.92 + 0.03 * min(1.0, done / total), "Instruments, vocals and genres")
    parts.append(mel.finish())
    logmel = np.concatenate(parts)
    if len(logmel) < tagger.MIN_FRAMES:
        return out("cpu", []), unavailable(TOO_SHORT)
    try:
        model = tagger.Tagger(models_dir, backend)
    except Exception as exc:
        # A broken install must not block analysis: report it, tags stay unavailable.
        reason = f"The tagging model could not be loaded ({type(exc).__name__}: {exc})."
        return out("cpu", [reason]), unavailable(reason)
    try:
        starts, scores, report = model.tag(
            logmel,
            progress=lambda f: progress(0.95 + 0.04 * f, "Instruments, vocals and genres"),
            cancelled=cancelled,
        )
    except tagger.TaggingCancelledError as exc:
        raise AnalysisCancelledError from exc
    finally:
        model.close()

    def pos(t: float) -> int:
        return int(min(max(round(t * sample_rate), 0), frames))

    by_kind: dict[str, list[TagOut]] = {"genre": [], "instrument": [], "vocal": []}
    for summary in tagger.summarise(starts, scores, len(logmel)):
        by_kind[summary.label.kind].append(
            TagOut(
                kind=summary.label.kind,
                label=summary.label.name,
                audioset_classes=list(summary.label.class_names),
                score=round(summary.score, 4),
                status=summary.status,
                evaluated=summary.evaluated,
                activity=[
                    TagSegment(start=pos(a), end=pos(z), score=round(sc, 4))
                    for a, z, sc in summary.activity
                ],
            )
        )
    fields: TagFields = (
        TagsOut(tags=by_kind["genre"]),
        TagsOut(tags=by_kind["instrument"]),
        TagsOut(tags=by_kind["vocal"]),
    )
    return out(report.backend, report.notes), fields


def _track_beats(
    models_dir: Path,
    backend: str,
    spect: beat_model.F32,
    progress: Callable[[float, str], None],
    cancelled: Callable[[], bool],
) -> tuple[tuple[Any, Any] | None, BeatTrackerOut]:
    try:
        tracker = beat_model.BeatTracker(models_dir, backend)
    except Exception as exc:
        # A broken install must not block analysis: report it and use the DSP tracker.
        note = f"The beat model could not be loaded ({type(exc).__name__}: {exc}); DSP was used."
        return None, BeatTrackerOut(method="dsp", notes=[note])
    try:
        beats, downs, report = tracker.track(
            spect,
            progress=lambda f: progress(0.6 + 0.15 * f, "Beats and downbeats (model)"),
            cancelled=cancelled,
        )
    except beat_model.BeatTrackingCancelledError as exc:
        raise AnalysisCancelledError from exc
    finally:
        tracker.close()
    out = BeatTrackerOut(
        method="model",
        model_id=beat_model.MODEL_ID,
        model_name=beat_model.DISPLAY_NAME,
        model_sha256=[
            beat_model.PINNED[f][0] for f in (beat_model.FULL_FILE, beat_model.SHORT_FILE)
        ],
        license=beat_model.LICENSE,
        backend=report.backend,  # type: ignore[arg-type]
        notes=report.notes,
    )
    return (beats, downs), out


def write_sidecar(path: Path, result: AnalysisResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(result.model_dump_json())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
