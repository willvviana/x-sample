"""Versioned analysis sidecar schema (ADR-0011).

Detected results live in `analysis/results/<source_id>/<cache_key>.json`. All positions are
integer samples at the project rate, relative to the start of the source (clips map them to
the timeline). User corrections are not stored here; they live in the project manifest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 2: beat_tracker, backend may be directml, timings.beat_model_s.
# 3: tagger, genres/instruments/vocals may be TagsOut, timings.tagger_s.
SCHEMA_VERSION = 3
# Bump whenever an analyzer change can alter results; it invalidates every cached sidecar.
ANALYZER_VERSION = "dsp-0.2.0"
ANALYZER_ID = "xsample.analysis.dsp"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Unavailable(_M):
    """A result this build cannot produce honestly (e.g. it needs a model not yet approved)."""

    status: Literal["unavailable"] = "unavailable"
    reason: str


class TempoCandidateOut(_M):
    bpm: float
    relation: Literal["primary", "half", "double", "other"]
    score: float


class TempoPoint(_M):
    position: int
    bpm: float


class RhythmOut(_M):
    bpm: float = Field(description="0 when no periodic rhythm was found")
    confidence: float
    candidates: list[TempoCandidateOut]
    tempo_curve: list[TempoPoint]
    variable_tempo: bool
    tempo_changes: list[TempoPoint]
    beats: list[int]
    beat_strength: list[float]
    downbeats: list[int]
    meter: int | None
    meter_confidence: float
    subdivision: Literal["duple", "triplet"] | None
    subdivision_confidence: float
    swing_ratio: float | None
    swing_confidence: float
    syncopation: float | None
    microtiming_ms: float | None
    onsets: list[int]
    onset_strength: list[float]
    notes: list[str]


class KeyOut(_M):
    tonic: int = Field(ge=0, le=11)
    mode: Literal["major", "minor"]
    label: str
    score: float


class KeyRegion(_M):
    start: int
    end: int
    key: KeyOut


class ChordOut(_M):
    start: int
    end: int
    root: int | None
    quality: Literal["maj", "min", "7", "maj7", "min7"] | None
    bass: int | None
    label: str
    confidence: float


class BassNoteOut(_M):
    start: int
    end: int
    pitch_class: int


class MelodyNoteOut(_M):
    start: int
    end: int
    midi: float
    articulation: Literal["attack", "legato", "glide"]
    vibrato_hz: float | None
    vibrato_cents: float | None


class ContourPoint(_M):
    position: int
    midi: float | None


class HarmonyOut(_M):
    key_candidates: list[KeyOut]
    key_confidence: float
    local_keys: list[KeyRegion]
    chords: list[ChordOut]
    harmonic_rhythm_beats: float | None
    changes_per_bar: float | None
    bass_notes: list[BassNoteOut]
    melody_notes: list[MelodyNoteOut]
    melody_contour: list[ContourPoint]
    voiced_ratio: float
    notes: list[str]


class SectionOut(_M):
    start: int
    end: int
    label: str
    confidence: float
    loudness_db: float


class LevelPoint(_M):
    position: int
    db: float


class StructureOut(_M):
    sections: list[SectionOut]
    loudness_curve: list[LevelPoint]
    dynamic_range_db: float | None


class BoundaryEdge(_M):
    """Measurements at one end of the source, for intro/outro repair (requirement 13).

    These are raw measurements, not verdicts: problem detection is a v0.4.0 feature.
    """

    position: int
    edge_amplitude: float = Field(description="|sample| at the boundary (a large value is a step)")
    level_dbfs: float = Field(description="RMS of the 10 ms at the boundary")
    level_vs_median_db: float = Field(description="boundary level relative to the median level")
    beat_offset: float | None = Field(
        description="signed distance to the nearest beat, in beats (0 = on the beat)"
    )
    beats_from_downbeat: float | None = Field(
        description="start: beats until the first downbeat; end: beats since the last downbeat"
    )
    section_label: str | None
    chord: str | None = Field(description="chord at the boundary")
    chord_is_tonic: bool | None = Field(description="that chord is the tonic triad of the key")


class BoundariesOut(_M):
    start: BoundaryEdge
    end: BoundaryEdge


class BeatTrackerOut(_M):
    """Which beat/downbeat tracker produced rhythm.beats and rhythm.downbeats, and where."""

    method: Literal["dsp", "model"]
    model_id: str | None = None
    model_name: str | None = None
    model_sha256: list[str] = Field(default_factory=list)
    license: str | None = None
    backend: Literal["cpu", "directml"] = "cpu"
    notes: list[str] = Field(default_factory=list)


class TagSegment(_M):
    """A stretch of windows where the label scored at or above the uncertain threshold."""

    start: int
    end: int
    score: float


class TagOut(_M):
    kind: Literal["genre", "instrument", "vocal"]
    label: str
    audioset_classes: list[str]
    score: float = Field(description="highest window score (AudioSet sigmoid, not calibrated)")
    status: Literal["detected", "uncertain", "not_detected"]
    evaluated: bool = Field(
        description="per-class precision/recall was measured on labelled held-out audio "
        "(docs/EVALUATION.md); false means its accuracy has not been measured"
    )
    activity: list[TagSegment]


class TagsOut(_M):
    status: Literal["available"] = "available"
    tags: list[TagOut]


class TaggerOut(_M):
    """Which tagging model produced genres, instruments and vocals, and where it ran."""

    model_id: str
    model_name: str
    model_sha256: str
    license: str
    attribution: str
    backend: Literal["cpu", "directml"] = "cpu"
    window_s: float
    hop_s: float
    detect_threshold: float
    uncertain_threshold: float
    notes: list[str] = Field(default_factory=list)


class Timings(_M):
    decode_features_s: float
    beat_model_s: float = 0.0
    tagger_s: float = 0.0
    rhythm_s: float
    harmony_s: float
    structure_s: float
    total_s: float


class AnalysisResult(_M):
    schema_version: int = SCHEMA_VERSION
    analyzer_id: str = ANALYZER_ID
    analyzer_version: str = ANALYZER_VERSION
    backend: Literal["cpu", "directml"] = Field(
        "cpu",
        description="where model inference ran; DSP analyzers are CPU-only, and directml is "
        "reported only after strict, profiled placement and a CPU parity check",
    )
    source_id: str
    cache_key: str
    media_sha256: str
    sample_rate: int
    analysis_rate: int
    frames: int
    params: dict[str, float | int | str]
    created_at: str
    timings: Timings
    beat_tracker: BeatTrackerOut
    tagger: TaggerOut | None = None
    rhythm: RhythmOut
    harmony: HarmonyOut
    structure: StructureOut
    boundaries: BoundariesOut
    genres: TagsOut | Unavailable = Field(discriminator="status")
    instruments: TagsOut | Unavailable = Field(discriminator="status")
    vocals: TagsOut | Unavailable = Field(discriminator="status")
