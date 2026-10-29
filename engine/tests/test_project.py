"""ADR-0006: persistence, integrity, recovery, locking and non-destructive edits."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.audio_fixtures import ramp_wav, sine_array, write_wav
from xsample.audio import media
from xsample.audio.render import render_blocks
from xsample.config import EngineSettings
from xsample.project import layout, ops, store
from xsample.project.model import Anchor, Clip, Project, source_to_timeline, timeline_to_source
from xsample.project.service import NoProjectOpenError, ProjectService
from xsample.project.store import (
    InvalidProjectError,
    ProjectError,
    ProjectExistsError,
    ProjectLockedError,
    ProjectNotFoundError,
)

RATE = 44100


@pytest.fixture
def service(settings: EngineSettings) -> ProjectService:
    return ProjectService(settings)


def import_file(service: ProjectService, src: Path, channels: str = "keep") -> str:
    """Import a real file in-process (same code a worker runs) and apply the result."""
    root = service.root()
    incoming = service.new_incoming_path(src.suffix)
    incoming.write_bytes(src.read_bytes())
    params = {"project_dir": str(root)}
    result = media.import_audio(
        str(root),
        str(incoming),
        src.name,
        service.view_required().sample_rate,
        channels,
        "import",
        None,
        lambda _f, _m: None,
        lambda: False,
    )
    service.apply_import(result, params)
    return str(result["source"]["id"])


@pytest.fixture
def ramp(tmp_path: Path) -> Path:
    return ramp_wav(tmp_path / "ramp.wav", 100_000, RATE)


# --- create / open / save ----------------------------------------------------------------


def test_create_makes_the_documented_layout(
    service: ProjectService, settings: EngineSettings
) -> None:
    view = service.create("My Song", RATE)
    root = Path(view.path)
    assert root == (settings.projects_dir / "My Song").resolve()
    for d in layout.DIRECTORIES:
        assert (root / d).is_dir(), d
    manifest = json.loads((root / "project.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == 1 and manifest["sample_rate"] == RATE
    assert manifest["tracks"] == [] and manifest["clips"] == []
    assert (root / layout.LOCK).exists()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("a/b:c", "a_b_c"), ("  name.  ", "name"), ("CON", "_CON"), ("x" * 200, "x" * 80)],
)
def test_project_names_are_made_safe(raw: str, expected: str) -> None:
    assert store.sanitize_name(raw) == expected
    with pytest.raises(ProjectError):
        store.sanitize_name("  ...  ")


def test_create_rejects_existing_folder_and_bad_rate(
    service: ProjectService, settings: EngineSettings
) -> None:
    (settings.projects_dir / "Taken").mkdir(parents=True)
    (settings.projects_dir / "Taken" / "file.txt").write_text("x")
    with pytest.raises(ProjectExistsError):
        service.create("Taken", RATE)
    with pytest.raises(ProjectError, match="sample rate"):
        service.create("Odd", 12345)
    assert not service.is_open


def test_save_and_reopen_preserves_everything(
    service: ProjectService, ramp: Path, tmp_path: Path
) -> None:
    service.create("Persist", RATE)
    source_id = import_file(service, ramp)
    import_file(service, write_wav(tmp_path / "t.wav", sine_array(1.0, RATE, (440, 440)), RATE))
    view = service.view_required()
    clip = view.clips[0]
    service.edit(lambda p: ops.trim_clip(p, clip.id, 1_000, 51_000))
    sample = service.edit(lambda p: ops.slice_clip_to_sample(p, clip.id, 2_000, 12_000, "Hit"))
    service.edit(
        lambda p: ops.update_track(p, view.tracks[1].id, muted=True, gain=0.5, name="Tone")
    )
    before = service.view_required()
    root = Path(before.path)
    original = (root / before.sources[0].stored_path).read_bytes()

    service.close()
    reopened = service.open(str(root))
    assert reopened.model_dump() | {"recovered_from_backup": False} == before.model_dump()
    assert {s.id for s in reopened.sources} >= {source_id}
    assert reopened.clips[0].source_in == 1_000 and reopened.clips[0].duration == 50_000
    assert reopened.samples[0].id == sample.id and reopened.samples[0].name == "Hit"
    assert reopened.tracks[1].muted and reopened.tracks[1].gain == 0.5
    assert (root / before.sources[0].stored_path).read_bytes() == original


def test_every_edit_is_persisted_immediately(service: ProjectService, ramp: Path) -> None:
    service.create("Autosave", RATE)
    import_file(service, ramp)
    root = service.root()
    clip = service.view_required().clips[0]
    service.edit(lambda p: ops.trim_clip(p, clip.id, 10, 90))
    on_disk = json.loads((root / "project.json").read_text(encoding="utf-8"))
    assert on_disk["clips"][0]["source_in"] == 10 and on_disk["clips"][0]["duration"] == 80
    assert on_disk["revision"] == service.view_required().revision
    assert any(h["operation"] == "clip.trim" for h in on_disk["history"])
    assert all(h["operation"] for h in on_disk["history"])


def test_failed_edit_changes_nothing(service: ProjectService, ramp: Path) -> None:
    service.create("Atomic", RATE)
    import_file(service, ramp)
    before = service.view_required()
    root = service.root()
    raw_before = (root / "project.json").read_bytes()

    def broken(p: Project) -> None:
        p.tracks[0].name = "half-applied"
        ops.trim_clip(p, p.clips[0].id, 5, 10**9)  # out of range -> raises

    with pytest.raises(ProjectError):
        service.edit(broken)
    assert service.view_required() == before
    assert (root / "project.json").read_bytes() == raw_before


def test_operations_need_an_open_project(service: ProjectService) -> None:
    assert service.view() is None
    with pytest.raises(NoProjectOpenError):
        service.edit(lambda p: None)
    with pytest.raises(NoProjectOpenError):
        service.root()


def test_cannot_open_a_second_project_without_closing(service: ProjectService) -> None:
    service.create("One", RATE)
    with pytest.raises(ProjectError, match="Close the current project"):
        service.create("Two", RATE)
    service.close()
    service.create("Two", RATE)


# --- invalid and damaged projects -----------------------------------------------------------


def test_open_errors_are_clear(service: ProjectService, tmp_path: Path) -> None:
    with pytest.raises(ProjectNotFoundError):
        service.open(str(tmp_path / "nope"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(InvalidProjectError, match="does not contain an X-Sample project"):
        service.open(str(empty))
    assert not service.is_open


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "not valid JSON"),
        ("{not json", "not valid JSON"),
        ('{"hello": 1}', "not an X-Sample project"),
        ('{"format_version": 999}', "newer X-Sample"),
        ('{"format_version": 1, "id": "x"}', "invalid at"),
    ],
)
def test_damaged_manifest_without_backup_is_rejected(
    content: str, message: str, tmp_path: Path
) -> None:
    root = tmp_path / "p"
    root.mkdir()
    (root / "project.json").write_text(content, encoding="utf-8")
    with pytest.raises(InvalidProjectError, match=message):
        store.load_project(root)


def test_manifest_with_dangling_references_is_rejected(service: ProjectService, ramp: Path) -> None:
    service.create("Dangling", RATE)
    import_file(service, ramp)
    root = service.root()
    service.close()
    manifest = json.loads((root / "project.json").read_text(encoding="utf-8"))
    manifest["clips"][0]["track_id"] = "trk_missing"
    (root / "project.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "project.json.bak").unlink()
    with pytest.raises(InvalidProjectError, match="unknown track"):
        service.open(str(root))


def test_clip_past_end_of_source_and_overlaps_are_rejected(
    service: ProjectService, ramp: Path
) -> None:
    service.create("Bounds", RATE)
    import_file(service, ramp)
    clip = service.view_required().clips[0]
    with pytest.raises(ValueError, match="extends past the end"):
        service.edit(
            lambda p: (
                setattr(p.clips[0], "duration", 100_001) or Project.model_validate(p.model_dump())
            )
        )
    track = service.view_required().tracks[0].id
    sample = service.edit(lambda p: ops.slice_clip_to_sample(p, clip.id, 0, 1000, None))
    with pytest.raises(ProjectError, match="overlap"):
        service.edit(lambda p: ops.place_sample(p, sample.id, track, 500))
    service.edit(lambda p: ops.place_sample(p, sample.id, track, 100_000))


def test_corrupt_manifest_recovers_from_backup(service: ProjectService, ramp: Path) -> None:
    service.create("Recover", RATE)
    import_file(service, ramp)
    clip = service.view_required().clips[0]
    service.edit(lambda p: ops.trim_clip(p, clip.id, 100, 900))  # makes a .bak of the previous save
    root = service.root()
    expected_clips = service.view_required().clips
    service.close()

    (root / "project.json").write_text(
        '{"format_version": 1, "id": "tru', encoding="utf-8"
    )  # crash mid-write
    view = service.open(str(root))
    assert view.recovered_from_backup and any("backup" in i for i in view.issues)
    assert view.sources and view.clips  # work survived
    assert len(view.clips) == len(expected_clips)
    # the manifest was rewritten: next open is clean
    service.close()
    assert not service.open(str(root)).recovered_from_backup


def test_stale_temp_files_are_cleaned_but_recordings_kept(service: ProjectService) -> None:
    root = Path(service.create("Stale", RATE).path)
    service.close()
    (root / layout.TMP / "x.wav").write_bytes(b"junk")
    (root / layout.INCOMING / "y.wav").write_bytes(b"junk")
    (root / "project.json.tmp").write_text("partial")
    (root / layout.EXPORTS / ".abc.part").write_bytes(b"junk")
    (root / layout.RECORDING / "rec.wav").write_bytes(b"maybe recoverable")
    service.open(str(root))
    assert not list((root / layout.TMP).iterdir()) and not list((root / layout.INCOMING).iterdir())
    assert not (root / "project.json.tmp").exists() and not list(
        (root / layout.EXPORTS).glob(".*.part")
    )
    assert (root / layout.RECORDING / "rec.wav").exists()


def test_lock_blocks_another_live_process_and_survives_stale(
    tmp_path: Path, settings: EngineSettings
) -> None:
    import psutil

    first = ProjectService(settings)
    root = Path(first.create("Locked", RATE).path)
    first.close()

    parent = psutil.Process().parent()
    assert parent is not None
    (root / layout.LOCK).write_text(
        json.dumps({"pid": parent.pid, "created": parent.create_time(), "token": "x"})
    )
    with pytest.raises(ProjectLockedError, match="already open"):
        ProjectService(settings).open(str(root))

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    (root / layout.LOCK).write_text(json.dumps({"pid": dead.pid, "created": 0, "token": "x"}))
    second = ProjectService(settings)
    second.open(str(root))  # stale lock from a crashed process is taken over
    second.close()
    assert not (root / layout.LOCK).exists()


def test_recent_and_listing(service: ProjectService, settings: EngineSettings) -> None:
    service.create("Alpha", RATE)
    service.close()
    service.create("Beta", RATE)
    service.close()
    listing = service.list_projects()
    assert listing["projects_dir"] == str(settings.projects_dir)
    assert [p["name"] for p in listing["projects"]][:2] == ["Beta", "Alpha"]


# --- non-destructive edits -----------------------------------------------------------------------


def test_trim_and_slice_are_sample_accurate_and_never_touch_audio(
    service: ProjectService, ramp: Path
) -> None:
    service.create("Edits", RATE)
    import_file(service, ramp)
    view = service.view_required()
    root = Path(view.path)
    clip, src = view.clips[0], view.sources[0]
    stored = root / src.stored_path
    media_file = root / src.media.path
    hashes = [hashlib.sha256(f.read_bytes()).hexdigest() for f in (stored, media_file)]
    ref, _ = sf.read(ramp, dtype="float32", always_2d=True)

    trimmed = service.edit(lambda p: ops.trim_clip(p, clip.id, 7_001, 40_001))
    assert (trimmed.source_in, trimmed.timeline_start, trimmed.duration) == (7_001, 7_001, 33_000)

    sample = service.edit(lambda p: ops.slice_clip_to_sample(p, clip.id, 10_000, 20_000, "S"))
    assert (sample.source_in, sample.duration) == (10_000, 10_000)
    with pytest.raises(ProjectError, match="inside the clip"):
        service.edit(lambda p: ops.slice_clip_to_sample(p, clip.id, 0, 50_000, None))

    plan = service.sample_plan(sample.id)
    out = np.concatenate(list(render_blocks(plan, 0, sample.duration)))
    assert np.array_equal(out[:, 0], ref[10_000:20_000, 0])
    mix = np.concatenate(list(render_blocks(service.mix_plan(), 7_001, 40_001)))
    assert np.array_equal(mix[:, 0], ref[7_001:40_001, 0])
    assert not np.concatenate(list(render_blocks(service.mix_plan(), 0, 7_001))).any()

    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in (stored, media_file)] == hashes
    assert not stored.stat().st_mode & 0o200

    reset = service.edit(lambda p: ops.reset_trim(p, clip.id))
    assert (reset.source_in, reset.timeline_start, reset.duration) == (0, 0, 100_000)


def test_trim_keeps_audio_in_place_and_reset_restores_it(
    service: ProjectService, ramp: Path
) -> None:
    service.create("Place", RATE)
    import_file(service, ramp)
    clip = service.view_required().clips[0]
    service.edit(lambda p: ops.move_clip(p, clip.id, 5_000))
    ref, _ = sf.read(ramp, dtype="float32", always_2d=True)
    service.edit(lambda p: ops.trim_clip(p, clip.id, 6_000, 9_000))
    got = np.concatenate(list(render_blocks(service.mix_plan(), 6_000, 9_000)))
    assert np.array_equal(got[:, 0], ref[1_000:4_000, 0]), (
        "audio stays where it was on the timeline"
    )
    reset = service.edit(lambda p: ops.reset_trim(p, clip.id))
    assert (reset.source_in, reset.timeline_start) == (0, 5_000)


def test_time_map_math_including_non_unity_maps() -> None:
    clip = Clip(
        id="c",
        track_id="t",
        source_id="s",
        source_in=100,
        timeline_start=1000,
        duration=500,
        time_map=[
            Anchor(source=100, timeline=1000),
            Anchor(source=300, timeline=1200),
            Anchor(source=500, timeline=1600),
        ],
    )
    assert timeline_to_source(clip, 1000) == 100 and timeline_to_source(clip, 1200) == 300
    assert timeline_to_source(clip, 1400) == 400  # second segment runs at 0.5 source/timeline
    assert source_to_timeline(clip, 400) == 1400 and source_to_timeline(clip, 300) == 1200
    for t in range(1000, 1600, 37):  # integer round trip never drifts by more than a frame
        assert abs(source_to_timeline(clip, timeline_to_source(clip, t)) - t) <= 2


def test_non_unity_time_maps_are_rejected_in_phase_1(service: ProjectService, ramp: Path) -> None:
    service.create("Stretch", RATE)
    import_file(service, ramp)

    def stretch(p: Project) -> None:
        c = p.clips[0]
        c.time_map = [
            Anchor(source=0, timeline=0),
            Anchor(source=c.duration, timeline=c.duration * 2),
        ]
        c.duration *= 2
        Project.model_validate(p.model_dump())

    with pytest.raises(ValueError, match="unity"):
        service.edit(stretch)


def test_multiple_tracks_mix_mute_solo_and_gain(service: ProjectService, tmp_path: Path) -> None:
    service.create("Multi", RATE)
    a = write_wav(tmp_path / "a.wav", np.full((1000, 2), 0.25, np.float32), RATE, "FLOAT")
    b = write_wav(tmp_path / "b.wav", np.full((1000, 2), 0.5, np.float32), RATE, "FLOAT")
    import_file(service, a)
    import_file(service, b)
    view = service.view_required()
    assert [t.name for t in view.tracks] == ["a", "b"] and len(view.clips) == 2

    def level() -> float:
        return float(np.concatenate(list(render_blocks(service.mix_plan(), 0, 1000)))[500, 0])

    assert level() == pytest.approx(0.75)
    service.edit(lambda p: ops.update_track(p, view.tracks[1].id, gain=0.5))
    assert level() == pytest.approx(0.5)
    service.edit(lambda p: ops.update_track(p, view.tracks[0].id, muted=True))
    assert level() == pytest.approx(0.25)
    service.edit(lambda p: ops.update_track(p, view.tracks[0].id, muted=False, solo=True))
    assert level() == pytest.approx(0.25)  # solo A only
    service.edit(lambda p: ops.remove_track(p, view.tracks[0].id))
    assert len(service.view_required().clips) == 1 and len(service.view_required().tracks) == 1
    with pytest.raises(ProjectError, match="gain"):
        service.edit(lambda p: ops.update_track(p, p.tracks[0].id, gain=9))


def test_same_file_twice_shares_one_source(service: ProjectService, ramp: Path) -> None:
    service.create("Dedup", RATE)
    first = import_file(service, ramp)
    second = import_file(service, ramp)
    view = service.view_required()
    assert first == second and len(view.sources) == 1 and len(view.clips) == 2
    assert len(list((Path(view.path) / layout.SOURCES).iterdir())) == 1


def test_source_status_and_rebuild_flag(service: ProjectService, ramp: Path) -> None:
    service.create("Status", RATE)
    import_file(service, ramp)
    view = service.view_required()
    root = Path(view.path)
    assert view.sources[0].status == "ok" and not view.issues
    (root / view.sources[0].media.peaks_path).unlink()
    degraded = service.view_required()
    assert degraded.sources[0].status == "needs_rebuild" and degraded.issues
    with pytest.raises(ProjectError, match=r"rebuilt|missing"):
        peak_file = root / view.sources[0].media.path
        peak_file.unlink()
        service.mix_plan()


def test_processing_metadata_is_versioned(service: ProjectService, ramp: Path) -> None:
    service.create("Prov", RATE)
    import_file(service, ramp)
    m = service.view_required().sources[0].media
    rec = m.processing[0]
    assert rec.operation == "import.decode" and rec.engine_version
    assert rec.tool["ffmpeg"].startswith("ffmpeg version") and rec.tool["libsndfile"]
    assert rec.inputs == [service.view_required().sources[0].sha256]
    assert rec.params["sample_rate"] == RATE and rec.output["frames"] == m.frames
    assert m.pcm_sha256 == rec.output["pcm_sha256"]
