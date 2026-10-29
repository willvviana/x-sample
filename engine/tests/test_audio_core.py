"""Decode, peak pyramid, render and meter tests using real generated audio."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tests.audio_fixtures import make_all_formats, ramp_wav, sine_array, write_wav
from xsample.audio import ffmpeg, media
from xsample.audio.errors import AudioError, Cancelled
from xsample.audio.meter import LevelMeter
from xsample.audio.peaks import BASE_BLOCK, PeakBuilder, Pyramid, read_columns
from xsample.audio.render import RenderPlan, Segment, export_audio, preview_wav, render_blocks
from xsample.project import layout


@pytest.fixture(scope="module")
def formats(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return make_all_formats(tmp_path_factory.mktemp("formats"), seconds=3.0, rate=44100)


def _project(tmp_path: Path) -> Path:
    layout.ensure_layout(tmp_path)
    return tmp_path


def _import(tmp_path: Path, src: Path, rate: int = 44100, channels: str = "keep") -> dict:
    project = _project(tmp_path)
    incoming = project / layout.INCOMING / src.name
    incoming.write_bytes(src.read_bytes())
    return media.import_audio(
        str(project),
        str(incoming),
        src.name,
        rate,
        channels,
        "import",
        None,
        lambda _f, _m: None,
        lambda: False,
    )


@pytest.mark.parametrize("fmt", ["wav", "flac", "ogg", "mp3", "m4a"])
def test_probe_and_import_every_format(fmt: str, formats: dict[str, Path], tmp_path: Path) -> None:
    info = ffmpeg.probe(formats[fmt])
    assert info.sample_rate == 44100 and info.channels == 2

    result = _import(tmp_path, formats[fmt])
    m = result["media"]
    assert m["sample_rate"] == 44100 and m["channels"] == 2
    # Lossy codecs add encoder delay/padding; WAV/FLAC must be exact.
    expected = 3 * 44100
    tolerance = 0 if fmt in {"wav", "flac"} else 4096
    assert abs(m["frames"] - expected) <= tolerance, (fmt, m["frames"])
    # AAC/Vorbis can overshoot at the start (verified: ~0.76 at frame 587 for AAC).
    assert 0.3 < m["peak"] < (0.6 if fmt in {"wav", "flac"} else 0.9)

    data, rate = sf.read(tmp_path / m["path"], dtype="float32")
    assert rate == 44100 and data.shape == (m["frames"], 2)
    # Left channel is a 440 Hz tone: the dominant FFT bin must be there.
    mid = data[44100 : 44100 + 32768, 0]
    spectrum = np.abs(np.fft.rfft(mid * np.hanning(len(mid))))
    assert abs(np.argmax(spectrum) * 44100 / len(mid) - 440) < 3
    right = np.abs(np.fft.rfft(data[44100 : 44100 + 32768, 1] * np.hanning(32768)))
    assert abs(np.argmax(right) * 44100 / 32768 - 660) < 3


def test_original_is_preserved_byte_for_byte_and_read_only(
    formats: dict[str, Path], tmp_path: Path
) -> None:
    original = formats["flac"].read_bytes()
    result = _import(tmp_path, formats["flac"])
    stored = tmp_path / result["source"]["stored_path"]
    assert stored.read_bytes() == original
    assert stored.parent.name == "sources" and stored.stem == result["source"]["sha256"]
    assert not stored.stat().st_mode & 0o200, "original must be read-only"
    assert not list((tmp_path / layout.INCOMING).iterdir()), "upload must be consumed"


def test_wav_import_is_sample_exact(tmp_path: Path) -> None:
    src = ramp_wav(tmp_path / "ramp.wav", 100_000)
    expected, _ = sf.read(src, dtype="float32", always_2d=True)
    result = _import(tmp_path / "p", src)
    got, _ = sf.read(tmp_path / "p" / result["media"]["path"], dtype="float32", always_2d=True)
    assert result["media"]["frames"] == 100_000
    assert np.array_equal(got[:, 0], expected[:, 0])


def test_resample_and_channel_conversion(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "s.wav", sine_array(2.0, 44100, (440.0, 880.0)), 44100)
    up = _import(tmp_path / "a", src, rate=48000)["media"]
    assert up["sample_rate"] == 48000 and abs(up["frames"] - 96000) <= 2
    mono = _import(tmp_path / "b", src, channels="mono")["media"]
    assert mono["channels"] == 1 and mono["frames"] == 88200
    data, _ = sf.read(tmp_path / "b" / mono["path"], dtype="float32")
    assert np.max(np.abs(data)) <= 1.0

    m = write_wav(tmp_path / "m.wav", sine_array(1.0, 22050, (300.0,)), 22050)
    stereo = _import(tmp_path / "c", m, rate=44100, channels="stereo")["media"]
    assert stereo["channels"] == 2 and abs(stereo["frames"] - 44100) <= 2


def test_multichannel_source_is_downmixed_to_stereo(tmp_path: Path) -> None:
    six = write_wav(
        tmp_path / "six.wav", sine_array(1.0, 44100, (100, 200, 300, 400, 500, 600)), 44100
    )
    result = _import(tmp_path / "p", six)
    assert result["source"]["channels"] == 6 and result["media"]["channels"] == 2


def test_duplicate_import_reuses_media(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "s.wav", sine_array(1.0, 44100, (440.0, 440.0)), 44100)
    first = _import(tmp_path / "p", src)
    second = _import(tmp_path / "p", src)
    assert second["reused_media"] and second["source"]["id"] == first["source"]["id"]
    assert len(list((tmp_path / "p" / layout.SOURCES).iterdir())) == 1


@pytest.mark.parametrize(
    "name,payload,message",
    [
        ("empty.wav", b"", "empty"),
        ("garbage.mp3", b"this is not audio" * 1000, "Not a readable audio file"),
        ("truncated.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", "Not a readable audio file"),
    ],
)
def test_invalid_files_fail_cleanly_and_leave_nothing(
    name: str, payload: bytes, message: str, tmp_path: Path
) -> None:
    bad = tmp_path / name
    bad.write_bytes(payload)
    with pytest.raises(AudioError, match=message):
        _import(tmp_path / "p", bad)
    p = tmp_path / "p"
    assert not list((p / layout.SOURCES).iterdir())
    assert not list((p / layout.INCOMING).iterdir())
    assert not list((p / layout.TMP).iterdir())
    assert not list((p / layout.MEDIA).iterdir())


def test_video_only_file_is_rejected(tmp_path: Path) -> None:
    from tests.audio_fixtures import run_ffmpeg

    video = tmp_path / "v.mp4"
    run_ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x64:d=1", "-an", str(video))
    with pytest.raises(AudioError, match="no audio stream"):
        _import(tmp_path / "p", video)


def test_cancelled_import_cleans_up(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "s.wav", sine_array(20.0, 44100, (440.0, 440.0)), 44100)
    project = _project(tmp_path / "p")
    incoming = project / layout.INCOMING / "s.wav"
    incoming.write_bytes(src.read_bytes())
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 6

    with pytest.raises(Cancelled):
        media.import_audio(
            str(project),
            str(incoming),
            "s.wav",
            44100,
            "keep",
            "import",
            None,
            lambda _f, _m: None,
            cancelled,
        )
    for sub in (layout.SOURCES, layout.INCOMING, layout.TMP, layout.MEDIA, layout.PEAKS):
        assert not list((project / sub).iterdir()), sub


# --- peak pyramid -------------------------------------------------------------------


def _pyramid(data: np.ndarray, rate: int = 44100) -> Pyramid:
    builder = PeakBuilder(rate, data.shape[1])
    for i in range(0, len(data), 10_007):  # awkward chunking exercises the carry logic
        builder.add(data[i : i + 10_007])
    return builder.finalize()


def test_pyramid_matches_brute_force_at_every_level() -> None:
    rng = np.random.default_rng(1)
    data = rng.uniform(-0.95, 0.95, (200_003, 2)).astype(np.float32)
    pyr = _pyramid(data)
    assert pyr.frames == len(data) and len(pyr.levels) == 6
    for k, size in enumerate(pyr.block_sizes):
        lvl = pyr.levels[k]
        assert len(lvl) == -(-len(data) // size)
        for b in (0, len(lvl) // 2, len(lvl) - 1):
            seg = data[b * size : (b + 1) * size]
            assert lvl[b, 0, 0] == np.rint(seg[:, 0].min() * 32767)
            assert lvl[b, 1, 1] == np.rint(seg[:, 1].max() * 32767)


def test_peak_columns_bound_the_true_extrema(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    data = rng.uniform(-0.95, 0.95, (300_000, 1)).astype(np.float32)
    pyr = _pyramid(data)
    pyr.save(tmp_path / "p.npz")
    loaded = Pyramid.load(tmp_path / "p.npz")
    assert all(lv.tolist() == o.tolist() for lv, o in zip(loaded.levels, pyr.levels, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(loaded.levels, pyr.levels, strict=True))

    def raw(a: int, b: int) -> np.ndarray:
        return data[a:b]

    for start, end, cols in [(0, 300_000, 1000), (50_000, 120_000, 700), (0, 300_000, 40)]:
        mins, maxs = read_columns(loaded, raw, start, end, cols)
        assert mins.shape == (cols, 1)
        span = (end - start) / cols
        for c in (0, cols // 3, cols - 1):
            lo, hi = int(start + c * span), int(start + (c + 1) * span)
            seg = data[lo:hi, 0]
            # Block alignment widens a column slightly: it must contain the true extrema.
            assert maxs[c, 0] >= np.rint(seg.max() * 32767) - 1
            assert mins[c, 0] <= np.rint(seg.min() * 32767) + 1


def test_zoomed_in_columns_use_raw_samples() -> None:
    data = sine_array(0.1, 44100, (1000.0,), amp=0.8)
    pyr = _pyramid(data)
    start, end, cols = 100, 100 + 4 * BASE_BLOCK // 2, 32  # < 64 frames per column
    mins, maxs = read_columns(pyr, lambda a, b: data[a:b], start, end, cols)
    expected = np.rint(data[start : start + 32 * 4, 0].reshape(32, 4) * 32767)
    assert np.array_equal(mins[:, 0], expected.min(axis=1))
    assert np.array_equal(maxs[:, 0], expected.max(axis=1))


def test_pyramid_is_small_relative_to_audio() -> None:
    data = np.zeros((441_000, 2), dtype=np.float32)
    pyr = _pyramid(data)
    peak_bytes = sum(lv.nbytes for lv in pyr.levels)
    assert peak_bytes < data.nbytes / 20


# --- render / export ----------------------------------------------------------------


def _plan(path: Path, frames: int, rate: int = 44100, **seg: float) -> RenderPlan:
    channels = sf.info(path).channels
    return RenderPlan(
        rate,
        (
            Segment(
                str(path),
                channels,
                int(seg.get("source_in", 0)),
                int(seg.get("start", 0)),
                frames,
                float(seg.get("gain", 1.0)),
            ),
        ),
    )


def test_render_is_sample_accurate_and_mixes(tmp_path: Path) -> None:
    ramp = ramp_wav(tmp_path / "r.wav", 50_000)
    ref, _ = sf.read(ramp, dtype="float32", always_2d=True)
    plan = _plan(ramp, 10_000, source_in=12_345, start=500)
    out = np.concatenate(list(render_blocks(plan, 0, 11_000, block_frames=3001)))
    assert out.shape == (11_000, 2)
    assert not out[:500].any() and not out[10_500:].any()
    assert np.array_equal(out[500:10_500, 0], ref[12_345 : 12_345 + 10_000, 0])
    assert np.array_equal(out[:, 0], out[:, 1]), "mono must be duplicated to both channels"

    both = RenderPlan(44100, plan.segments + _plan(ramp, 10_000, source_in=0, start=500).segments)
    mixed = np.concatenate(list(render_blocks(both, 500, 10_500)))
    assert np.allclose(mixed[:, 0], ref[12_345:22_345, 0] + ref[0:10_000, 0], atol=1e-6)


def test_export_is_valid_audio_and_exact_for_float(tmp_path: Path) -> None:
    ramp = ramp_wav(tmp_path / "r.wav", 80_000)
    ref, _ = sf.read(ramp, dtype="float32", always_2d=True)
    plan = _plan(ramp, 80_000)
    stats = export_audio(
        plan,
        1_000,
        41_000,
        tmp_path / "out",
        "slice",
        "wav",
        "FLOAT",
        lambda _f, _m: None,
        lambda: False,
        {},
    )
    out, rate = sf.read(tmp_path / "out" / stats["filename"], dtype="float32")
    assert rate == 44100 and out.shape == (40_000, 2) and stats["frames"] == 40_000
    assert np.array_equal(out[:, 0], ref[1_000:41_000, 0])

    info = ffmpeg.probe(tmp_path / "out" / stats["filename"])  # a second, independent reader
    assert info.sample_rate == 44100 and info.channels == 2
    assert abs(info.duration_s - 40_000 / 44100) < 1e-3

    again = export_audio(
        plan,
        0,
        100,
        tmp_path / "out",
        "slice",
        "wav",
        "PCM_16",
        lambda _f, _m: None,
        lambda: False,
        {},
    )
    assert again["filename"] != stats["filename"], "exports never overwrite"
    pcm16, _ = sf.read(tmp_path / "out" / again["filename"], dtype="float32")
    assert np.max(np.abs(pcm16[:, 0] - ref[0:100, 0])) <= 1 / 32767 + 1e-7


@pytest.mark.parametrize(
    ("fmt", "subtype"), [("wav", "PCM_24"), ("flac", "PCM_16"), ("flac", "PCM_24")]
)
def test_export_formats_are_playable(fmt: str, subtype: str, tmp_path: Path) -> None:
    src = write_wav(tmp_path / "t.wav", sine_array(1.0, 44100, (440.0, 440.0)), 44100)
    stats = export_audio(
        _plan(src, 44100),
        0,
        44100,
        tmp_path / "o",
        "x",
        fmt,
        subtype,
        lambda _f, _m: None,
        lambda: False,
        {},
    )
    assert ffmpeg.probe(tmp_path / "o" / stats["filename"]).sample_rate == 44100
    with pytest.raises(AudioError):
        export_audio(
            _plan(src, 44100),
            0,
            10,
            tmp_path / "o",
            "x",
            "flac",
            "FLOAT",
            lambda _f, _m: None,
            lambda: False,
            {},
        )
    with pytest.raises(AudioError, match="empty"):
        export_audio(
            _plan(src, 44100),
            5,
            5,
            tmp_path / "o",
            "x",
            "wav",
            "PCM_16",
            lambda _f, _m: None,
            lambda: False,
            {},
        )


def test_preview_wav_is_decodable(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "t.wav", sine_array(1.0, 44100, (440.0, 440.0)), 44100)
    data = preview_wav(_plan(src, 44100), 1000, 2000)
    assert data[:4] == b"RIFF"
    path = tmp_path / "p.wav"
    path.write_bytes(data)
    assert sf.info(path).frames == 1000


def test_export_cancel_removes_partial_file(tmp_path: Path) -> None:
    src = write_wav(tmp_path / "t.wav", sine_array(5.0, 44100, (440.0, 440.0)), 44100)
    n = {"c": 0}

    def cancelled() -> bool:
        n["c"] += 1
        return n["c"] > 1

    with pytest.raises(Cancelled):
        export_audio(
            _plan(src, 5 * 44100),
            0,
            5 * 44100,
            tmp_path / "o",
            "x",
            "wav",
            "PCM_16",
            lambda _f, _m: None,
            cancelled,
            {},
        )
    assert not list((tmp_path / "o").iterdir())


# --- meter ---------------------------------------------------------------------------


def test_meter_levels_and_clipping() -> None:
    meter = LevelMeter(2)
    block = np.zeros((1000, 2), dtype=np.float32)
    block[:, 0] = 0.5
    block[10, 1] = 1.0
    assert meter.feed(block) == 1
    peak, rms = meter.read_window()
    assert abs(peak[0] - (-6.02)) < 0.05 and abs(rms[0] - (-6.02)) < 0.05
    assert peak[1] == 0.0 and meter.clip_samples == 1 and meter.clip_events == 1
    quiet = meter.feed(np.full((100, 2), 0.01, dtype=np.float32))
    assert quiet == 0 and meter.clip_events == 1
    assert meter.peak_hold[1] == 1.0
