"""Real audio capture through Windows WASAPI (PyAudioWPatch). Worker processes only.

Microphones are WASAPI input devices; "loopback" devices record what a playback device is
currently rendering (system audio). Capture starts only when a record job is submitted by
the user. Audio is streamed to a float32 WAV on disk, so memory does not grow with the
recording length, and levels are reported as job telemetry.
"""

from __future__ import annotations

import queue
import time
import uuid
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from xsample.audio.errors import AudioError, Cancelled
from xsample.audio.meter import LevelMeter, to_db
from xsample.project import layout

BLOCK_FRAMES = 1024
TELEMETRY_INTERVAL_S = 0.08
QUEUE_BLOCKS = 4000  # ~85 s of audio at 48 kHz if the disk stalls
PRIVACY_HINT = (
    " Check Windows Settings > Privacy & security > Microphone and make sure desktop apps "
    "may use the microphone, and that no other app has exclusive control of the device."
)


class RecordContext(Protocol):
    def progress(
        self, fraction: float, message: str | None = None, data: dict[str, Any] | None = None
    ) -> None: ...

    def cancelled(self) -> bool: ...

    def stop_requested(self) -> bool: ...


def _pyaudio() -> Any:
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise AudioError("The audio capture library (PyAudioWPatch) is not installed") from exc
    return pyaudio


def _wasapi(p: Any, pyaudio: Any) -> dict[str, Any]:
    try:
        return dict(p.get_host_api_info_by_type(pyaudio.paWASAPI))
    except OSError as exc:
        raise AudioError("Windows WASAPI audio is not available on this system") from exc


def _device_entry(d: dict[str, Any], kind: str, is_default: bool) -> dict[str, Any]:
    return {
        "index": int(d["index"]),
        "name": str(d["name"]),
        "kind": kind,
        "channels": int(d["maxInputChannels"]),
        "sample_rate": int(d["defaultSampleRate"]),
        "is_default": is_default,
    }


def list_devices() -> dict[str, Any]:
    """Enumerate devices without opening any stream (no microphone access happens here)."""
    pyaudio = _pyaudio()
    p = pyaudio.PyAudio()
    try:
        wasapi = _wasapi(p, pyaudio)
        default_in = wasapi.get("defaultInputDevice", -1)
        default_out_name = None
        if wasapi.get("defaultOutputDevice", -1) >= 0:
            default_out_name = p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
        microphones: list[dict[str, Any]] = []
        loopbacks: list[dict[str, Any]] = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] != wasapi["index"]:
                continue
            if d.get("isLoopbackDevice"):
                is_default = default_out_name is not None and d["name"].startswith(default_out_name)
                loopbacks.append(_device_entry(d, "loopback", is_default))
            elif d["maxInputChannels"] > 0:
                microphones.append(_device_entry(d, "microphone", d["index"] == default_in))
        return {"microphones": microphones, "loopbacks": loopbacks}
    finally:
        p.terminate()


def _open_silent_output(p: Any, pyaudio: Any, loopback_name: str) -> Any | None:
    """Keep the render engine running so loopback delivers continuous (silent) frames.

    Without a running render stream WASAPI loopback produces no packets while nothing
    plays, which would remove pauses from the recording and shorten the timeline.
    """
    base = loopback_name.removesuffix(" [Loopback]")
    wasapi_index = _wasapi(p, pyaudio)["index"]
    for i in range(p.get_device_count()):
        d = p.get_device_info_by_index(i)
        # The same endpoint is also listed under MME/DirectSound; only the WASAPI entry
        # is the render stream that the loopback device mirrors.
        if (
            d["hostApi"] == wasapi_index
            and d["maxOutputChannels"] > 0
            and not d.get("isLoopbackDevice")
            and d["name"] == base
        ):
            channels = int(d["maxOutputChannels"])

            def silence(
                _in: Any, frames: int, _t: Any, _s: Any, channels: int = channels
            ) -> tuple[bytes, int]:
                return (bytes(frames * channels * 4), pyaudio.paContinue)

            try:
                return p.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=int(d["defaultSampleRate"]),
                    output=True,
                    output_device_index=int(d["index"]),
                    stream_callback=silence,
                )
            except OSError:
                return None
    return None


def record(ctx: RecordContext, params: dict[str, Any]) -> dict[str, Any]:
    pyaudio = _pyaudio()
    project = Path(params["project_dir"])
    max_seconds = int(params["max_seconds"])
    kind = params["kind"]
    p = pyaudio.PyAudio()
    stream: Any = None
    silent: Any = None
    out: sf.SoundFile | None = None
    out_path = project / layout.RECORDING / f"{uuid.uuid4().hex}.wav"
    finished = False
    try:
        try:
            info = dict(p.get_device_info_by_index(int(params["device_index"])))
        except (OSError, ValueError) as exc:
            raise AudioError(
                "The selected audio device is no longer available; refresh the list"
            ) from exc
        if info["name"] != params["device_name"]:
            raise AudioError("The audio device list changed; refresh the device list and try again")
        if kind == "loopback" and not info.get("isLoopbackDevice"):
            raise AudioError("The selected device is not a loopback device")
        if kind == "microphone" and (info.get("isLoopbackDevice") or info["maxInputChannels"] < 1):
            raise AudioError("The selected device is not a microphone")
        channels, rate = int(info["maxInputChannels"]), int(info["defaultSampleRate"])

        blocks: queue.Queue[bytes] = queue.Queue(maxsize=QUEUE_BLOCKS)
        dropped = [0]

        def on_audio(
            data: bytes | None, _frames: int, _time: Any, _status: int
        ) -> tuple[None, int]:
            if data:
                try:
                    blocks.put_nowait(data)
                except queue.Full:
                    dropped[0] += 1
            return (None, pyaudio.paContinue)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = sf.SoundFile(
            out_path, mode="w", samplerate=rate, channels=channels, format="WAV", subtype="FLOAT"
        )
        if kind == "loopback":
            silent = _open_silent_output(p, pyaudio, info["name"])
        try:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=int(info["index"]),
                frames_per_buffer=BLOCK_FRAMES,
                stream_callback=on_audio,
            )
        except OSError as exc:
            hint = PRIVACY_HINT if kind == "microphone" else ""
            raise AudioError(f"Could not open the audio device: {exc}.{hint}") from exc

        meter = LevelMeter(channels)
        frames = 0
        started = time.monotonic()
        last_emit = 0.0
        stopping = False
        while True:
            if ctx.cancelled():
                raise Cancelled()
            elapsed = time.monotonic() - started
            if not stopping and (ctx.stop_requested() or elapsed >= max_seconds):
                stopping = True
                stream.stop_stream()
            try:
                data = blocks.get(timeout=0.02)
            except queue.Empty:
                data = None
                if stopping:
                    break  # drained after the stream stopped
            while data is not None:
                block = np.frombuffer(data, dtype=np.float32).reshape(-1, channels)
                out.write(block)
                meter.feed(block)
                frames += len(block)
                try:
                    data = blocks.get_nowait()
                except queue.Empty:
                    data = None
            now = time.monotonic()
            if now - last_emit >= TELEMETRY_INTERVAL_S:
                last_emit = now
                window_peak, window_rms = meter.read_window()
                ctx.progress(
                    min(0.999, elapsed / max_seconds),
                    "Recording",
                    {
                        "elapsed_s": elapsed,
                        "captured_s": frames / rate,
                        "sample_rate": rate,
                        "channels": channels,
                        "peak_db": window_peak,
                        "rms_db": window_rms,
                        "hold_db": [to_db(v) for v in meter.peak_hold],
                        "clipping": meter.clip_events > 0,
                        "clip_samples": meter.clip_samples,
                        "dropped_blocks": dropped[0],
                        "device": info["name"],
                    },
                )
        out.close()
        out = None
        if frames == 0:
            raise AudioError("No audio was captured from the device")
        finished = True
        return {
            "path": str(out_path),
            "relative_path": f"{layout.RECORDING}/{out_path.name}",
            "frames": frames,
            "sample_rate": rate,
            "channels": channels,
            "elapsed_s": time.monotonic() - started,
            "captured_s": frames / rate,
            "peak_db": to_db(meter.session_peak),
            "clip_samples": meter.clip_samples,
            "clip_events": meter.clip_events,
            "dropped_blocks": dropped[0],
            "device": info["name"],
            "kind": kind,
        }
    finally:
        for s in (stream, silent):
            if s is not None:
                try:
                    s.stop_stream()
                    s.close()
                except OSError:
                    pass
        p.terminate()
        if out is not None:
            out.close()
        if not finished:
            out_path.unlink(missing_ok=True)
