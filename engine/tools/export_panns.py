"""Export PANNs MobileNetV1 (CC BY 4.0 weights, MIT code) to the ONNX file the tagger uses.

Not part of the engine environment: torch, torchlibrosa and librosa are needed only here.
Run it in an isolated, temporary virtual environment on a checkpoint downloaded with
approval that matches CHECKPOINT_SHA256, with the model code from audioset_tagging_cnn at
commit d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4 (pytorch/models.py, pytorch/pytorch_utils.py):

    uv venv --python 3.12 <tmp>/.venv
    uv pip install --python <tmp>/.venv/Scripts/python.exe \
        --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
    uv pip install --python <tmp>/.venv/Scripts/python.exe onnx==1.23.2 onnxscript==0.7.2 \
        numpy==2.5.4 librosa==0.11.0
    uv pip install --python <tmp>/.venv/Scripts/python.exe --no-deps torchlibrosa==0.1.0
    <tmp>/.venv/Scripts/python.exe -I -X utf8 tools/export_panns.py <code dir> \
        MobileNetV1_mAP=0.389.pth <out_dir> [32 kHz mono float32 .npy files ...]

Then install with `uv run python tools/install_panns.py <out_dir>`.

The graph starts after the STFT/log-mel layers (re-implemented in NumPy in
xsample.analysis.tagger) and ends with the sigmoid; the forward pass is MobileNetV1.forward in
eval mode, checked here against the complete torch model. Reference outputs of the complete
model are written to golden.npz for parity checks. numba is replaced by a no-op module:
librosa only applies its decorators at import, and the STFT and mel weights come from the
checkpoint (Windows Application Control blocks numba's DLLs on the reference machine).
"""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_SHA256 = "5f1d094310e439620b63d377e89e69bc663b782f1c242c0695ceb5db6db4271f"
CONFIG = dict(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527,
)
FILE, OPSET = "panns_mobilenetv1.onnx", 18


def _fake_numba() -> None:
    def decorator(*args, **kwargs):  # type: ignore[no-untyped-def]
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f

    mod = types.ModuleType("numba")
    for name in ("jit", "njit", "stencil", "guvectorize", "vectorize", "generated_jit"):
        setattr(mod, name, decorator)

    def attr(name: str) -> object:
        if name.startswith("__"):
            raise AttributeError(name)
        return object()  # e.g. numba.uint in signatures that are never compiled

    mod.__getattr__ = attr  # type: ignore[method-assign]
    sys.modules["numba"] = mod


class Head(torch.nn.Module):
    """MobileNetV1.forward (eval) after the log-mel extractor: (1, frames, 64) -> (1, 527)."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.m = model

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        x = logmel[:, None, :, :].transpose(1, 3)
        x = self.m.bn0(x).transpose(1, 3)
        x = torch.mean(self.m.features(x), dim=3)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)
        x = torch.relu(self.m.fc1(x))
        return torch.sigmoid(self.m.fc_audioset(x))


def main() -> None:
    code, ckpt, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    if hashlib.sha256(ckpt.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        sys.exit("checkpoint does not match CHECKPOINT_SHA256")
    _fake_numba()
    sys.path.insert(0, str(code))
    from models import MobileNetV1  # type: ignore[import-not-found]

    model = MobileNetV1(**CONFIG)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)["model"]
    # The STFT kernels and mel matrix are checkpoint tensors, not librosa output.
    assert "logmel_extractor.melW" in state
    assert "spectrogram_extractor.stft.conv_real.weight" in state
    model.load_state_dict(state)
    model.eval()
    head = Head(model).eval()

    rng = np.random.default_rng(0)
    sr = CONFIG["sample_rate"]
    t = np.arange(sr * 10) / sr
    refs = {
        "tone440": 0.3 * np.sin(2 * np.pi * 440 * t),
        "noise": 0.1 * rng.standard_normal(t.size),
        "chord": 0.2 * sum(np.sin(2 * np.pi * f * t) for f in (261.6, 329.6, 392.0)),
        "clicks": np.where((t * 2) % 1 < 0.01, 0.8, 0.0) + 0.01 * rng.standard_normal(t.size),
        "short": 0.3 * np.sin(2 * np.pi * 660 * t[: sr * 3]),
    }
    for extra in sys.argv[4:]:
        refs[Path(extra).stem] = np.load(extra)[: sr * 10]
    golden = {}
    out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for name, wav in refs.items():
            w = torch.tensor(wav, dtype=torch.float32)[None]
            lm = model.logmel_extractor(model.spectrogram_extractor(w))[:, 0]
            full = model(w)["clipwise_output"]
            assert torch.allclose(full, head(lm), atol=1e-6), name
            golden[f"{name}_wav"] = w[0].numpy()
            golden[f"{name}_logmel"] = lm[0].numpy()
            golden[f"{name}_probs"] = full[0].numpy()
    np.savez_compressed(out / "golden.npz", **golden)
    np.save(out / "melW.npy", model.logmel_extractor.melW.numpy())
    example = torch.tensor(golden["tone440_logmel"])[None]
    torch.onnx.export(
        head,
        (example,),
        str(out / FILE),
        input_names=["logmel"],
        output_names=["probs"],
        dynamic_shapes={"logmel": {1: torch.export.Dim("frames", min=32, max=6001)}},
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
    )
    data = (out / FILE).read_bytes()
    print(FILE, len(data), hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main()
