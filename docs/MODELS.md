# Models: installed and candidates

Nothing in this file is downloaded automatically. Downloads need explicit approval
(CLAUDE.md, ADR-0001); weights live in `%LOCALAPPDATA%\X-Sample\models`, never in Git.

## Installed (from verified local files)

| Model | Use | Weights license | Files | Notes |
|---|---|---|---|---|
| Beat This! small0 (CPJKU) | beats, downbeats | MIT (code + weights); some training audio copyrighted / limited CC | 2 ONNX exports, 12.1 + 10.4 MB, SHA-256 pinned in `beat_model.PINNED` | ADR-0012. Trained on GuitarSet (+15 sets), GTZAN held out. |
| PANNs MobileNetV1 (Kong et al.) | instrument, vocal and genre tags | **CC BY 4.0** (weights, Zenodo 3987831); code MIT (`audioset_tagging_cnn` d2f4b8c); torchlibrosa 0.1.0 MIT (export only) | 1 ONNX export, 19.4 MB, SHA-256 pinned in `tagger.PINNED`; checkpoint SHA-256 `5f1d0943…271f` | ADR-0013. Trained on AudioSet (YouTube audio; labels CC BY 4.0). Accuracy and limits: `docs/EVALUATION.md`. |

### Attribution (CC BY 4.0)

PANNs MobileNetV1 weights: Q. Kong, Y. Cao, T. Iqbal, Y. Wang, W. Wang, M. D. Plumbley,
"PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition", IEEE/ACM
TASLP 28 (2020), 2880–2894. File `MobileNetV1_mAP=0.389.pth` from
https://zenodo.org/records/3987831, licensed CC BY 4.0; converted to ONNX (graph after the
log-mel front end) without changing the weights. The same text is shipped with every analysis
result (`tagger.license`, `tagger.attribution`). Redistributing the ONNX file with the
application requires keeping this attribution and the license notice.

## Candidates for genre, instrument and vocal detection (research 2026-10-29)

Priority classes: R&B, piano, violin, saxophone; broad genre and instrument coverage kept.
AudioSet's ontology covers all three tasks in one model, for example genres "Rhythm and
blues", "Soul music", "Funk", "Jazz", "Classical music", "Pop music", "Hip hop music";
instruments "Piano", "Violin, fiddle", "Saxophone", "Guitar", "Drum kit", "Bass guitar";
vocals "Singing", "Male singing", "Female singing", "Speech". AudioSet labels are CC BY 4.0;
the training audio is YouTube content (the same kind of caveat as Beat This!).

| Candidate | Weights license | Download | AudioSet mAP | Runtime path | Verdict |
|---|---|---|---|---|---|
| PANNs MobileNetV1 (Kong et al. 2020), Zenodo 3987831 | **CC BY 4.0** (attribution); code MIT | 23.6 MB | 0.389 | PyTorch → ONNX in the isolated export venv; 32 kHz log-mel front end re-implemented in NumPy | **Installed** (approved 2026-10-29, ADR-0013) |
| PANNs Cnn14, same record | **CC BY 4.0** | 327 MB | 0.431 | same | Not approved; candidate if MobileNetV1's recall is not enough |
| YAMNet (Google) | **Apache-2.0** (TensorFlow Model Garden; confirmed on the audioset-users list, Kaggle card not checked directly) | ~15 MB Keras H5 | ~0.31 | needs TensorFlow (Keras 2) in a temporary venv for a TF → ONNX conversion | Usable, lower accuracy, heavier conversion |
| EfficientAT mn10_as / dymn10_as (Schmid et al.) | repository MIT; **weights' license not stated** in the release | ~20 / 40 MB | 0.471 / 0.477 | PyTorch → ONNX | **Unresolved**: best accuracy per size, but not downloaded until the author confirms the weights' license |
| Essentia / MTG models (Discogs-EffNet genres, MTG-Jamendo instruments, voice/instrumental) | **CC BY-NC-SA 4.0** | small | — | TF / ONNX | **Excluded**: non-commercial (proprietary license on request) |
| LAION-CLAP music checkpoints (zero-shot genre) | not checked in detail | ~0.6–2 GB | — | — | Too large for a first step |

Expected cost on the reference machine (to be measured, not claimed): MobileNet-class
taggers are about 0.5 GMAC per second of audio, i.e. well under a second for a 3-minute
piece on CPU. DirectML would be verified with the same strict, profiled parity check as
ADR-0012.

### Evaluation plan (independent of the model's training data)

- **Instruments:** BabySlakh/Slakh metadata (per-track instrument classes from MIDI:
  piano, strings, reeds incl. sax, brass, guitar, bass, drums); MusicNet (piano, violin);
  GuitarSet (guitar). None of these are AudioSet clips. Per-class precision/recall and the
  threshold used are reported, as the v0.2.0 acceptance criteria require.
- **Genre:** GuitarSet styles (funk, jazz, rock, bossa nova, singer-songwriter) and MusicNet
  (classical) only give coarse labels; R&B needs a licensed, labelled held-out set (none
  approved yet). Results are reported per class with their sample sizes.
- **Vocals:** none of the current subsets contain singing. The tagger reports singing,
  rapping and speech, but only their false-positive rate on instrumental audio is measured;
  recall needs a licensed set with singing (not approved yet).

Results of this plan (2026-10-29) are in `docs/EVALUATION.md`.

### Held-out beat evaluation data

Beat This! was trained on ASAP, Ballroom, Beatles, Candombe, Filosax, Groove MIDI,
GuitarSet, Hainsworth, Harmonix, HJDB, JAAH, RWC, Simac, SMC and TapCorrect (GTZAN is its
test set; GTZAN has no clear license and is not used). Held-out, commercially licensed
options:

- **BabySlakh, complete** (CC BY 4.0, Zenodo 4603870): fetched 2026-10-29 with approval —
  the 883 MB archive was streamed and verified (MD5), keeping mixes, MIDI and metadata of all
  20 tracks (152 MB).
- **MusicNet, more recordings** (CC BY 4.0): 11.1 GB archive, streamed prefix only (we hold
  400 MB / 10 recordings). Its beat references follow the notated beat (eighths in compound
  meters), so it measures metrical-level agreement with the score as much as tracking.
