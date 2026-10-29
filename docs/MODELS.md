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

## Accuracy candidates — key, chords, beats, vocals, R&B (research 2026-10-29)

Context: the accuracy investigation in `docs/EVALUATION.md` ("Accuracy investigation"). Nothing
below has been downloaded. Licenses were read from the projects' own pages where stated;
entries marked *unverified* need checking before any approval.

### Models

| Candidate | Task | License | Training data (provenance) | Download / runtime | Fits Windows + 3700X + RX 5700 XT? | Expected benefit / limits | Independent of our eval data? |
|---|---|---|---|---|---|---|---|
| Beat This! **final0** (CPJKU) | beats, downbeats | MIT (code + weights, same as small0) | same 16 sets as small0 (incl. GuitarSet, Harmonix, RWC; YouTube/commercial audio in some) | ~80 MB (20.3 M params) checkpoint → ONNX via the existing isolated export | Yes: same graph family as small0, same verification path; several times the compute per chunk (to be measured) | The paper reports a somewhat higher GTZAN F for the full model than for the small one; may close part of the 0.74 → 0.85 held-out beat gap. Does not address metrical level (2:3) by itself | MusicNet, BabySlakh: yes. GuitarSet: no |
| CREMA chord model (McFee & Bello 2017) | chords (602 classes), structured training | BSD-2-Clause (repo LICENSE) | Isophonics, Billboard, RWC-Pop, uspop (~1200 commercial pop/rock tracks, annotations only) | small (a few MB of weights; *size unverified*); Keras/TensorFlow — would need a TF → ONNX conversion in a temporary venv | Yes after ONNX conversion (small conv + GRU; DirectML op support unverified) | Learned features handle extended voicings far better than templates; trained on pop, not jazz guitar | GuitarSet, Winterreise: yes |
| BTC (Park et al., ISMIR 2019) | chords (majmin or 170 classes) | code MIT; **weights' license not stated** upstream (a third-party conversion is CC BY-NC-SA) | Isophonics, Robbie Williams, UsPop2002 | ~12 MB *unverified*; PyTorch → ONNX | Yes (transformer, small) | Similar to CREMA; bidirectional attention gives stable segments | GuitarSet: yes. **Blocked until the weights' license is confirmed** |
| NNLS-chroma-style front end (Mauch & Dixon 2010), re-implemented | chroma for key/chords | algorithm (paper); the Vamp plugin is GPL — do not copy code | none (no training) | no download; NumPy | Yes | Tuning estimation, log-frequency spectrogram, note-salience by NNLS against harmonic templates — reduces partials and fixed resonances leaking into chroma. Moderate effort; gains uncertain | n/a |
| madmom / Essentia chord, key, beat models | — | **non-commercial** (CC BY-NC-SA) | — | — | — | **Excluded** | — |

### Datasets

| Dataset | Content | License / commercial use | Size, selective fetch | What it would validate | Independence |
|---|---|---|---|---|---|
| **Schubert Winterreise** v2.1 (Zenodo 10839767) | 24 songs; **baritone + piano**; audio of 2 of 9 performances included (one CC BY 3.0 from the Isabella Stewart Gardner Museum, no modified redistribution; one public domain) | CC BY 3.0 (attribution) | one 517 MB zip; members can be range-fetched (we only need audio + chord/key/measure CSVs) | **singing recall** (voice present throughout), piano, global/local **key**, **chords**, measures (downbeats) | Not in Beat This!, AudioSet (as far as known) or CREMA/BTC training. Classical Lied only — not R&B |
| **OpenMIC-2018** (Zenodo 1432913) | 20,000 × 10 s FMA excerpts, crowd labels for 20 instruments incl. **voice, saxophone, piano, violin** | CC BY 4.0 (dataset); audio under each track's CC license (some NC) | one 2.4–2.6 GB `.tgz` (gzip: streamed from the start only) | instrument and **voice** recall/precision on produced, mixed music; genre via FMA metadata (incl. Soul-RnB) | FMA audio is not AudioSet; YouTube re-uploads cannot be ruled out |
| FMA metadata + Soul-RnB subset | genre labels from uploaders; Soul-RnB is a top-level FMA genre | metadata CC BY 4.0; audio per-track CC (filter to commercial-compatible licenses) | `fma_metadata.zip` (~342 MB) + selected tracks from `fma_large`/`fma_full` zips by range request | **R&B genre** precision/recall (noisy, uploader-assigned labels) | as above |
| Medley-solos-DB 1.2 (Zenodo 3464194) | 3 s solo clips: piano, violin, tenor sax, **female singer**, … | CC BY 4.0 | one 7.9 GB `.tar.gz`, no selective fetch | solo instrument / singing recall (not mixtures) | derived from MedleyDB (some tracks may be on YouTube) |
| GuitarSet **hex-pickup mix** audio | the same 48 takes we hold, recorded via pickup instead of the mic | CC BY 4.0 | range-fetch from `audio_mono-pickup_mix.zip` | separates the mic/body-resonance bias from the analyzer (see EVALUATION) | same items — diagnostic only, not new evidence |
| Harmonix Set, GTZAN, Isophonics, Billboard, MUSDB18, MedleyDB, MoisesDB, IRMAS | — | audio copyrighted, no clear license, or non-commercial | — | — | Harmonix and others are also in Beat This!'s training set. **Not proposed** |
