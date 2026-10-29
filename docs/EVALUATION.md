# Analysis evaluation (v0.2.0)

How the musical analysis is measured, on what data, and the current results. Synthetic and
real-audio results are reported **separately**: the synthetic numbers show that the analyzers
work as designed on clean material; only the real-audio numbers say how they behave on music.

## Evaluation data

All datasets live outside the repository in `%LOCALAPPDATA%\X-Sample\eval-datasets` and are
never shipped with the application. Each folder has a `MANIFEST.json` with the source URL, DOI,
license, attribution and the SHA-256 of every kept file. Only small subsets were fetched.

| Dataset | Source / license | Fetched | Annotations used |
|---|---|---|---|
| GuitarSet 1.1.0 | Zenodo 10.5281/zenodo.3371780, CC BY 4.0 (Xi et al., ISMIR 2018) | `annotation.zip` in full (39.1 MB, MD5 verified); 48 mono-mic takes (players 00 and 03, Funk and Jazz, comp and solo) pulled from `audio_mono-mic.zip` by HTTP range requests (77.7 MB, per-member CRC-32 verified) | beats/downbeats, tempo, key, performed chords, notes |
| BabySlakh (16 kHz) | Zenodo 10.5281/zenodo.4603870, CC BY 4.0 (Manilow et al., WASPAA 2019) | First 150 MB of the 883 MB archive streamed and stopped: 2 complete tracks (Track00018, Track00020); mix, MIDI and metadata kept, stems dropped | tempo map and meter from MIDI; instrument classes listed (not scored) |
| MusicNet 1.0 | Zenodo 10.5281/zenodo.5120004, CC BY 4.0 (Thickstun et al., ICLR 2017) | First 400 MB of the 11.1 GB archive streamed and stopped; kept 10 recordings of the solo piano / violin ensembles with their labels: 8 solo piano, 1 solo violin, 1 violin + piano | note labels (beat positions, violin notes); movement key for 7 recordings |
| Medley-solos-DB 1.2 | Zenodo 10.5281/zenodo.3464194, CC BY 4.0 | **Not fetched.** One 7.9 GB `.tar.gz`; the piano, tenor sax and violin test clips start 2.2–3.4 GB into the stream, so no small selective download exists. Needs approval. | — |
| MAESTRO, FMA medium | — | Not fetched (by decision) | — |

Notes:
- MusicNet keys come from the movement (e.g. Op. 14/2 i in G major), not only the work title;
  movements whose key differs from the title or is ambiguous are not key-scored.
- MusicNet labels have a reported ~4 % error rate; GuitarSet chords are the performed (not the
  lead-sheet) annotation, mapped to our five qualities; chords outside maj/min classes (dim,
  hdim, aug, sus) are excluded from chord scores.
- The melody tracker only searches from C3 upwards; reference notes below C3 are excluded.

Reproduce:

```powershell
cd engine
uv run python tools/bench_synthetic.py      # -> test-reports/synthetic_bench.json
uv run python tools/eval_real_audio.py      # -> test-reports/real_audio_eval.json (needs the local datasets)
```

Metrics: tempo Acc1 (±4 %) and Acc2 (also ×2, ×½, ×3, ×⅓); beat/downbeat F-measure (±70 ms,
first 5 s skipped); MIREX key score (1 / 0.5 fifth / 0.3 relative / 0.2 parallel); chord
overlap at 50 ms steps (maj/min classes, and the full five qualities); melody note accuracy
(reference notes whose middle falls on an estimated note within ±0.5 semitone).

## Results — DSP analyzers (CPU), 2026-10-29

### Synthetic (13 generated pieces with exact ground truth)

Pop, minor, noisy, slow, fast, tempo drift, 3/4, drumless piano, R&B with sevenths and
inversions, R&B with sax lead, R&B with violin lead and sections, violin ballad, noisy sax.

| Acc1 | Acc2 | Beat F | Downbeat F | Key (MIREX) | Chord maj/min | Chord full | Lead melody |
|---|---|---|---|---|---|---|---|
| 0.77 | 1.00 | 0.92 | 0.92 | 0.91 | 0.82 | 0.66 | 0.81 |

Failures: the 72, 174 and 3/4-at-150 BPM pieces are detected at the octave (Acc2 only), which
also breaks their beat F (~0.65) and the waltz's chords and key; sevenths under a lead line; the
drumless violin ballad's melody (0.25).

### Real audio (licensed subsets above)

| Subset | n | Acc1 | Acc2 | Beat F | Downbeat F | Key | Chord maj/min | Chord full | Melody |
|---|---|---|---|---|---|---|---|---|---|
| GuitarSet Funk comp | 12 | 0.50 | 0.50 | 0.62 | 0.45 | 0.72 | 0.49 | 0.26 | — |
| GuitarSet Funk solo | 12 | 1.00 | 1.00 | 0.57 | 0.29 | 0.43 | — | — | 0.51 |
| GuitarSet Jazz comp | 12 | 0.42 | 0.83 | 0.46 | 0.30 | 0.48 | 0.30 | 0.19 | — |
| GuitarSet Jazz solo | 12 | 0.58 | 0.92 | 0.51 | 0.23 | 0.32 | — | — | 0.57 |
| MusicNet solo piano | 8 | 0.13 | 0.50 | 0.48 | — | 0.88 (6 scored) | — | — | — |
| MusicNet solo violin | 1 | 0.00 | 1.00 | 0.45 | — | 0.30 | — | — | 0.39 (violin) |
| MusicNet violin + piano | 1 | 0.00 | 0.00 | 0.54 | — | — | — | — | 0.20 (violin) |
| BabySlakh (MIDI-rendered band) | 2 | 0.50 | 0.50 | 0.56 | 0.38 | — | — | — | — |

**Against the v0.2.0 acceptance targets** (tempo Acc2 ≥ 95 %, beat F ≥ 0.85, key ≥ 80 %, chord
maj/min ≥ 70 %), the DSP analyzers pass on synthetic material but **fail on real audio** except
for key on classical piano. v0.2.0 is therefore not complete.

Diagnosed causes (not tuned away, to avoid fitting this small subset):
- Funk comping: tempo locks to a 4:3-related rate (114→150, 97→130, 108→143 BPM); dense
  16th-note strumming makes a dotted-beat period look strong.
- Solo guitar: tempo right but beats on the off-beat (one Funk solo: F 0.0, 0.96 when shifted
  by half a beat); the phase check relies on kick/bass energy that solo instruments lack.
- Classical: expressive tempo; beat tracking assumes a slowly varying tempo.
- Chords: real voicings (9ths, 13ths, altered dominants, rootless jazz voicings) and guitar
  string resonance differ from triad/seventh templates.
- Melody: guitar and violin solos have fast passages, double stops and notes below C3.

Instrument and genre recognition, and vocal activity, are **not evaluated**: they need models
that have not been approved. The UI reports them as unavailable.

### Resources

10-minute R&B fixture through the real worker (CPU): 12.7 s analysis, 216 MiB peak worker
memory (+24 MiB vs a 60 s file), 712 KiB sidecar. 30–45 s pieces analyse in 0.3–0.9 s.

## Results — Beat This! small0 beats and downbeats (ADR-0012), 2026-10-29

`uv run python tools/eval_real_audio.py --beat-model --backend directml` scores the installed,
pinned ONNX exports through the engine's `BeatTracker`, next to the DSP tracker on the same
files. Model tempo is 60 / median beat interval. 32 items ran on DirectML; the 28 shorter than
30 s ran the dynamic-length export on CPU. A CPU-only run gives identical scores on every item.

**Training-data overlap.** Beat This! was trained on 16 datasets including **GuitarSet**
(CPJKU/beat_this_annotations), so its GuitarSet scores are *not* a held-out measurement and
must not be read as expected real-world accuracy. MusicNet and Slakh/BabySlakh are not in its
training data and are the held-out results. Nothing was tuned on any of these subsets.

`F any level` is a secondary metric: the best beat F against the reference at ×½, ×1 and ×2
metrical level, because MusicNet's reference beats follow the notated beat (e.g. eighths at
225 BPM, halves at 44 BPM). The primary metric is plain beat F.

| Subset | Held out? | n | Beat F (DSP → model) | F any level | Downbeat F | Acc1 | Acc2 |
|---|---|---|---|---|---|---|---|
| MusicNet solo piano | yes | 8 | 0.48 → 0.62 | 0.53 → 0.63 | — | 0.13 → 0.38 | 0.50 → 0.38 |
| MusicNet solo violin | yes | 1 | 0.45 → 0.43 | 0.45 → 0.56 | — | 0.00 → 0.00 | 1.00 → 1.00 |
| MusicNet violin + piano | yes | 1 | 0.54 → 0.82 | 0.55 → 0.82 | — | 0.00 → 1.00 | 0.00 → 1.00 |
| **MusicNet (all)** | yes | 10 | **0.48 → 0.62** | 0.52 → 0.64 | — | 0.10 → 0.40 | 0.50 → 0.50 |
| **BabySlakh** | yes | 2 | **0.56 → 0.53** | 0.60 → 0.66 | 0.38 → 0.41 | 0.50 → 0.00 | 0.50 → 0.50 |
| GuitarSet Funk comp | **no (trained on)** | 12 | 0.62 → 0.94 | 0.70 → 0.94 | 0.45 → 0.90 | 0.50 → 1.00 | 0.50 → 1.00 |
| GuitarSet Funk solo | **no** | 12 | 0.57 → 0.62 | 0.72 → 0.70 | 0.29 → 0.33 | 1.00 → 1.00 | 1.00 → 1.00 |
| GuitarSet Jazz comp | **no** | 12 | 0.46 → 0.81 | 0.65 → 0.90 | 0.30 → 0.72 | 0.42 → 0.67 | 0.83 → 1.00 |
| GuitarSet Jazz solo | **no** | 12 | 0.51 → 0.70 | 0.74 → 0.81 | 0.23 → 0.45 | 0.58 → 0.50 | 0.92 → 1.00 |
| GuitarSet (all) | **no** | 48 | 0.54 → 0.77 | 0.70 → 0.84 | 0.32 → 0.60 | 0.63 → 0.79 | 0.81 → 1.00 |

Per-item notes (held out): the model fixes the accompanied-violin tempo (100.9 → 187.5 BPM vs
191.4) and most piano tempi, but chooses a different metrical level on some piano movements
(2390: 166.7 vs 224.7; 2392: 93.7 vs 43.8) and the solo violin (93.7 vs 187.9). BabySlakh
Track00018 is found at half tempo (69.8 vs 137.0; F 0.75, downbeat F 0.76); Track00020 fails
for both trackers (model 176.5 vs 120.0, F 0.31).

**Against the v0.2.0 targets** (Acc2 ≥ 95 %, beat F ≥ 0.85): still not met on held-out real
audio. The remaining gaps are metrical-level choice (no tempo continuity or prior in the
minimal postprocessor) and expressive classical timing. The held-out sample is small (12
items), so these figures have wide uncertainty.

### Model resources (reference machine)

| | CPU (Ryzen 7 3700X) | DirectML (RX 5700 XT) |
|---|---|---|
| One 30 s chunk | 720 ms | 115 ms |
| Beat model, 10 min of audio (in the worker) | 15.1 s | 4.1 s |
| Whole analysis, 10 min | 26.7 s | ~16 s |
| Real-audio eval, 73 min total | 111 s | 31–33 s |
| Peak dedicated VRAM | — | ~570 MiB |
| Worker peak memory, 10 min (60 s → 600 s) | 1025 → 1058 MiB | 1661 → 1690 MiB |
| Max logit difference vs torch / vs CPU | 1e-4 | 1e-4 |

The DirectML figure includes one extra CPU run of the first chunk for the parity check.
