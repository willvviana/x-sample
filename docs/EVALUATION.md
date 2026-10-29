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
| BabySlakh (16 kHz) | Zenodo 10.5281/zenodo.4603870, CC BY 4.0 (Manilow et al., WASPAA 2019) | Complete archive streamed (882,818,115 bytes, MD5 from Zenodo verified): all 20 tracks; mix, `all_src.mid`, per-stem MIDI and metadata kept (152 MB), audio stems dropped | tempo map and meter from MIDI; instruments from stem programs and MIDI notes (tagging) |
| MusicNet 1.0 | Zenodo 10.5281/zenodo.5120004, CC BY 4.0 (Thickstun et al., ICLR 2017) | First 400 MB of the 11.1 GB archive streamed and stopped; kept 10 recordings of the solo piano / violin ensembles with their labels: 8 solo piano, 1 solo violin, 1 violin + piano | note labels (beat positions, violin notes); movement key for 7 recordings |
| Medley-solos-DB 1.2 | Zenodo 10.5281/zenodo.3464194, CC BY 4.0 | **Not fetched.** One 7.9 GB `.tar.gz`; the piano, tenor sax and violin test clips start 2.2–3.4 GB into the stream, so no small selective download exists. Needs approval. | — |
| MAESTRO, FMA medium | — | Not fetched (by decision) | — |

Notes:
- MusicNet keys come from the movement (e.g. Op. 14/2 i in G major), not only the work title;
  movements whose key differs from the title or is ambiguous are not key-scored.
- MusicNet labels have a reported ~4 % error rate; GuitarSet chords are the performed (not the
  lead-sheet) annotation. Since 2026-10-29 they are reduced like mir_eval's `majmin`: the
  chord's tones within the octave after extensions and omissions must form exactly a major or
  minor triad (`B:7(#9)` → B major, `D:(5,2,b7,4)` → excluded); 9ths, 11ths and 13ths are
  ignored; dim, hdim, aug, sus, power chords and chords without a third or fifth are excluded.
  Before that, interval-only chords without a third were mapped to major (an evaluator bug).
- The melody tracker only searches from C3 upwards; reference notes below C3 are excluded.

Reproduce:

```powershell
cd engine
uv run python tools/bench_synthetic.py      # -> test-reports/synthetic_bench.json
uv run python tools/eval_real_audio.py      # -> test-reports/real_audio_eval.json (needs the local datasets)
uv run python tools/eval_real_audio.py --beat-model --backend directml   # + Beat This! fields
uv run python tools/eval_tagger.py --backend directml   # -> test-reports/tagger_eval.json
```

Metrics: tempo Acc1 (±4 %) and Acc2 (also ×2, ×½, ×3, ×⅓); beat/downbeat F-measure (±70 ms,
first 5 s skipped); MIREX key score (1 / 0.5 fifth / 0.3 relative / 0.2 parallel); chord
overlap at 50 ms steps (maj/min classes, and the full five qualities); melody note accuracy
(reference notes whose middle falls on an estimated note within ±0.5 semitone).

## Results — DSP analyzers (CPU), 2026-10-29 (analyzer `dsp-0.2.0`)

The tables in this section are the first measurement, kept for reference; the current
numbers are in "Accuracy investigation" at the end.

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
files. Model tempo is 60 / median beat interval. Re-run with the complete BabySlakh set: 50
items ran on DirectML; the 28 shorter than 30 s ran the dynamic-length export on CPU. A
CPU-only run gave identical scores on every item (earlier run, 60 items).

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
| **BabySlakh** | yes | 20 | **0.70 → 0.80** | 0.79 → 0.82 | 0.51 → 0.63 | 0.70 → 0.75 | 0.90 → 0.80 |
| **Held out, pooled** | yes | 30 | **0.63 → 0.74** | — | — | 0.50 → 0.63 | 0.77 → 0.70 |
| GuitarSet Funk comp | **no (trained on)** | 12 | 0.62 → 0.94 | 0.70 → 0.94 | 0.45 → 0.90 | 0.50 → 1.00 | 0.50 → 1.00 |
| GuitarSet Funk solo | **no** | 12 | 0.57 → 0.62 | 0.72 → 0.70 | 0.29 → 0.33 | 1.00 → 1.00 | 1.00 → 1.00 |
| GuitarSet Jazz comp | **no** | 12 | 0.46 → 0.81 | 0.65 → 0.90 | 0.30 → 0.72 | 0.42 → 0.67 | 0.83 → 1.00 |
| GuitarSet Jazz solo | **no** | 12 | 0.51 → 0.70 | 0.74 → 0.81 | 0.23 → 0.45 | 0.58 → 0.50 | 0.92 → 1.00 |
| GuitarSet (all) | **no** | 48 | 0.54 → 0.77 | 0.70 → 0.84 | 0.32 → 0.60 | 0.63 → 0.79 | 0.81 → 1.00 |

Per-item notes (held out): the model fixes the accompanied-violin tempo (100.9 → 187.5 BPM vs
191.4) and most piano tempi, but chooses a different metrical level on some piano movements
(2390: 166.7 vs 224.7; 2392: 93.7 vs 43.8) and the solo violin (93.7 vs 187.9). On BabySlakh
the model scores F ≥ 0.82 on 15 of 20 tracks; failures: Track00009 (76.9 vs 120 BPM, F 0.21),
Track00020 (176.5 vs 120, F 0.31), Track00013 (correct tempo, F 0.26) and Track00012 (correct
tempo, F 0.00 for every tracker, which suggests an offset between the MIDI reference and the
render rather than a tracking error; it is kept, not removed). Track00018 is at half tempo
(69.8 vs 137.0). The model's Acc2 on BabySlakh (0.80) is below DSP's (0.90): Track00009 and
Track00020 are 2:3-type errors that DSP avoids.

**Against the v0.2.0 targets** (Acc2 ≥ 95 %, beat F ≥ 0.85): still not met on held-out real
audio. The remaining gaps are metrical-level choice (no tempo continuity or prior in the
minimal postprocessor) and expressive classical timing. The held-out sample (30 items: 10
MusicNet, 20 synthesised BabySlakh renders) is still small, so these figures have wide
uncertainty.

### Tempo-continuous decoding (DBN) — evaluated, not adopted

`analysis/beat_dbn.py` is a NumPy bar-pointer HMM with Viterbi decoding (Krebs et al. 2015,
Böck et al. 2016; no madmom code), run on the same model logits with beat_this's DBN
defaults (55–215 BPM, 3 or 4 beats per bar, transition λ 100, observation λ 16, peak
correction). No parameter was changed after seeing results.

| Subset | Held out? | Beat F minimal → DBN | Acc1 | Acc2 | Downbeat F |
|---|---|---|---|---|---|
| MusicNet (10) | yes | **0.62 → 0.54** | 0.40 → 0.20 | 0.50 → 0.50 | — |
| BabySlakh (20) | yes | 0.80 → 0.80 | 0.75 → 0.80 | 0.80 → 0.85 | 0.63 → 0.62 |
| Held out, pooled (30) | yes | **0.74 → 0.71** | 0.63 → 0.60 | 0.70 → 0.73 | — |
| GuitarSet (48) | no | 0.77 → 0.77 | 0.79 → 0.73 | 1.00 → 0.96 | 0.60 → 0.57 |

With the complete BabySlakh set the DBN is equal on BabySlakh and worse on MusicNet, so it
is still worse on held-out audio overall (pooled beat F 0.74 → 0.71). On MusicNet this is mainly because the 215 BPM ceiling forces fast
references (MusicNet 2628 at 191 BPM, 2633 at 215 BPM) to half tempo, which the minimal
postprocessor got right. Widening the range to fit these items would be tuning to the
evaluation set, so the minimal postprocessor stays the default; the DBN remains available
to `tools/eval_real_audio.py` (fields `dbn_*`) for future, larger held-out sets. It decodes
10 minutes in about 3 s on the reference CPU.

Remaining held-out tempo errors are mostly 2:3 and 3:4 relations (MusicNet 2390: 166.7 vs
224.7; 2556: 150 vs 224.7), consistent with compound meters whose reference counts eighth
notes, and one 3:2 error on a pop/rock render (BabySlakh Track00020). Better held-out data is
needed before the metrical level can be improved without fitting this subset
(`docs/MODELS.md`).

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

## Results — PANNs MobileNetV1 instrument, vocal and genre tagging (ADR-0013), 2026-10-29

`uv run python tools/eval_tagger.py --backend directml` runs the installed, pinned export
through the engine's `Tagger` (same NumPy front end, 10 s windows with a 5 s hop, label
mapping and thresholds as the analysis job) on all 78 local items; every item ran on
DirectML. The thresholds (detected ≥ 0.5, uncertain ≥ 0.2), the AudioSet → X-Sample label
mapping and the two decision rules below were fixed before the run and were not changed
afterwards. None of these datasets is part of AudioSet (the model's training data); whether a
MusicNet or GuitarSet recording also appears in some YouTube upload cannot be ruled out.

Ground truth (details in the tool's docstring): BabySlakh instruments from the stems' General
MIDI programs (a stem counts when its MIDI file has notes; programs that only partly match a
label, e.g. synth strings for Strings, are skipped for that label); MusicNet instruments from
its note labels and genre Classical; GuitarSet: guitar only, style code as genre (this subset
holds Funk and Jazz only). **No item contains singing, rapping or speech**, so vocal labels are
scored on negatives only; **no item is labelled R&B, Soul, Pop, Hip hop or any other genre
than Classical, Funk and Jazz**.

Decision rules (fixed beforehand): a label is *evaluated* when ≥ 5 labelled pieces have it and
≥ 5 do not (pooled); a genre may be reported as *detected* only with ≥ 10 pieces each way,
precision ≥ 0.8 and recall ≥ 0.5 at the detect threshold. Result: evaluated = Piano, Organ,
Synthesizer, Guitar, Bass, Drums, Strings, Flute, Classical, Funk, Jazz; **no genre meets the
evidence rule, so every genre is shown as uncertain at most.**

Piece level (a label is predicted when its highest window score reaches the threshold).
P = precision, R = recall, FPR = false-positive rate, AUC = threshold-free ranking quality;
"—" = undefined (no predictions or no positives).

| Label | Data (pos/neg pieces) | @0.5 P / R | @0.2 P / R | FPR @0.5 | AUC |
|---|---|---|---|---|---|
| **Piano** | MusicNet (9/1) | 1.00 / 0.89 | 1.00 / 1.00 | 0.00 | 1.00 |
| Piano | BabySlakh (19/0) | 1.00 / 0.05 | 1.00 / 0.79 | — | — |
| Piano | pooled (28/49) | 1.00 / 0.32 | 1.00 / 0.86 | 0.00 | 1.00* |
| **Violin** | MusicNet (2/8) | 1.00 / 1.00 | 0.67 / 1.00 | 0.00 | 1.00 |
| Violin | pooled (2/62) — *not evaluated (< 5 positives)* | 1.00 / 1.00 | 0.67 / 1.00 | 0.00 | 1.00 |
| **Saxophone** | BabySlakh (4/16) — *not evaluated* | — / 0.00 | 0.50 / 0.25 | 0.00 | 0.78 |
| Strings | MusicNet (2/8) | 1.00 / 1.00 | 0.67 / 1.00 | 0.00 | 1.00 |
| Strings | BabySlakh (8/6) | — / 0.00 | — / 0.00 | 0.00 | 0.48 |
| Guitar | GuitarSet (48/0) | 1.00 / 1.00 | 1.00 / 1.00 | — | — |
| Guitar | BabySlakh (20/0) | 1.00 / 0.15 | 1.00 / 0.60 | — | — |
| Guitar | pooled (68/10) | 1.00 / 0.75 | 0.98 / 0.88 | 0.00 | 0.95* |
| Drums | BabySlakh (20/0); pooled (20/58) | 1.00 / 0.05 | 1.00 / 0.20 | 0.00 | 0.97* |
| Bass | BabySlakh (19/1); pooled (19/59) | — / 0.00 | — / 0.00 | 0.00 | 0.37 |
| Organ | BabySlakh (5/15) | 0.00 / 0.00 | 0.42 / 1.00 | 0.27 | 0.69 |
| Synthesizer | BabySlakh (9/4) | — / 0.00 | 0.50 / 0.11 | 0.00 | 0.69 |
| Brass | BabySlakh (4/16) — *not evaluated* | 1.00 / 0.50 | 1.00 / 0.75 | 0.00 | 0.80 |
| Flute | BabySlakh (6/14) | — / 0.00 | 0.67 / 0.33 | 0.00 | 0.85 |
| Cello, Clarinet | no positives (0/64, 0/78) | — | — | 0.00 | — |
| **Singing** | no positives (0/66 pieces, 0/1149 windows) | — | — | **0.00** | — |
| Rapping / Speech | no positives (0/78) | — | — | 0.00 / 0.013 | — |
| **R&B**, Soul, Pop, Hip hop, … | no positives (0/10, MusicNet only) | — | — | 0.00 | — |
| Classical | MusicNet (10/0); pooled (10/48) | — / 0.00 | — / 0.00 | 0.00 | 1.00* |
| Funk | GuitarSet (24/24) | — / 0.00 | — / 0.00 | 0.00 | 0.79 |
| Jazz | GuitarSet (24/24) | — / 0.00 | — / 0.00 | 0.00 | 0.37 |

\* Pooled AUC compares positives and negatives from *different datasets* (e.g. Piano
negatives are almost all GuitarSet), so it partly measures dataset separation; the
per-dataset rows are the fairer reading. Window-level figures (10 s windows, BabySlakh and
MusicNet note onsets) are in `test-reports/tagger_eval.json`; they follow the same pattern
(e.g. Piano window recall 0.27 at 0.5, Violin 0.55, no false positives for any label at 0.5
except Organ 0.6 %).

What this supports and what it does not:
- **Piano and violin on real acoustic recordings** are detected reliably at the conservative
  threshold, with no false positives in this set — but on 9 and 2 positive recordings only.
- **Synthesised BabySlakh renders score low** for most instruments (piano recall 0.05 at 0.5,
  0.79 at 0.2; drums 0.05 / 0.20; bass never). BabySlakh is 16 kHz audio (no content above
  8 kHz, the model sees up to 14 kHz) rendered from MIDI with sample libraries, which is far
  from AudioSet's YouTube audio; this is a domain gap, not necessarily the expected accuracy
  on produced music, but it is the only multi-instrument labelled data we have.
- **Saxophone: not validated** (4 synthetic positives, recall 0 at 0.5).
- **Singing / vocal activity: not validated.** Only the false-positive rate on instrumental
  material is measured (0 of 66 pieces, 0 of 1149 windows). Recall needs licensed recordings
  with singing.
- **Genres: not validated.** R&B and every genre except Classical/Funk/Jazz have no labelled
  data; Classical scores stay below 0.2 on all MusicNet recordings (perfect ranking, no
  detections); Funk/Jazz on solo guitar are never shown. Genre output is therefore labelled
  uncertain and the user is expected to correct it.
- The 0.5 operating point favours precision; recall is low outside piano, violin and guitar.
  The thresholds were not tuned to these results (that would be fitting the evaluation set).

### Tagger resources (reference machine)

| | CPU (Ryzen 7 3700X) | DirectML (RX 5700 XT) |
|---|---|---|
| One 10 s window | ~6–10 ms | ~1.4–2.3 ms |
| Tagger inference, 10 min (120 windows) | 1.0 s | 0.38 s |
| Tagging step in the analysis job, 10 min (32 kHz decode + log-mel + inference) | 2.1 s | 1.6 s |
| Peak dedicated VRAM | — | ~48 MiB |
| Process RSS (tagger alone, 10 min) | 147 MiB | 194 MiB |
| Analysis worker peak with beat model + tagger, 10 min (60 s → 600 s) | 1024 → 1056 MiB | 1638 → 1668 MiB |
| Max difference vs complete torch model / DirectML vs CPU | ≤ 1e-3 (pure tones), ≤ 1e-6 (real audio) | 3e-7 |

Front-end parity: the NumPy log-mel matches torchlibrosa within 3e-4 dB on bins above −40 dB
(≤ 0.03 dB on near-silent bins of pure tones, a float32 STFT effect); the mel matrix equals
the checkpoint's within 1e-9. The tagging step is dominated by the second (32 kHz) decode, so
DirectML saves little; it is used only after the same strict, profiled parity check as the
beat model.

## Accuracy investigation, 2026-10-29 (analyzer `dsp-0.2.1`)

Goal: find out why key, chords, beats and tempo miss the v0.2.0 targets on real audio before
assuming a larger model is needed. Nothing was downloaded; no threshold or acceptance target
was changed.

### Root causes found

1. **Evaluation misalignment (evaluator).** With `--beat-model`, `eval_real_audio.py` scored
   key and chords on harmony computed from the DSP beat grid, while the analysis job uses the
   model's grid when the model is installed (`run.py`). Fixed: key/chord fields now come from
   the production path; the DSP-grid result is kept as `dspgrid_key_mirex` and
   `dspgrid_chord_majmin` for comparison.
2. **Chord-label reduction (evaluator).** GuitarSet's performed chords are full jazz voicings
   (`B:7(#9)/1`, `D:(5,2,b7,4)/4`, `G:9(13,*1,*5)/b7`). The old mapping read only the shorthand
   and mapped interval-only chords without a third to major. Now a mir_eval-like `majmin`
   reduction (see "Notes" above). This changes the measured chord score without any analyzer
   change, so it is a **scoring correction, not an improvement**.
3. **Chord decoder too sticky (analyzer).** In the chord HMM, one change cost 5.8 nats
   (self-transition 0.85, 60 chord states), i.e. a summed cosine margin of ~0.49 at emission
   scale 12. On real voicings the median per-beat margin between the best and second-best
   template is ~0.03–0.06, so the decoder held one chord through whole progressions (Gmaj7
   throughout the B blues `00_Jazz1-200-B_comp`; Em throughout `03_Jazz1-130-D_comp`). Key
   estimation inherits this because chord roots feed its tonic bonus.
4. **Fixed instrument/microphone resonances in GuitarSet's mono-mic audio (data).** The
   long-term spectrum of every comp take shows peaks near 196 Hz and 210 Hz **in every key**
   (Ab, A, B, D, Eb takes alike) — body resonances or open-string sympathetic ringing, not
   played notes. They add G and Ab to both treble and bass chroma, which explains the
   recurring G major / E minor estimates (Pachelbel progression in Eb → G minor for both
   players; B blues → G major). This is specific to the recording; correcting it in the
   analyzer would be fitting this dataset, so it is not corrected. GuitarSet's pickup-mix
   audio would separate the effect (needs approval, `docs/MODELS.md`).
5. **Annotation conventions (data).** GuitarSet keys are the lead-sheet key; solo takes are
   single-line improvisations, so their key score measures something the audio barely
   implies. MusicNet beat references follow the notated beat (eighths in compound meters),
   which turns plausible tracking into 2:3 "errors" that Acc2 does not cover. BabySlakh
   Track00012 scores F 0 for every tracker at the correct tempo (a MIDI-to-render offset is
   likely) and Track00011's reference tempo map reads 250 BPM. No item was removed.
6. **Tuning (ruled out as a main cause).** Estimated deviations from A440: GuitarSet +4 to
   +11 cents, MusicNet −3 to +26 cents (the solo violin), BabySlakh ~1 cent — inside the
   chroma's ±35-cent window.
7. **Beats and tempo.** Remaining held-out errors are metrical-level choices (2:3, 3:4, ×2) and
   expressive classical timing (Beat This! section above). Nothing safe to change without
   new held-out data or a larger model; the minimal postprocessor stays the default.

### Change made to the analyzer

`harmony.CHORD_EMISSION_SCALE` 12 → 30 (the self-transition stays 0.85); `ANALYZER_VERSION`
`dsp-0.2.0` → `dsp-0.2.1`, which invalidates cached sidecars. User corrections are untouched
(they live in the manifest). To avoid fitting the evaluation data, a sweep of scale
{6, 12, 20, 30, 45} × self-transition {0.5, 0.7, 0.85, 0.95} was judged only on the
**synthetic benchmark and GuitarSet player 00**; player 03 was held out. Results are flat from
20 to 45 on the development data; 30 is in the middle of that plateau.

| Chord maj/min (DSP grid, decoder only) | Synthetic (13) | GuitarSet p00 comps (dev, 12) | GuitarSet p03 comps (held out, 10) |
|---|---|---|---|
| scale 12 (before) | 0.82 | 0.58 | 0.42 |
| scale 30 (after) | 0.86 | 0.69 | 0.46 |

A regression test (`test_chord_decoder_follows_dense_voicings`) builds a two-beat progression
of 9th/13th voicings over a leaky chroma floor; it fails at the old scale (0.56) and passes at
30 (0.81).

### Before / after on real audio (production path: Beat This! grid on DirectML)

"Before" = the committed evaluator and analyzer; "aligned" = evaluator fixes 1–2 only;
"after" = aligned + analyzer change 3. Same files, same metrics. Beat This! was trained on
GuitarSet, but key and chords are computed by the DSP analyzers, which were not fitted to it
except as described above (player 00).

| Subset | n | Key: before → aligned → after | Chord maj/min: before → aligned → after | Chord full: before → after |
|---|---|---|---|---|
| GuitarSet Funk comp | 12 | 0.72 → 0.76 → 0.76 | 0.49 → 0.57 → **0.62** | 0.26 → 0.34 |
| GuitarSet Jazz comp | 12 | 0.48 → 0.48 → 0.58 | 0.30 → 0.41 → **0.54** | 0.19 → 0.33 |
| GuitarSet Funk solo | 12 | 0.43 → 0.39 → 0.52 | — | — |
| GuitarSet Jazz solo | 12 | 0.32 → 0.29 → 0.38 | — | — |
| GuitarSet player 00 (dev) | 24 | 0.56 → 0.57 → 0.71 | 0.52 → 0.57 → 0.70 | — |
| **GuitarSet player 03 (held out)** | 24 | **0.42 → 0.40 → 0.40** | **0.28 → 0.39 → 0.43** (10 scored) | — |
| GuitarSet (all) | 48 | 0.49 → 0.48 → **0.56** | 0.40 → 0.49 → **0.58** | 0.22 → 0.33 |
| MusicNet (key-scored) | 7 | 0.80 → 0.90 → 0.80 | — | — |

Two comp takes of player 03 have no performed chord that reduces to maj/min after fix 2 and
are no longer chord-scored (12 → 10). Synthetic benchmark after the change: Acc1 0.77,
Acc2 1.00, beat F 0.92, downbeat F 0.92, key 0.89 (was 0.91: the 3/4 piece, already at the
wrong metrical level, moves from Bb major to D minor), chord maj/min 0.86 (0.82), chord full
0.72 (0.66), melody 0.81. Beats and tempo are unchanged by design.

Regressions: MusicNet 1759 (Schubert D.958 iii, C minor) is estimated as its relative major
again (it was right only with the aligned grid and the old decoder; the relative-key choice
there is fragile), and per-item chord scores drop on 5 of 22 comps (largest:
`03_Jazz3-137-Eb_comp` 1.00 → 0.49). The held-out key score does not improve.

### Status against the v0.2.0 targets

| Target | Result (held-out or fairest reading) | Met? |
|---|---|---|
| Tempo Acc2 ≥ 95 % | 0.70 pooled held out (MusicNet 0.50, BabySlakh 0.80) | no |
| Beat F ≥ 0.85 | 0.74 pooled held out | no |
| Key ≥ 80 % | MusicNet 0.80 (7 movements); GuitarSet 0.56 (player 03 held out: 0.40) | only on classical piano, small n |
| Chord maj/min ≥ 70 % | GuitarSet 0.58 (player 03 held out: 0.43) | no |
| Singing / R&B | no labelled data | not validated |

The decoder fix removes a real defect, but templates over chroma remain the limit on real
voicings: even the per-beat best template is Gmaj7 for a B7(#9) comp. Reaching the chord and
key targets needs a learned chord model or a front end that suppresses partials and fixed
resonances (NNLS-style), and labelled audio closer to the product's material (sung, produced
music). Candidates and their licenses: `docs/MODELS.md`, "Accuracy candidates".
