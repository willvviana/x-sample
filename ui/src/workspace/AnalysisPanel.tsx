import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import type {
  AnalysisCorrections,
  AnalysisView,
  ChordOut,
  EngineClient,
  ProjectView,
  TagOut,
} from "../api/client";
import { usePolling } from "../usePolling";
import { formatPosition } from "./time";

export interface BeatGrid {
  sourceId: string;
  beats: number[];
  downbeats: number[];
}

interface Props {
  client: EngineClient;
  project: ProjectView;
  preferredSourceId: string | undefined;
  onGrid: (grid: BeatGrid | undefined) => void;
  onError: (message: string) => void;
}

const PITCHES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const QUALITIES = ["maj", "min", "7", "maj7", "min7"] as const;
type Quality = (typeof QUALITIES)[number];
const QUALITY_LABEL: Record<Quality, string> = {
  maj: "major",
  min: "minor",
  "7": "dominant 7",
  maj7: "major 7",
  min7: "minor 7",
};
const NUDGE_MS = 10;

type Grid = NonNullable<AnalysisCorrections["beat_grid"]>;
const DEFAULT_GRID: Grid = { scale: "normal", half_phase: 0, offset: 0, downbeat_shift: 0 };

function emptyCorrections(): AnalysisCorrections {
  return {
    beat_grid: DEFAULT_GRID,
    key: null,
    chords: [],
    tags: [],
  };
}

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${String(Math.round(value * 100))} %`;
}

export function AnalysisPanel({ client, project, preferredSourceId, onGrid, onError }: Props) {
  const sources = project.sources.filter((s) => s.media !== null);
  const [sourceId, setSourceId] = useState<string | undefined>(preferredSourceId);
  const current =
    sources.find((s) => s.id === sourceId) ??
    sources.find((s) => s.id === preferredSourceId) ??
    sources[0];
  const id = current?.id;

  useEffect(() => {
    if (preferredSourceId) setSourceId(preferredSourceId);
  }, [preferredSourceId]);

  const load = useCallback(
    () => (id ? client.analysis(id) : Promise.resolve(undefined)),
    [client, id],
  );
  const [fast, setFast] = useState(true);
  const poll = usePolling(load, fast ? 600 : 4000);
  const view = poll.data && poll.data.source_id === id ? poll.data : undefined;
  const running = view?.status === "queued" || view?.status === "running";
  useEffect(() => {
    setFast(running || view === undefined);
  }, [running, view]);

  const [busy, setBusy] = useState(false);
  const [editing, setEditing] = useState<ChordOut>();
  const act = async (action: () => Promise<AnalysisView>) => {
    setBusy(true);
    try {
      await action();
      poll.refresh();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const effective = view?.effective;
  useEffect(() => {
    onGrid(
      effective && id
        ? { sourceId: id, beats: effective.beats, downbeats: effective.downbeats }
        : undefined,
    );
  }, [effective, id, onGrid]);

  const corrections: AnalysisCorrections = view?.corrections ?? emptyCorrections();
  const correct = (change: Partial<AnalysisCorrections>) => {
    if (!id) return;
    void act(() => client.putCorrections(id, { ...corrections, ...change }));
  };

  if (sources.length === 0) {
    return (
      <section className="panel full" aria-labelledby="analysis-title" data-testid="analysis-panel">
        <h2 id="analysis-title">Analysis</h2>
        <p className="muted">Import or record audio to analyse it.</p>
      </section>
    );
  }

  const result = view?.result;
  const job = view?.job;
  return (
    <section className="panel full" aria-labelledby="analysis-title" data-testid="analysis-panel">
      <div className="panel-head">
        <h2 id="analysis-title">Analysis</h2>
        <div className="row">
          <label>
            Source{" "}
            <select
              value={id}
              onChange={(e) => {
                setSourceId(e.target.value);
              }}
              data-testid="analysis-source"
            >
              {sources.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.original_name}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            disabled={busy || running || !id}
            onClick={() => {
              if (id) void act(() => client.runAnalysis(id, view?.status === "ready"));
            }}
            data-testid="analysis-run"
          >
            {view?.status === "ready" ? "Re-run analysis" : "Analyse"}
          </button>
        </div>
      </div>

      <AnalysisStatus view={view} />
      {running && job && (
        <div
          className="progress"
          role="progressbar"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(job.progress * 100)}
          aria-label="Analysis progress"
        >
          <div className="bar" style={{ width: `${String(job.progress * 100)}%` }} />
        </div>
      )}

      {result && effective && id && (
        <>
          <p className="small muted" data-testid="analysis-provenance">
            Ran on <strong data-testid="analysis-backend">{result.backend.toUpperCase()}</strong>
            {result.beat_tracker.method === "model" ? (
              <>
                {" "}
                · beats and downbeats:{" "}
                <strong data-testid="analysis-beat-model">
                  {result.beat_tracker.model_name}
                </strong>{" "}
                on {result.beat_tracker.backend.toUpperCase()} (
                {result.timings.beat_model_s.toFixed(1)} s)
              </>
            ) : null}
            {result.tagger ? (
              <>
                {" "}
                · instruments, vocals and genres: {result.tagger.model_name} on{" "}
                {result.tagger.backend.toUpperCase()} ({result.timings.tagger_s.toFixed(1)} s)
              </>
            ) : null}
            {result.beat_tracker.method === "model" || result.tagger
              ? " · everything else: signal processing on CPU"
              : " (signal-processing analyzers, no AI model)"}{" "}
            · {result.analyzer_version} · {result.timings.total_s.toFixed(1)} s ·{" "}
            {result.created_at}
          </p>
          {result.beat_tracker.method === "model" && result.beat_tracker.license && (
            <p className="small muted" data-testid="analysis-beat-model-license">
              Model license: {result.beat_tracker.license}
            </p>
          )}
          {(result.beat_tracker.notes ?? []).map((note) => (
            <p key={note} className="small muted" data-testid="analysis-beat-model-note">
              {note}
            </p>
          ))}
          <Summary view={view} onCorrect={correct} />
          <p className="small muted">
            Scores rank the alternatives found by the analyzer; they are not calibrated
            probabilities.
          </p>
          <AnalysisLanes
            view={view}
            rate={result.sample_rate}
            onEditChord={(chord) => {
              setEditing(chord);
            }}
          />
          <ChordList
            chords={effective.chords}
            rate={result.sample_rate}
            onEdit={(chord) => {
              setEditing(chord);
            }}
          />
          {editing && (
            <ChordEditor
              key={`${String(editing.start)}-${String(editing.end)}`}
              chord={editing}
              rate={result.sample_rate}
              onCancel={() => {
                setEditing(undefined);
              }}
              onApply={(fix) => {
                setEditing(undefined);
                const others = (corrections.chords ?? []).filter(
                  (c) => c.end <= fix.start || c.start >= fix.end,
                );
                correct({ chords: [...others, fix].sort((a, b) => a.start - b.start) });
              }}
              onRevert={() => {
                setEditing(undefined);
                correct({
                  chords: (corrections.chords ?? []).filter(
                    (c) => c.end <= editing.start || c.start >= editing.end,
                  ),
                });
              }}
            />
          )}
          <div className="analysis-grid">
            <BeatGridEditor view={view} onCorrect={correct} />
            <KeyEditor view={view} onCorrect={correct} />
            <TagEditor view={view} onCorrect={correct} />
            <Boundaries view={view} />
          </div>
          <div className="row">
            <button
              type="button"
              className="secondary"
              disabled={busy || effective.corrected.length === 0}
              onClick={() => {
                void act(() => client.resetCorrections(id));
              }}
              data-testid="analysis-reset"
            >
              Reset all corrections
            </button>
            {effective.corrected.length > 0 && (
              <span className="small muted" data-testid="analysis-corrected">
                Corrected: {effective.corrected.join(", ")}
              </span>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function AnalysisStatus({ view }: { view: AnalysisView | undefined }) {
  if (!view) return <p className="muted">Loading…</p>;
  const text: Record<AnalysisView["status"], string> = {
    none: "Not analysed yet.",
    queued: "Waiting to start…",
    running: view.job?.message ?? "Analysing…",
    ready: "",
    failed: `Analysis failed: ${view.error ?? "unknown error"}`,
    cancelled: "Analysis was cancelled.",
    stale: "The audio or the analyzer changed since the last analysis; run it again.",
  };
  const message = text[view.status];
  if (!message) return null;
  const tone = view.status === "failed" ? " bad" : view.status === "stale" ? " warn" : "";
  return (
    <p className={tone ? `notice${tone}` : "muted"} data-testid="analysis-status">
      {message}
    </p>
  );
}

function Summary({
  view,
  onCorrect,
}: {
  view: AnalysisView;
  onCorrect: (c: Partial<AnalysisCorrections>) => void;
}) {
  const result = view.result;
  const eff = view.effective;
  if (!result || !eff) return null;
  const r = result.rhythm;
  const h = result.harmony;
  const notes = h.melody_notes;
  const vibrato = notes.filter((n) => n.vibrato_hz !== null).length;
  const articulation = (kind: string) => notes.filter((n) => n.articulation === kind).length;
  return (
    <dl className="facts analysis-facts" data-testid="analysis-summary">
      <dt>Tempo</dt>
      <dd>
        <strong data-testid="analysis-bpm">{eff.bpm.toFixed(1)} BPM</strong>{" "}
        <span className="muted small">
          detected {r.bpm.toFixed(1)}, score {pct(r.confidence)}
          {r.variable_tempo ? ", tempo varies" : ""}
        </span>
        {r.candidates.length > 1 && (
          <span className="row wrap small">
            Alternatives:
            {r.candidates
              .filter((c) => c.relation !== "primary")
              .map((c) => (
                <button
                  type="button"
                  className="secondary small-btn"
                  key={`${c.relation}-${String(c.bpm)}`}
                  onClick={() => {
                    const grid = view.corrections.beat_grid ?? DEFAULT_GRID;
                    onCorrect({
                      beat_grid:
                        c.relation === "half" || c.relation === "double"
                          ? { ...grid, bpm: null, scale: c.relation }
                          : { ...grid, bpm: Math.round(c.bpm * 10) / 10, scale: "normal" },
                    });
                  }}
                >
                  {c.bpm.toFixed(1)} ({c.relation})
                </button>
              ))}
          </span>
        )}
      </dd>
      <dt>Meter</dt>
      <dd>
        {eff.meter === null ? "—" : `${String(eff.meter)} beats/bar`}{" "}
        <span className="muted small">score {pct(r.meter_confidence)}</span>
      </dd>
      <dt>Groove</dt>
      <dd>
        {r.subdivision ?? "—"} subdivision
        {r.swing_ratio !== null ? `, swing ${r.swing_ratio.toFixed(2)}` : ""}
        {r.syncopation !== null ? `, syncopation ${r.syncopation.toFixed(2)}` : ""}
        {r.microtiming_ms !== null ? `, timing spread ${r.microtiming_ms.toFixed(0)} ms` : ""}
      </dd>
      <dt>Key</dt>
      <dd>
        <strong data-testid="analysis-key">{eff.key?.label ?? "—"}</strong>{" "}
        <span className="muted small">
          score {pct(h.key_confidence)}
          {h.key_candidates.length > 1 &&
            `; also ${h.key_candidates
              .slice(1)
              .map((k) => k.label)
              .join(", ")}`}
          {h.local_keys.length > 1 && `; ${String(h.local_keys.length)} key regions`}
        </span>
      </dd>
      <dt>Harmony</dt>
      <dd>
        {String(eff.chords.filter((c) => c.root !== null).length)} chord segments
        {h.harmonic_rhythm_beats !== null
          ? `, a change every ${h.harmonic_rhythm_beats.toFixed(1)} beats`
          : ""}
      </dd>
      <dt>Melody</dt>
      <dd>
        {notes.length === 0
          ? "No clear lead line"
          : `${String(notes.length)} notes (voiced ${pct(h.voiced_ratio)}), ${String(vibrato)} with vibrato; ${String(articulation("attack"))} attacked, ${String(articulation("legato"))} legato, ${String(articulation("glide"))} glides`}
      </dd>
      <dt>Dynamics</dt>
      <dd>
        {result.structure.dynamic_range_db === null
          ? "—"
          : `${result.structure.dynamic_range_db.toFixed(1)} dB range`}
        , {String(result.structure.sections.length)} sections
      </dd>
      {[...r.notes, ...h.notes].length > 0 && (
        <>
          <dt>Notes</dt>
          <dd className="small">
            {[...r.notes, ...h.notes].map((n) => (
              <div key={n}>{n}</div>
            ))}
          </dd>
        </>
      )}
    </dl>
  );
}

function BeatGridEditor({
  view,
  onCorrect,
}: {
  view: AnalysisView;
  onCorrect: (c: Partial<AnalysisCorrections>) => void;
}) {
  const grid = view.corrections.beat_grid ?? DEFAULT_GRID;
  const rate = view.result?.sample_rate ?? 48000;
  const meter = view.effective?.meter ?? 4;
  const [bpm, setBpm] = useState("");
  const set = (change: Partial<Grid>) => {
    onCorrect({ beat_grid: { ...grid, ...change } });
  };
  return (
    <fieldset className="analysis-box" data-testid="beat-grid-editor">
      <legend>Beat grid</legend>
      <div className="row wrap" role="group" aria-label="Tempo scale">
        {(["half", "normal", "double"] as const).map((s) => (
          <button
            type="button"
            key={s}
            className={grid.scale === s && !grid.bpm ? "" : "secondary"}
            aria-pressed={grid.scale === s && !grid.bpm}
            onClick={() => {
              set({ scale: s, bpm: null });
            }}
          >
            {s === "half" ? "Half time" : s === "double" ? "Double time" : "As detected"}
          </button>
        ))}
        {grid.scale === "half" && !grid.bpm && (
          <button
            type="button"
            className="secondary"
            onClick={() => {
              set({ half_phase: grid.half_phase === 0 ? 1 : 0 });
            }}
          >
            Use the other beats
          </button>
        )}
      </div>
      <form
        className="row wrap"
        onSubmit={(e) => {
          e.preventDefault();
          const value = Number(bpm);
          if (value >= 20 && value <= 300) set({ bpm: value, scale: "normal" });
        }}
      >
        <label>
          Fixed BPM{" "}
          <input
            type="number"
            min={20}
            max={300}
            step={0.1}
            value={bpm}
            placeholder={view.effective?.bpm.toFixed(1)}
            onChange={(e) => {
              setBpm(e.target.value);
            }}
            data-testid="bpm-input"
          />
        </label>
        <button type="submit" className="secondary">
          Set
        </button>
        {grid.bpm && (
          <button
            type="button"
            className="secondary"
            onClick={() => {
              set({ bpm: null });
            }}
          >
            Clear
          </button>
        )}
      </form>
      <div className="row wrap">
        <span className="small">Nudge</span>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            set({ offset: grid.offset - Math.round((NUDGE_MS * rate) / 1000) });
          }}
        >
          −{NUDGE_MS} ms
        </button>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            set({ offset: grid.offset + Math.round((NUDGE_MS * rate) / 1000) });
          }}
        >
          +{NUDGE_MS} ms
        </button>
        <span className="small muted">offset {((grid.offset / rate) * 1000).toFixed(0)} ms</span>
      </div>
      <div className="row wrap">
        <label>
          Beats per bar{" "}
          <select
            value={grid.meter ?? ""}
            onChange={(e) => {
              set({ meter: e.target.value ? Number(e.target.value) : null, downbeat_shift: 0 });
            }}
            data-testid="meter-select"
          >
            <option value="">As detected</option>
            {[2, 3, 4, 5, 6, 7, 12].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          className="secondary"
          onClick={() => {
            set({ downbeat_shift: (grid.downbeat_shift + 1) % meter });
          }}
          data-testid="shift-downbeat"
        >
          Move beat 1 later
        </button>
      </div>
    </fieldset>
  );
}

function KeyEditor({
  view,
  onCorrect,
}: {
  view: AnalysisView;
  onCorrect: (c: Partial<AnalysisCorrections>) => void;
}) {
  const key = view.corrections.key;
  const detected = view.result?.harmony.key_candidates ?? [];
  return (
    <fieldset className="analysis-box" data-testid="key-editor">
      <legend>Key</legend>
      <div className="row wrap">
        <select
          aria-label="Key tonic"
          value={key ? String(key.tonic) : ""}
          onChange={(e) => {
            const tonic = e.target.value;
            onCorrect({
              key: tonic === "" ? null : { tonic: Number(tonic), mode: key?.mode ?? "major" },
            });
          }}
          data-testid="key-tonic"
        >
          <option value="">As detected</option>
          {PITCHES.map((p, i) => (
            <option key={p} value={i}>
              {p}
            </option>
          ))}
        </select>
        <select
          aria-label="Key mode"
          value={key?.mode ?? "major"}
          disabled={!key}
          onChange={(e) => {
            if (key) onCorrect({ key: { ...key, mode: e.target.value as "major" | "minor" } });
          }}
          data-testid="key-mode"
        >
          <option value="major">major</option>
          <option value="minor">minor</option>
        </select>
      </div>
      {detected.length > 0 && (
        <div className="row wrap small">
          Detected:
          {detected.map((k) => (
            <button
              type="button"
              className="secondary small-btn"
              key={k.label}
              onClick={() => {
                onCorrect({ key: { tonic: k.tonic, mode: k.mode } });
              }}
            >
              {k.label} ({pct(k.score)})
            </button>
          ))}
        </div>
      )}
    </fieldset>
  );
}

type TagKind = "genre" | "instrument" | "vocal";
const KIND_TITLE: Record<TagKind, string> = {
  instrument: "Instruments",
  vocal: "Vocals",
  genre: "Genres",
};

function tagKey(kind: string, label: string): string {
  return `${kind}:${label.trim().toLowerCase()}`;
}

function activityText(tag: TagOut | undefined, rate: number): string {
  if (!tag || tag.activity.length === 0) return "";
  return tag.activity
    .map((a) => `${formatPosition(a.start, rate)}–${formatPosition(a.end, rate)}`)
    .join(", ");
}

function TagEditor({
  view,
  onCorrect,
}: {
  view: AnalysisView;
  onCorrect: (c: Partial<AnalysisCorrections>) => void;
}) {
  const [kind, setKind] = useState<TagKind>("genre");
  const [label, setLabel] = useState("");
  const tags = view.corrections.tags ?? [];
  const result = view.result;
  const shown = view.effective?.tags ?? [];
  const fields = result
    ? ([
        ["instrument", result.instruments],
        ["vocal", result.vocals],
        ["genre", result.genres],
      ] as const)
    : [];
  const detected = new Map<string, TagOut>();
  for (const [, field] of fields) {
    if (field.status === "available") {
      for (const t of field.tags) detected.set(tagKey(t.kind, t.label), t);
    }
  }
  const unavailable = fields.filter(([, f]) => f.status === "unavailable");
  const rejected = tags.filter((t) => !t.present);
  const rate = result?.sample_rate ?? 1;
  const tagger = result?.tagger;
  return (
    <fieldset className="analysis-box" data-testid="tag-editor">
      <legend>Instruments, vocals and genres</legend>
      {unavailable.length > 0 && (
        <ul className="small muted unavailable">
          {unavailable.map(([k, f]) => (
            <li key={k}>
              {k === "genre"
                ? "Genre recognition"
                : k === "instrument"
                  ? "Instrument recognition"
                  : "Vocal activity"}
              : {"reason" in f ? f.reason : ""}
            </li>
          ))}
        </ul>
      )}
      {tagger && (
        <p className="small muted" data-testid="tagger-provenance">
          Detected by <strong>{tagger.model_name}</strong> on{" "}
          <strong data-testid="tagger-backend">{tagger.backend.toUpperCase()}</strong> (
          {result.timings.tagger_s.toFixed(1)} s) in {tagger.window_s.toFixed(0)} s windows. Scores
          are model outputs, not probabilities: detected from {pct(tagger.detect_threshold)}, shown
          as uncertain from {pct(tagger.uncertain_threshold)}. Labels marked “not evaluated” have no
          measured accuracy yet. License: {tagger.license}
          {(tagger.notes ?? []).map((note) => (
            <span key={note} className="block" data-testid="tagger-note">
              {note}
            </span>
          ))}
        </p>
      )}
      {(["instrument", "vocal", "genre"] as const).map((k) => {
        const group = shown.filter((t) => t.kind === k);
        if (group.length === 0) {
          const checked = fields.some(([fk, f]) => fk === k && f.status === "available");
          return checked ? (
            <p key={k} className="small muted" data-testid={`tag-none-${k}`}>
              {KIND_TITLE[k]}: none detected
            </p>
          ) : null;
        }
        return (
          <div key={k} className="tag-group">
            <span className="small muted">{KIND_TITLE[k]}</span>
            <ul className="tag-list" data-testid={`tag-list-${k}`}>
              {group.map((t) => {
                const info = detected.get(tagKey(t.kind, t.label));
                const activity = activityText(info, rate);
                const user = t.origin === "user";
                return (
                  <li
                    key={tagKey(t.kind, t.label)}
                    className={`tag${t.uncertain ? " uncertain" : ""}`}
                    title={activity ? `Active: ${activity}` : undefined}
                    data-testid="tag"
                    data-origin={t.origin}
                  >
                    {t.label}
                    {!user && <span className="muted small">{pct(t.confidence)}</span>}
                    {t.uncertain && <span className="muted small">uncertain</span>}
                    {!user && !t.evaluated && <span className="muted small">not evaluated</span>}
                    {user && <span className="muted small">yours</span>}
                    {!user && t.uncertain && (
                      <button
                        type="button"
                        className="secondary small-btn"
                        aria-label={`Confirm ${t.label}`}
                        onClick={() => {
                          onCorrect({
                            tags: [...tags, { kind: t.kind, label: t.label, present: true }],
                          });
                        }}
                      >
                        ✓
                      </button>
                    )}
                    <button
                      type="button"
                      className="secondary small-btn"
                      aria-label={user ? `Remove ${t.label}` : `Not present: ${t.label}`}
                      onClick={() => {
                        const rest = tags.filter(
                          (x) => tagKey(x.kind, x.label) !== tagKey(t.kind, t.label),
                        );
                        onCorrect({
                          tags:
                            user && !info
                              ? rest
                              : [...rest, { kind: t.kind, label: t.label, present: false }],
                        });
                      }}
                    >
                      ×
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
      {rejected.length > 0 && (
        <p className="small muted" data-testid="tag-rejected">
          Marked not present:{" "}
          {rejected.map((t) => (
            <button
              type="button"
              className="secondary small-btn"
              key={tagKey(t.kind, t.label)}
              aria-label={`Undo ${t.label}`}
              onClick={() => {
                onCorrect({ tags: tags.filter((x) => x !== t) });
              }}
            >
              {t.label} ↺
            </button>
          ))}
        </p>
      )}
      <form
        className="row wrap"
        onSubmit={(e) => {
          e.preventDefault();
          const value = label.trim();
          if (!value) return;
          const rest = tags.filter((x) => tagKey(x.kind, x.label) !== tagKey(kind, value));
          onCorrect({ tags: [...rest, { kind, label: value, present: true }] });
          setLabel("");
        }}
      >
        <select
          aria-label="Tag kind"
          value={kind}
          onChange={(e) => {
            setKind(e.target.value as TagKind);
          }}
        >
          <option value="genre">Genre</option>
          <option value="instrument">Instrument</option>
          <option value="vocal">Vocal</option>
        </select>
        <input
          aria-label="Tag"
          value={label}
          maxLength={60}
          list="tag-suggestions"
          placeholder="e.g. R&B, Piano, Violin, Saxophone"
          onChange={(e) => {
            setLabel(e.target.value);
          }}
          data-testid="tag-input"
        />
        <datalist id="tag-suggestions">
          {[
            "R&B",
            "Contemporary R&B",
            "Soul",
            "Neo soul",
            "Jazz",
            "Funk",
            "Pop",
            "Hip hop",
            "Classical",
            "Acoustic piano",
            "Electric piano",
            "Keyboards",
            "Violin",
            "Viola",
            "Cello",
            "Strings",
            "Saxophone",
            "Bass",
            "Drums",
            "Guitar",
            "Singing",
            "Rapping",
          ].map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
        <button type="submit" className="secondary">
          Add tag
        </button>
      </form>
    </fieldset>
  );
}

function Boundaries({ view }: { view: AnalysisView }) {
  const b = view.result?.boundaries;
  if (!b) return null;
  const rows = [
    ["Start", b.start],
    ["End", b.end],
  ] as const;
  return (
    <fieldset className="analysis-box" data-testid="boundaries">
      <legend>Start and end</legend>
      <p className="small muted">
        Measurements only, for intro/outro repair in a later version. Nothing is judged or changed.
      </p>
      <table className="small">
        <thead>
          <tr>
            <th />
            <th>Level vs median</th>
            <th>Edge sample</th>
            <th>Beat offset</th>
            <th>Beats to/from bar</th>
            <th>Chord</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([name, e]) => (
            <tr key={name}>
              <th scope="row">{name}</th>
              <td>{e.level_vs_median_db.toFixed(1)} dB</td>
              <td>{e.edge_amplitude.toFixed(3)}</td>
              <td>{e.beat_offset === null ? "—" : e.beat_offset.toFixed(2)}</td>
              <td>{e.beats_from_downbeat === null ? "—" : e.beats_from_downbeat.toFixed(1)}</td>
              <td>
                {e.chord ?? "—"}
                {e.chord_is_tonic ? " (tonic)" : ""}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </fieldset>
  );
}

function ChordList({
  chords,
  rate,
  onEdit,
}: {
  chords: ChordOut[];
  rate: number;
  onEdit: (c: ChordOut) => void;
}) {
  return (
    <details className="chord-list">
      <summary>Chord list ({chords.length})</summary>
      <ol data-testid="chord-list">
        {chords.map((c) => (
          <li key={`${String(c.start)}-${String(c.end)}`}>
            <button
              type="button"
              className="secondary small-btn"
              onClick={() => {
                onEdit(c);
              }}
            >
              {formatPosition(c.start, rate)} {c.label}
              {c.confidence >= 1 ? " ✎" : ""}
            </button>
          </li>
        ))}
      </ol>
    </details>
  );
}

function ChordEditor({
  chord,
  rate,
  onApply,
  onRevert,
  onCancel,
}: {
  chord: ChordOut;
  rate: number;
  onApply: (fix: NonNullable<AnalysisCorrections["chords"]>[number]) => void;
  onRevert: () => void;
  onCancel: () => void;
}) {
  const [root, setRoot] = useState<number | null>(chord.root ?? null);
  const [quality, setQuality] = useState<Quality>(chord.quality ?? "maj");
  const [bass, setBass] = useState<number | null>(chord.bass ?? chord.root ?? null);
  return (
    <form
      className="chord-editor row wrap"
      aria-label="Edit chord"
      data-testid="chord-editor"
      onSubmit={(e) => {
        e.preventDefault();
        onApply(
          root === null
            ? { start: chord.start, end: chord.end, root: null, quality: null, bass: null }
            : { start: chord.start, end: chord.end, root, quality, bass },
        );
      }}
    >
      <strong>
        {formatPosition(chord.start, rate)}–{formatPosition(chord.end, rate)}
      </strong>
      <select
        aria-label="Chord root"
        value={root === null ? "" : String(root)}
        onChange={(e) => {
          const value = e.target.value === "" ? null : Number(e.target.value);
          setRoot(value);
          setBass(value);
        }}
        data-testid="chord-root"
      >
        <option value="">No chord</option>
        {PITCHES.map((p, i) => (
          <option key={p} value={i}>
            {p}
          </option>
        ))}
      </select>
      <select
        aria-label="Chord quality"
        value={quality}
        disabled={root === null}
        onChange={(e) => {
          setQuality(e.target.value as Quality);
        }}
        data-testid="chord-quality"
      >
        {QUALITIES.map((q) => (
          <option key={q} value={q}>
            {QUALITY_LABEL[q]}
          </option>
        ))}
      </select>
      <select
        aria-label="Bass note"
        value={bass === null ? "" : String(bass)}
        disabled={root === null}
        onChange={(e) => {
          setBass(e.target.value === "" ? null : Number(e.target.value));
        }}
      >
        {PITCHES.map((p, i) => (
          <option key={p} value={i}>
            bass {p}
          </option>
        ))}
      </select>
      <button type="submit">Apply</button>
      <button type="button" className="secondary" onClick={onRevert}>
        Use detected
      </button>
      <button type="button" className="secondary" onClick={onCancel}>
        Cancel
      </button>
    </form>
  );
}

const LANES = { sections: 20, chords: 26, melody: 70, loudness: 34 };
const LANES_HEIGHT = LANES.sections + LANES.chords + LANES.melody + LANES.loudness;

function AnalysisLanes({
  view,
  rate,
  onEditChord,
}: {
  view: AnalysisView;
  rate: number;
  onEditChord: (c: ChordOut) => void;
}) {
  const frame = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = useState(800);
  const [zoom, setZoom] = useState(1);
  const result = view.result;
  const eff = view.effective;
  const frames = result?.frames ?? 1;
  const pixels = Math.min(16000, Math.max(100, Math.round(width * zoom)));

  useLayoutEffect(() => {
    const el = frame.current;
    if (!el) return;
    const measure = () => {
      setWidth(Math.max(100, Math.floor(el.clientWidth)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => {
      observer.disconnect();
    };
  }, []);

  const melodyRange = useMemo(() => {
    const values = (result?.harmony.melody_contour ?? [])
      .map((p) => p.midi)
      .filter((m): m is number => m !== null);
    if (values.length === 0) return [55, 80] as const;
    return [Math.min(...values) - 2, Math.max(...values) + 2] as const;
  }, [result]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || !result || !eff) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const css = getComputedStyle(el);
    const color = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
    const x = (s: number) => (s / frames) * pixels;
    ctx.clearRect(0, 0, el.width, el.height);
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "middle";
    // Sections
    let y = 0;
    result.structure.sections.forEach((s, i) => {
      ctx.fillStyle = i % 2 ? color("--lane", "#eef0f5") : color("--sel", "#ccd");
      ctx.fillRect(x(s.start), y, Math.max(1, x(s.end) - x(s.start)), LANES.sections - 2);
      ctx.fillStyle = color("--text", "#111");
      ctx.fillText(s.label, x(s.start) + 3, y + LANES.sections / 2);
    });
    // Chords
    y += LANES.sections;
    for (const c of eff.chords) {
      const w = Math.max(1, x(c.end) - x(c.start));
      ctx.fillStyle = c.root === null ? color("--lane", "#eee") : color("--info-bg", "#e7edfb");
      ctx.fillRect(x(c.start), y, w - 1, LANES.chords - 2);
      if (c.confidence >= 1) {
        ctx.strokeStyle = color("--accent", "#5b4bdb");
        ctx.strokeRect(x(c.start) + 0.5, y + 0.5, w - 2, LANES.chords - 3);
      }
      if (w > 24) {
        ctx.fillStyle = color("--info-fg", "#2c4a9a");
        ctx.fillText(c.label, x(c.start) + 3, y + LANES.chords / 2, w - 4);
      }
    }
    // Melody contour
    y += LANES.chords;
    const [lo, hi] = melodyRange;
    ctx.strokeStyle = color("--wave", "#5b4bdb");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    let pen = false;
    for (const p of result.harmony.melody_contour) {
      if (p.midi === null) {
        pen = false;
        continue;
      }
      const py = y + LANES.melody - ((p.midi - lo) / (hi - lo)) * LANES.melody;
      if (pen) ctx.lineTo(x(p.position), py);
      else ctx.moveTo(x(p.position), py);
      pen = true;
    }
    ctx.stroke();
    // Loudness
    y += LANES.melody;
    const curve = result.structure.loudness_curve;
    ctx.fillStyle = color("--muted", "#666");
    for (let i = 0; i < curve.length; i++) {
      const p = curve[i];
      const next = curve[i + 1];
      if (!p) continue;
      const h = Math.max(0, Math.min(1, (p.db + 60) / 60)) * (LANES.loudness - 2);
      const end = next ? next.position : frames;
      ctx.fillRect(x(p.position), y + LANES.loudness - h, Math.max(1, x(end) - x(p.position)), h);
    }
    // Beat grid over everything (downbeats stronger), only when lines stay apart.
    const spacing = eff.beats.length > 1 ? pixels / eff.beats.length : 0;
    ctx.fillStyle = color("--muted", "#666");
    if (spacing >= 5) {
      ctx.globalAlpha = 0.35;
      for (const b of eff.beats) ctx.fillRect(Math.round(x(b)), 0, 1, LANES_HEIGHT);
    }
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = color("--accent", "#5b4bdb");
    for (const d of eff.downbeats) ctx.fillRect(Math.round(x(d)), 0, 1, LANES_HEIGHT);
    ctx.globalAlpha = 1;
  }, [result, eff, frames, pixels, melodyRange]);

  if (!result || !eff) return null;
  return (
    <div className="analysis-lanes" ref={frame}>
      <div className="row small">
        <span className="muted">Sections · chords (click to correct) · melody · loudness</span>
        <span className="spacer" />
        <label>
          Zoom{" "}
          <select
            value={zoom}
            onChange={(e) => {
              setZoom(Number(e.target.value));
            }}
          >
            {[1, 2, 4, 8, 16].map((z) => (
              <option key={z} value={z}>
                {z}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="analysis-scroll">
        <canvas
          ref={canvas}
          width={pixels}
          height={LANES_HEIGHT}
          style={{ width: pixels, height: LANES_HEIGHT }}
          data-testid="analysis-canvas"
          aria-label="Analysis lanes: sections, chords, melody contour, loudness and beat grid"
          role="img"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const py = e.clientY - rect.top;
            if (py < LANES.sections || py >= LANES.sections + LANES.chords) return;
            const sample = Math.round(((e.clientX - rect.left) / pixels) * frames);
            const chord = eff.chords.find((c) => sample >= c.start && sample < c.end);
            if (chord) onEditChord(chord);
          }}
        />
      </div>
      <p className="small muted">
        {formatPosition(0, rate)} – {formatPosition(frames, rate)} of the source (before trims)
      </p>
    </div>
  );
}
