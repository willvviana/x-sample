import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";

import type { Clip, EngineClient, Peaks, ProjectView } from "../api/client";
import type { BeatGrid } from "./AnalysisPanel";
import { clamp, formatPosition } from "./time";

export interface Selection {
  trackId: string;
  start: number;
  end: number;
}

interface Props {
  client: EngineClient;
  project: ProjectView;
  playhead: number;
  selection: Selection | undefined;
  selectedClipId: string | undefined;
  onPlayhead: (sample: number) => void;
  onSelect: (selection: Selection | undefined) => void;
  onPickClip: (clipId: string | undefined) => void;
  onTogglePlay: () => void;
  grid?: BeatGrid | undefined;
}

const LANE_HEIGHT = 84;
const LABEL_WIDTH = 100;

export function Timeline({
  client,
  project,
  playhead,
  selection,
  selectedClipId,
  onPlayhead,
  onSelect,
  onPickClip,
  onTogglePlay,
  grid,
}: Props) {
  const rate = project.sample_rate;
  const frame = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const total = Math.max(project.length_samples, rate);
  // Zoom is samples per pixel; viewStart is the first visible sample.
  const [spp, setSpp] = useState<number>();
  const [viewStart, setViewStart] = useState(0);

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

  const fitSpp = Math.max(1, Math.ceil(total / width));
  const minSpp = 1;
  const scale = clamp(spp ?? fitSpp, minSpp, fitSpp);
  const visible = scale * width;
  const maxStart = Math.max(0, total - visible);
  const start = clamp(viewStart, 0, maxStart);

  const zoom = (factor: number, anchorSample = start + visible / 2) => {
    const next = clamp(Math.round(scale * factor), minSpp, fitSpp);
    const nextStart = clamp(
      Math.round(anchorSample - ((anchorSample - start) / scale) * next),
      0,
      Math.max(0, total - next * width),
    );
    setSpp(next);
    setViewStart(nextStart);
  };

  const toSample = (clientX: number, laneLeft: number) =>
    clamp(Math.round(start + (clientX - laneLeft) * scale), 0, total);
  const toX = (sample: number) => (sample - start) / scale;

  const onKeyDown = (event: KeyboardEvent) => {
    const step = (event.ctrlKey ? 10 : 1) * Math.max(1, Math.round(scale * 10));
    if (event.key === " ") {
      event.preventDefault();
      onTogglePlay();
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      const next = clamp(playhead + (event.key === "ArrowLeft" ? -step : step), 0, total);
      if (event.shiftKey && selection) {
        const anchor = playhead === selection.start ? selection.end : selection.start;
        onSelect({
          trackId: selection.trackId,
          start: Math.min(anchor, next),
          end: Math.max(anchor, next),
        });
      }
      onPlayhead(next);
    } else if (event.key === "+" || event.key === "=") {
      zoom(0.5);
    } else if (event.key === "-") {
      zoom(2);
    } else if (event.key === "Escape") {
      onSelect(undefined);
    }
  };

  const ticks = buildTicks(start, visible, rate);

  return (
    <div className="timeline" data-testid="timeline">
      <div className="timeline-tools">
        <button
          type="button"
          onClick={() => {
            zoom(0.5);
          }}
          aria-label="Zoom in"
        >
          Zoom in
        </button>
        <button
          type="button"
          onClick={() => {
            zoom(2);
          }}
          aria-label="Zoom out"
        >
          Zoom out
        </button>
        <button
          type="button"
          onClick={() => {
            setSpp(undefined);
            setViewStart(0);
          }}
        >
          Fit
        </button>
        <span className="muted" data-testid="zoom-level">
          {scale === fitSpp ? "fit" : `${String(scale)} samples/px`}
        </span>
        <span className="muted spacer" aria-live="off">
          Playhead <strong data-testid="playhead-time">{formatPosition(playhead, rate)}</strong>
          {selection && (
            <>
              {" "}
              · Selection{" "}
              <strong data-testid="selection-range">
                {formatPosition(selection.start, rate)}–{formatPosition(selection.end, rate)}
              </strong>
            </>
          )}
        </span>
      </div>
      <div
        className="timeline-body"
        tabIndex={0}
        role="application"
        aria-label="Timeline. Arrow keys move the playhead, shift extends the selection, space plays, plus and minus zoom."
        onKeyDown={onKeyDown}
        data-testid="timeline-body"
      >
        <div className="timeline-row ruler-row">
          <div className="lane-label" style={{ width: LABEL_WIDTH }} />
          <div className="ruler" ref={frame} style={{ height: 22 }}>
            {ticks.map((tick) => (
              <span key={tick} className="tick" style={{ left: toX(tick) }}>
                {formatPosition(tick, rate)}
              </span>
            ))}
          </div>
        </div>
        {project.tracks.map((track) => {
          const clips = project.clips.filter((c) => c.track_id === track.id);
          return (
            <div className="timeline-row" key={track.id}>
              <div
                className={`lane-label${track.muted ? " muted-track" : ""}`}
                style={{ width: LABEL_WIDTH }}
              >
                {track.name}
              </div>
              <Lane
                width={width}
                clips={clips}
                trackId={track.id}
                client={client}
                project={project}
                start={start}
                scale={scale}
                selectedClipId={selectedClipId}
                selection={selection?.trackId === track.id ? selection : undefined}
                toSample={toSample}
                toX={toX}
                onPlayhead={onPlayhead}
                onSelect={onSelect}
                onPickClip={onPickClip}
                grid={grid}
              />
            </div>
          );
        })}
        <div
          className="playhead"
          style={{ left: LABEL_WIDTH + toX(playhead) }}
          aria-hidden="true"
        />
      </div>
      <input
        type="range"
        className="scroll"
        aria-label="Scroll timeline"
        min={0}
        max={maxStart}
        step={Math.max(1, Math.round(scale))}
        value={start}
        disabled={maxStart === 0}
        onChange={(e) => {
          setViewStart(Number(e.target.value));
        }}
      />
    </div>
  );
}

function buildTicks(start: number, visible: number, rate: number): number[] {
  const seconds = visible / rate;
  const steps = [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 300];
  const step = (steps.find((s) => seconds / s <= 8) ?? 600) * rate;
  const first = Math.ceil(start / step) * step;
  const out: number[] = [];
  for (let t = first; t < start + visible && out.length < 20; t += step) out.push(Math.round(t));
  return out;
}

interface LaneProps {
  width: number;
  clips: Clip[];
  trackId: string;
  client: EngineClient;
  project: ProjectView;
  start: number;
  scale: number;
  selectedClipId: string | undefined;
  selection: Selection | undefined;
  toSample: (clientX: number, laneLeft: number) => number;
  toX: (sample: number) => number;
  onPlayhead: (sample: number) => void;
  onSelect: (selection: Selection | undefined) => void;
  onPickClip: (clipId: string | undefined) => void;
  grid: BeatGrid | undefined;
}

function Lane({
  width,
  clips,
  trackId,
  client,
  project,
  start,
  scale,
  selectedClipId,
  selection,
  toSample,
  toX,
  onPlayhead,
  onSelect,
  onPickClip,
  grid,
}: LaneProps) {
  const ref = useRef<HTMLDivElement>(null);
  const drag = useRef<{ anchor: number; moved: boolean }>(undefined);

  const clipAt = (sample: number) =>
    clips.find((c) => sample >= c.timeline_start && sample < c.timeline_start + c.duration);

  const onPointerDown = (event: React.PointerEvent) => {
    if (event.button !== 0) return;
    const left = ref.current?.getBoundingClientRect().left ?? 0;
    const sample = toSample(event.clientX, left);
    drag.current = { anchor: sample, moved: false };
    ref.current?.setPointerCapture(event.pointerId);
    onPlayhead(sample);
    onSelect(undefined);
    onPickClip(clipAt(sample)?.id);
  };

  const onPointerMove = (event: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const left = ref.current?.getBoundingClientRect().left ?? 0;
    const sample = toSample(event.clientX, left);
    if (!d.moved && Math.abs(sample - d.anchor) < 3 * scale) return;
    d.moved = true;
    // A selection never leaves the clip it started in.
    const clip = clipAt(d.anchor);
    const low = clip ? clip.timeline_start : 0;
    const high = clip ? clip.timeline_start + clip.duration : 0;
    const a = clamp(Math.min(d.anchor, sample), low, high);
    const b = clamp(Math.max(d.anchor, sample), low, high);
    if (clip && b > a) onSelect({ trackId, start: a, end: b });
  };

  const onPointerUp = (event: React.PointerEvent) => {
    drag.current = undefined;
    if (ref.current?.hasPointerCapture(event.pointerId)) {
      ref.current.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div
      className="lane"
      ref={ref}
      style={{ height: LANE_HEIGHT }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      data-testid={`lane-${trackId}`}
    >
      {clips.map((clip) => (
        <ClipWave
          key={clip.id}
          clip={clip}
          client={client}
          project={project}
          x0={toX(clip.timeline_start)}
          x1={toX(clip.timeline_start + clip.duration)}
          start={start}
          scale={scale}
          width={width}
          selected={clip.id === selectedClipId}
        />
      ))}
      {grid && <GridLines clips={clips} grid={grid} start={start} scale={scale} width={width} />}
      {selection && (
        <div
          className="selection"
          style={{
            left: toX(selection.start),
            width: Math.max(1, toX(selection.end) - toX(selection.start)),
          }}
          data-testid="selection"
        />
      )}
    </div>
  );
}

/** Effective beat grid of the analysed source, mapped through each clip onto the timeline. */
function GridLines(props: {
  clips: Clip[];
  grid: BeatGrid;
  start: number;
  scale: number;
  width: number;
}) {
  const { clips, grid, start, scale, width } = props;
  const lines: { x: number; down: boolean }[] = [];
  const downs = new Set(grid.downbeats);
  const spacing = grid.beats.length > 1 ? ((grid.beats[1] ?? 0) - (grid.beats[0] ?? 0)) / scale : 0;
  for (const clip of clips) {
    if (clip.source_id !== grid.sourceId) continue;
    for (const b of grid.beats) {
      if (b < clip.source_in || b >= clip.source_in + clip.duration) continue;
      const down = downs.has(b);
      if (!down && spacing < 5) continue;
      const x = (clip.timeline_start + (b - clip.source_in) - start) / scale;
      if (x >= 0 && x < width) lines.push({ x, down });
      if (lines.length > 4000) break;
    }
  }
  return (
    <div className="grid-lines" aria-hidden="true" data-testid="beat-grid">
      {lines.map((l) => (
        <span
          key={l.x}
          className={l.down ? "beat-line down" : "beat-line"}
          style={{ left: Math.round(l.x) }}
        />
      ))}
    </div>
  );
}

interface WaveProps {
  clip: Clip;
  client: EngineClient;
  project: ProjectView;
  x0: number;
  x1: number;
  start: number;
  scale: number;
  width: number;
  selected: boolean;
}

function ClipWave({ clip, client, project, x0, x1, start, scale, width, selected }: WaveProps) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [peaks, setPeaks] = useState<{ key: string; data: Peaks }>();
  const [failure, setFailure] = useState<string>();

  // Only the visible part of the clip is requested, one min/max pair per pixel column.
  const visLeft = Math.max(0, Math.floor(x0));
  const visRight = Math.min(width, Math.ceil(x1));
  const columns = visRight - visLeft;
  const tStart = Math.max(clip.timeline_start, Math.floor(start + visLeft * scale));
  const tEnd = Math.min(
    clip.timeline_start + clip.duration,
    Math.max(tStart + 1, Math.ceil(start + visRight * scale)),
  );
  const srcStart = clip.source_in + (tStart - clip.timeline_start);
  const srcEnd = clip.source_in + (tEnd - clip.timeline_start);
  const key = `${clip.source_id}:${String(srcStart)}:${String(srcEnd)}:${String(columns)}:${String(project.revision)}`;

  useEffect(() => {
    if (columns < 1 || tEnd <= tStart) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      client
        .peaks(clip.source_id, srcStart, srcEnd, Math.min(columns, 8192))
        .then((data) => {
          if (!cancelled) {
            setPeaks({ key, data });
            setFailure(undefined);
          }
        })
        .catch((err: unknown) => {
          if (!cancelled) setFailure(err instanceof Error ? err.message : "Waveform unavailable");
        });
    }, 80);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [client, clip.source_id, srcStart, srcEnd, columns, key, tStart, tEnd]);

  useEffect(() => {
    const el = canvas.current;
    if (!el || peaks?.key !== key) return;
    const ctx = el.getContext("2d");
    if (!ctx) return;
    const { data } = peaks;
    const css = getComputedStyle(el);
    ctx.clearRect(0, 0, el.width, el.height);
    ctx.fillStyle = css.getPropertyValue("--wave").trim() || "#5b4bdb";
    const channels = data.channels;
    const bandHeight = el.height / channels;
    for (let ch = 0; ch < channels; ch++) {
      const mins = data.min[ch] ?? [];
      const maxs = data.max[ch] ?? [];
      const mid = bandHeight * ch + bandHeight / 2;
      for (let x = 0; x < mins.length; x++) {
        const lo = ((mins[x] ?? 0) / data.scale) * (bandHeight / 2);
        const hi = ((maxs[x] ?? 0) / data.scale) * (bandHeight / 2);
        const top = mid - hi;
        ctx.fillRect(x, top, 1, Math.max(1, mid - lo - top));
      }
    }
  }, [peaks, key]);

  if (columns < 1) return null;
  return (
    <div
      className={`clip${selected ? " selected" : ""}`}
      style={{ left: x0, width: Math.max(1, x1 - x0) }}
      data-testid="clip"
      data-clip-id={clip.id}
      title={failure}
    >
      <canvas
        ref={canvas}
        width={columns}
        height={LANE_HEIGHT - 8}
        style={{ position: "absolute", left: visLeft - x0, top: 4, width: columns }}
        aria-hidden="true"
      />
      {failure && <span className="clip-error">Waveform unavailable</span>}
    </div>
  );
}
