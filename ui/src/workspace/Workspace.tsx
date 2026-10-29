import { useCallback, useEffect, useMemo, useRef, useState, type SyntheticEvent } from "react";

import type { ChannelMode, EngineClient, Job, ProjectList, ProjectView } from "../api/client";
import { isActive } from "../jobs";
import { usePolling } from "../usePolling";
import { Recorder } from "./Recorder";
import { SamplesPanel } from "./SamplesPanel";
import { Timeline, type Selection } from "./Timeline";
import { usePlayback } from "./playback";
import { clamp, formatDuration } from "./time";

interface Props {
  client: EngineClient;
  onUnauthorized: () => void;
}

const SAMPLE_RATES = [44100, 48000, 88200, 96000, 32000, 22050];
const PREVIEW_LIMIT_S = 120;
const AUDIO_KINDS = new Set(["import_audio", "export_audio", "record_audio", "verify_project"]);

const KIND_LABEL: Record<string, string> = {
  import_audio: "Import",
  export_audio: "Export",
  record_audio: "Recording",
  verify_project: "Verify",
};

export function Workspace({ client, onUnauthorized }: Props) {
  const loadProject = useCallback(() => client.currentProject(), [client]);
  const loadJobs = useCallback(() => client.jobs(), [client]);
  const projectPoll = usePolling(loadProject, 800);
  const jobsPoll = usePolling(loadJobs, 500);
  const [error, setError] = useState<string>();
  const [playhead, setPlayhead] = useState(0);
  const [selection, setSelection] = useState<Selection>();
  const [clipId, setClipId] = useState<string>();
  const playbackStart = useRef(0);
  const playback = usePlayback();

  const project = projectPoll.data ?? undefined;
  const jobs = useMemo(() => jobsPoll.data ?? [], [jobsPoll.data]);

  const unauthorized = [projectPoll.error, jobsPoll.error].some((e) => e?.status === 401);
  useEffect(() => {
    if (unauthorized) onUnauthorized();
  }, [unauthorized, onUnauthorized]);

  // Drop selection state that no longer refers to anything after edits or project changes.
  useEffect(() => {
    if (!project) {
      setSelection(undefined);
      setClipId(undefined);
      setPlayhead(0);
      return;
    }
    if (clipId && !project.clips.some((c) => c.id === clipId)) setClipId(undefined);
    if (selection) {
      const clip = project.clips.find(
        (c) =>
          c.track_id === selection.trackId &&
          selection.start >= c.timeline_start &&
          selection.end <= c.timeline_start + c.duration,
      );
      if (!clip) setSelection(undefined);
    }
  }, [project, clipId, selection]);

  const report = useCallback((message: string) => {
    setError(message);
  }, []);

  const apply = useCallback(
    async (action: () => Promise<unknown>) => {
      setError(undefined);
      try {
        await action();
        projectPoll.refresh();
        jobsPoll.refresh();
      } catch (err) {
        report(err instanceof Error ? err.message : String(err));
      }
    },
    [projectPoll, jobsPoll, report],
  );

  const rate = project?.sample_rate ?? 48000;
  const total = project?.length_samples ?? 0;
  const activeClip = useMemo(() => {
    if (!project) return undefined;
    if (selection) {
      return project.clips.find(
        (c) =>
          c.track_id === selection.trackId &&
          selection.start >= c.timeline_start &&
          selection.end <= c.timeline_start + c.duration,
      );
    }
    return project.clips.find((c) => c.id === clipId);
  }, [project, selection, clipId]);

  const togglePlay = useCallback(() => {
    if (playback.playing) {
      playback.stop();
      return;
    }
    if (!project || total === 0) return;
    const limit = PREVIEW_LIMIT_S * rate;
    let start = selection ? selection.start : playhead >= total ? 0 : playhead;
    let end = selection ? selection.end : total;
    end = Math.min(end, start + limit);
    start = clamp(start, 0, Math.max(0, end - 1));
    playbackStart.current = start;
    client
      .renderMix(start, end)
      .then((bytes) => playback.play(bytes))
      .catch((err: unknown) => {
        report(err instanceof Error ? err.message : String(err));
      });
  }, [client, playback, project, total, rate, selection, playhead, report]);

  // The playhead follows the audible position while playing.
  const livePlayhead =
    playback.playing && playback.elapsed !== null
      ? Math.round(playbackStart.current + playback.elapsed * rate)
      : playhead;

  const previewSample = (sampleId: string) => {
    client
      .renderSample(sampleId)
      .then((bytes) => playback.play(bytes))
      .catch((err: unknown) => {
        report(err instanceof Error ? err.message : String(err));
      });
  };

  const audioJobs = jobs.filter((j) => AUDIO_KINDS.has(j.kind));
  const busy = audioJobs.some(isActive);

  if (!project) {
    return (
      <ProjectStart
        client={client}
        busy={busy}
        error={error}
        onDone={() => {
          setError(undefined);
          projectPoll.refresh();
        }}
        onError={report}
      />
    );
  }

  return (
    <div className="workspace" data-testid="workspace">
      <section className="panel full project-bar" aria-label="Project">
        <div>
          <h2 data-testid="project-name">{project.name}</h2>
          <p className="muted small">
            {project.sample_rate} Hz · {formatDuration(total, rate)} · {project.tracks.length}{" "}
            track(s) · revision {project.revision}
          </p>
        </div>
        <div className="row">
          <button type="button" onClick={() => void apply(() => client.saveProject())}>
            Save
          </button>
          <button
            type="button"
            className="secondary"
            onClick={() => void apply(() => client.verifyProject())}
          >
            Verify files
          </button>
          <button
            type="button"
            className="secondary"
            disabled={busy}
            onClick={() => {
              playback.stop();
              void apply(() => client.closeProject());
            }}
            data-testid="close-project"
          >
            Close project
          </button>
        </div>
      </section>

      {project.issues.length > 0 && (
        <div className="notice warn full" role="alert" data-testid="project-issues">
          {project.issues.map((issue) => (
            <p key={issue}>{issue}</p>
          ))}
          {project.sources
            .filter((s) => s.status === "needs_rebuild")
            .map((s) => (
              <button
                key={s.id}
                type="button"
                className="secondary"
                onClick={() => void apply(() => client.rebuildSource(s.id))}
              >
                Rebuild {s.original_name}
              </button>
            ))}
        </div>
      )}

      {error && (
        <div className="notice bad full" role="alert" data-testid="workspace-error">
          {error}
          <button
            type="button"
            className="secondary"
            onClick={() => {
              setError(undefined);
            }}
          >
            Dismiss
          </button>
        </div>
      )}

      <section className="panel full" aria-labelledby="tl-title">
        <div className="panel-head">
          <h2 id="tl-title">Timeline</h2>
          <div className="row">
            <button
              type="button"
              onClick={togglePlay}
              disabled={total === 0}
              data-testid="play"
              aria-pressed={playback.playing}
            >
              {playback.playing ? "Pause" : "Play"}
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => {
                playback.stop();
                setPlayhead(0);
              }}
            >
              To start
            </button>
          </div>
        </div>
        {total === 0 ? (
          <p className="muted" data-testid="empty-timeline">
            This project has no audio yet. Import a file or record below.
          </p>
        ) : (
          <>
            <Timeline
              client={client}
              project={project}
              playhead={livePlayhead}
              selection={selection}
              selectedClipId={activeClip?.id}
              onPlayhead={(s) => {
                if (playback.playing) playback.stop();
                setPlayhead(s);
              }}
              onSelect={setSelection}
              onPickClip={setClipId}
              onTogglePlay={togglePlay}
            />
            {playback.error && <p className="notice bad">Playback failed: {playback.error}</p>}
            <EditBar
              disabled={busy}
              hasSelection={selection !== undefined}
              clipSelected={activeClip !== undefined}
              onTrim={() => {
                if (activeClip && selection) {
                  void apply(() => client.trimClip(activeClip.id, selection.start, selection.end));
                  setSelection(undefined);
                }
              }}
              onSlice={(name) => {
                if (activeClip && selection) {
                  void apply(() =>
                    client.sliceClip(activeClip.id, selection.start, selection.end, name),
                  );
                }
              }}
              onReset={() => {
                if (activeClip) void apply(() => client.resetTrim(activeClip.id));
              }}
              onDelete={() => {
                if (activeClip) void apply(() => client.deleteClip(activeClip.id));
              }}
              onExport={() => {
                if (selection) {
                  void apply(() =>
                    client.exportAudio({
                      kind: "selection",
                      start: selection.start,
                      end: selection.end,
                      name: `${project.name} selection`,
                    }),
                  );
                }
              }}
            />
          </>
        )}
      </section>

      <TrackList client={client} project={project} apply={apply} />

      <SamplesPanel
        client={client}
        project={project}
        playhead={playhead}
        onPreview={previewSample}
        onChanged={() => {
          projectPoll.refresh();
          jobsPoll.refresh();
        }}
        onError={report}
      />

      <ImportPanel client={client} disabled={false} apply={apply} />
      <Recorder client={client} jobs={jobs} onError={report} />

      <JobStrip jobs={audioJobs} client={client} />
    </div>
  );
}

function EditBar(props: {
  disabled: boolean;
  hasSelection: boolean;
  clipSelected: boolean;
  onTrim: () => void;
  onSlice: (name: string | undefined) => void;
  onReset: () => void;
  onDelete: () => void;
  onExport: () => void;
}) {
  const [name, setName] = useState("");
  return (
    <div className="edit-bar row" role="toolbar" aria-label="Edit selection">
      <button
        type="button"
        onClick={props.onTrim}
        disabled={!props.hasSelection}
        data-testid="trim"
      >
        Trim clip to selection
      </button>
      <input
        aria-label="Sample name"
        placeholder="Sample name (optional)"
        value={name}
        maxLength={120}
        onChange={(e) => {
          setName(e.target.value);
        }}
      />
      <button
        type="button"
        onClick={() => {
          props.onSlice(name.trim() || undefined);
          setName("");
        }}
        disabled={!props.hasSelection}
        data-testid="slice"
      >
        Slice to sample
      </button>
      <button
        type="button"
        className="secondary"
        onClick={props.onExport}
        disabled={!props.hasSelection || props.disabled}
        data-testid="export-selection"
      >
        Export selection
      </button>
      <button
        type="button"
        className="secondary"
        onClick={props.onReset}
        disabled={!props.clipSelected}
        data-testid="reset-trim"
      >
        Reset trim
      </button>
      <button
        type="button"
        className="secondary"
        onClick={props.onDelete}
        disabled={!props.clipSelected}
      >
        Remove clip
      </button>
      <p className="small muted full">
        Drag on a clip to select a region. Trimming and slicing never change the original file.
      </p>
    </div>
  );
}

function TrackList({
  client,
  project,
  apply,
}: {
  client: EngineClient;
  project: ProjectView;
  apply: (action: () => Promise<unknown>) => Promise<void>;
}) {
  return (
    <section className="panel" aria-labelledby="tracks-title" data-testid="tracks-panel">
      <div className="panel-head">
        <h2 id="tracks-title">Tracks</h2>
        <button type="button" onClick={() => void apply(() => client.addTrack())}>
          Add track
        </button>
      </div>
      <ul className="track-list">
        {project.tracks.map((track) => (
          <li key={track.id} data-testid="track-item">
            <input
              aria-label={`Name of ${track.name}`}
              defaultValue={track.name}
              maxLength={120}
              onBlur={(e) => {
                const name = e.target.value.trim();
                if (name && name !== track.name) {
                  void apply(() => client.patchTrack(track.id, { name }));
                }
              }}
            />
            <label className="check">
              <input
                type="checkbox"
                checked={track.muted}
                onChange={(e) =>
                  void apply(() => client.patchTrack(track.id, { muted: e.target.checked }))
                }
              />
              Mute
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={track.solo}
                onChange={(e) =>
                  void apply(() => client.patchTrack(track.id, { solo: e.target.checked }))
                }
              />
              Solo
            </label>
            <label className="gain">
              Gain
              <input
                type="range"
                min={0}
                max={2}
                step={0.05}
                defaultValue={track.gain}
                onPointerUp={(e) => {
                  const gain = Number(e.currentTarget.value);
                  void apply(() => client.patchTrack(track.id, { gain }));
                }}
                onKeyUp={(e) => {
                  const gain = Number(e.currentTarget.value);
                  void apply(() => client.patchTrack(track.id, { gain }));
                }}
              />
            </label>
            <button
              type="button"
              className="secondary"
              aria-label={`Delete track ${track.name}`}
              disabled={project.tracks.length <= 1}
              onClick={() => void apply(() => client.deleteTrack(track.id))}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function ImportPanel({
  client,
  disabled,
  apply,
}: {
  client: EngineClient;
  disabled: boolean;
  apply: (action: () => Promise<unknown>) => Promise<void>;
}) {
  const [mode, setMode] = useState<ChannelMode>("keep");
  const input = useRef<HTMLInputElement>(null);
  return (
    <section className="panel" aria-labelledby="import-title" data-testid="import-panel">
      <div className="panel-head">
        <h2 id="import-title">Import audio</h2>
      </div>
      <p className="small muted">
        WAV, MP3, FLAC, OGG and M4A. The original file is kept untouched inside the project; a
        working copy is made at the project sample rate.
      </p>
      <label>
        Channels
        <select
          value={mode}
          onChange={(e) => {
            setMode(e.target.value as ChannelMode);
          }}
        >
          <option value="keep">Keep (mono stays mono, multichannel → stereo)</option>
          <option value="mono">Convert to mono</option>
          <option value="stereo">Convert to stereo</option>
        </select>
      </label>
      <label>
        Audio file
        <input
          ref={input}
          type="file"
          accept=".wav,.mp3,.flac,.ogg,.oga,.opus,.m4a,.aac,.aif,.aiff,audio/*"
          disabled={disabled}
          data-testid="import-input"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            void apply(() => client.importFile(file, mode)).finally(() => {
              if (input.current) input.current.value = "";
            });
          }}
        />
      </label>
    </section>
  );
}

function JobStrip({ jobs, client }: { jobs: Job[]; client: EngineClient }) {
  const recent = jobs.slice(0, 6);
  if (recent.length === 0) return null;
  return (
    <section className="panel full" aria-labelledby="jobs-strip-title" data-testid="job-strip">
      <h2 id="jobs-strip-title">Activity</h2>
      <ul className="activity">
        {recent.map((job) => (
          <li key={job.id} data-testid="activity-item" data-state={job.state}>
            <span className="kind">{KIND_LABEL[job.kind] ?? job.kind}</span>
            <span className={`state-${job.state}`}>{job.state}</span>
            {isActive(job) && job.kind !== "record_audio" && (
              <progress value={job.progress} max={1} aria-label={`${job.kind} progress`} />
            )}
            <span className="muted small">{job.error ?? job.message ?? ""}</span>
            {isActive(job) && job.kind !== "record_audio" && (
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  client.cancelJob(job.id).catch(() => undefined);
                }}
              >
                Cancel
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ProjectStart({
  client,
  busy,
  error,
  onDone,
  onError,
}: {
  client: EngineClient;
  busy: boolean;
  error: string | undefined;
  onDone: () => void;
  onError: (message: string) => void;
}) {
  const list = usePolling<ProjectList>(
    useCallback(() => client.listProjects(), [client]),
    4000,
  );
  const [name, setName] = useState("");
  const [rate, setRate] = useState(48000);
  const [working, setWorking] = useState(false);

  const act = async (action: () => Promise<unknown>) => {
    setWorking(true);
    try {
      await action();
      onDone();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setWorking(false);
    }
  };

  const create = (event: SyntheticEvent) => {
    event.preventDefault();
    void act(() => client.createProject(name.trim(), rate));
  };

  return (
    <div className="workspace start" data-testid="project-start">
      <section className="panel" aria-labelledby="new-title">
        <h2 id="new-title">New project</h2>
        <form onSubmit={create} className="stack">
          <label>
            Name
            <input
              value={name}
              maxLength={120}
              required
              onChange={(e) => {
                setName(e.target.value);
              }}
              data-testid="project-name-input"
            />
          </label>
          <label>
            Sample rate
            <select
              value={rate}
              onChange={(e) => {
                setRate(Number(e.target.value));
              }}
              data-testid="project-rate"
            >
              {SAMPLE_RATES.map((r) => (
                <option key={r} value={r}>
                  {r} Hz
                </option>
              ))}
            </select>
          </label>
          <p className="small muted">
            Imported audio is converted to this rate for editing. Originals are never altered.
          </p>
          <button
            type="submit"
            disabled={working || busy || !name.trim()}
            data-testid="create-project"
          >
            Create project
          </button>
        </form>
        {list.data && (
          <p className="small muted">Projects are stored in {list.data.projects_dir}</p>
        )}
      </section>
      <section className="panel" aria-labelledby="open-title">
        <h2 id="open-title">Open project</h2>
        {list.data?.projects.length ? (
          <ul className="sample-list" data-testid="project-list">
            {list.data.projects.map((p) => (
              <li key={p.path}>
                <span>{p.name}</span>
                <span className="muted small">{p.path}</span>
                <button
                  type="button"
                  disabled={working}
                  onClick={() => void act(() => client.openProject(p.path))}
                >
                  Open
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">No projects yet.</p>
        )}
      </section>
      {error && (
        <div className="notice bad full" role="alert" data-testid="workspace-error">
          {error}
        </div>
      )}
    </div>
  );
}
