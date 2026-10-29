import { useState } from "react";

import type { EngineClient, ProjectView } from "../api/client";
import { formatDuration } from "./time";

interface Props {
  client: EngineClient;
  project: ProjectView;
  playhead: number;
  onPreview: (sampleId: string) => void;
  onChanged: (project: ProjectView) => void;
  onError: (message: string) => void;
}

export function SamplesPanel({ client, project, playhead, onPreview, onChanged, onError }: Props) {
  const [format, setFormat] = useState<"wav" | "flac">("wav");
  const [subtype, setSubtype] = useState<"PCM_16" | "PCM_24" | "FLOAT">("PCM_24");

  const run = async (action: () => Promise<unknown>) => {
    try {
      const result = await action();
      if (result && typeof result === "object" && "tracks" in result) {
        onChanged(result as ProjectView);
      }
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  const rate = project.sample_rate;
  return (
    <section className="panel" aria-labelledby="samples-title" data-testid="samples-panel">
      <div className="panel-head">
        <h2 id="samples-title">Samples</h2>
        <div className="row">
          <select
            aria-label="Export format"
            value={format}
            onChange={(e) => {
              setFormat(e.target.value as "wav" | "flac");
              if (e.target.value === "flac" && subtype === "FLOAT") setSubtype("PCM_24");
            }}
          >
            <option value="wav">WAV</option>
            <option value="flac">FLAC</option>
          </select>
          <select
            aria-label="Bit depth"
            value={subtype}
            onChange={(e) => {
              setSubtype(e.target.value as "PCM_16" | "PCM_24" | "FLOAT");
            }}
          >
            <option value="PCM_16">16-bit</option>
            <option value="PCM_24">24-bit</option>
            {format === "wav" && <option value="FLOAT">32-bit float</option>}
          </select>
        </div>
      </div>
      {project.samples.length === 0 ? (
        <p className="muted">
          No samples yet. Select a region on a clip and choose “Slice to sample”.
        </p>
      ) : (
        <ul className="sample-list">
          {project.samples.map((sample) => (
            <li key={sample.id} data-testid="sample-item">
              <input
                aria-label={`Name of ${sample.name}`}
                defaultValue={sample.name}
                maxLength={120}
                onBlur={(e) => {
                  const name = e.target.value.trim();
                  if (name && name !== sample.name) {
                    void run(() => client.renameSample(sample.id, name));
                  }
                }}
              />
              <span className="muted small">{formatDuration(sample.duration, rate)}</span>
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  onPreview(sample.id);
                }}
              >
                Preview
              </button>
              <button
                type="button"
                className="secondary"
                onClick={() => void run(() => client.placeSample(sample.id, null, playhead))}
              >
                To new track
              </button>
              <button
                type="button"
                onClick={() =>
                  void run(() =>
                    client.exportAudio({ kind: "sample", sample_id: sample.id, format, subtype }),
                  )
                }
                data-testid="export-sample"
              >
                Export
              </button>
              <button
                type="button"
                className="secondary"
                aria-label={`Delete ${sample.name}`}
                onClick={() => void run(() => client.deleteSample(sample.id))}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
      <h3>Exports</h3>
      {project.exports.length === 0 ? (
        <p className="muted">
          Exported files appear here and are stored in the project’s exports folder.
        </p>
      ) : (
        <ul className="sample-list" data-testid="export-list">
          {project.exports.map((e) => (
            <li key={e.id}>
              <span>{e.filename}</span>
              <span className="muted small">
                {formatDuration(e.frames, e.sample_rate)} · {e.subtype} ·{" "}
                {(e.size_bytes / 1024).toFixed(0)} KiB
              </span>
              <button
                type="button"
                onClick={() => {
                  client.downloadExport(e.filename).catch((err: unknown) => {
                    onError(err instanceof Error ? err.message : String(err));
                  });
                }}
                data-testid="download-export"
              >
                Download
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
