import { useEffect, useMemo, useState } from "react";

import type { AudioDevice, AudioDevices, EngineClient, Job } from "../api/client";
import { isActive } from "../jobs";
import { formatDb } from "./time";

interface Props {
  client: EngineClient;
  jobs: Job[];
  onError: (message: string) => void;
}

interface Telemetry {
  elapsed_s?: number;
  captured_s?: number;
  peak_db?: number[];
  rms_db?: number[];
  hold_db?: number[];
  clipping?: boolean;
  clip_samples?: number;
  dropped_blocks?: number;
  device?: string;
}

const MAX_SECONDS = 3600;

export function Recorder({ client, jobs, onError }: Props) {
  const [devices, setDevices] = useState<AudioDevices>();
  const [choice, setChoice] = useState<string>("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = () => {
      client
        .audioDevices()
        .then((d) => {
          if (alive) setDevices(d);
        })
        .catch(() => undefined);
    };
    load();
    const timer = window.setInterval(load, 2000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, [client]);

  const all = useMemo<AudioDevice[]>(
    () => [...(devices?.microphones ?? []), ...(devices?.loopbacks ?? [])],
    [devices],
  );
  const selected = all.find((d) => `${d.kind}:${String(d.index)}` === choice);

  useEffect(() => {
    if (selected || all.length === 0) return;
    const preferred = all.find((d) => d.kind === "microphone" && d.is_default) ?? all[0];
    if (preferred) setChoice(`${preferred.kind}:${String(preferred.index)}`);
  }, [all, selected]);

  const recording = jobs.find((j) => j.kind === "record_audio" && isActive(j));
  const lastFailed = jobs.find((j) => j.kind === "record_audio" && j.state === "failed");
  const telemetry = (recording?.data ?? {}) as Telemetry;

  const start = async () => {
    if (!selected) return;
    setBusy(true);
    try {
      await client.startRecording(selected, MAX_SECONDS);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const finish = async (action: "stop" | "cancel") => {
    if (!recording) return;
    try {
      if (action === "stop") await client.stopJob(recording.id);
      else await client.cancelJob(recording.id);
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section className="panel" aria-labelledby="rec-title" data-testid="recorder">
      <div className="panel-head">
        <h2 id="rec-title">Record</h2>
        <button
          type="button"
          className="secondary"
          disabled={recording !== undefined}
          onClick={() => {
            client.refreshDevices().catch((e: unknown) => {
              onError(e instanceof Error ? e.message : String(e));
            });
          }}
        >
          Refresh devices
        </button>
      </div>
      <p className="small muted">
        Recording happens only on this computer, only when you press Start, and is saved into the
        open project.
      </p>
      <label>
        Source
        <select
          value={choice}
          disabled={recording !== undefined || all.length === 0}
          onChange={(e) => {
            setChoice(e.target.value);
          }}
          data-testid="device-select"
        >
          {all.length === 0 && (
            <option value="">
              {devices?.state === "failed"
                ? "No devices (listing failed)"
                : devices?.state === "ready"
                  ? "No devices found"
                  : "Looking for devices…"}
            </option>
          )}
          <optgroup label="Microphones">
            {(devices?.microphones ?? []).map((d) => (
              <option key={`m${String(d.index)}`} value={`${d.kind}:${String(d.index)}`}>
                {d.name}
              </option>
            ))}
          </optgroup>
          <optgroup label="System audio (WASAPI loopback)">
            {(devices?.loopbacks ?? []).map((d) => (
              <option key={`l${String(d.index)}`} value={`${d.kind}:${String(d.index)}`}>
                {d.name}
              </option>
            ))}
          </optgroup>
        </select>
      </label>
      {devices?.error && <p className="notice bad">{devices.error}</p>}
      <div className="row">
        {!recording ? (
          <button
            type="button"
            onClick={() => void start()}
            disabled={!selected || busy}
            data-testid="record-start"
          >
            ● Start recording
          </button>
        ) : (
          <>
            <button type="button" onClick={() => void finish("stop")} data-testid="record-stop">
              ■ Stop and save
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => void finish("cancel")}
              data-testid="record-cancel"
            >
              Discard
            </button>
          </>
        )}
      </div>
      {recording && (
        <div className="meters" data-testid="meters" aria-live="off">
          <p>
            <strong data-testid="record-time">{(telemetry.captured_s ?? 0).toFixed(1)} s</strong>{" "}
            captured from <em>{telemetry.device ?? selected?.name}</em>
          </p>
          {(telemetry.peak_db ?? []).map((peak, ch) => (
            <Meter
              key={ch}
              label={ch === 0 ? "L" : ch === 1 ? "R" : String(ch + 1)}
              peak={peak}
              hold={telemetry.hold_db?.[ch]}
            />
          ))}
          {telemetry.clipping && (
            <p className="notice bad" role="alert" data-testid="clip-warning">
              Clipping detected ({telemetry.clip_samples ?? 0} samples). Lower the input level.
            </p>
          )}
          {(telemetry.dropped_blocks ?? 0) > 0 && (
            <p className="notice warn">
              {telemetry.dropped_blocks} audio blocks were dropped; the recording has gaps.
            </p>
          )}
        </div>
      )}
      {!recording && lastFailed?.error && (
        <p className="notice bad" role="alert" data-testid="record-error">
          Recording failed: {lastFailed.error}
        </p>
      )}
    </section>
  );
}

function Meter({ label, peak, hold }: { label: string; peak: number; hold: number | undefined }) {
  // −60 dB … 0 dB mapped to the bar width.
  const pct = (db: number) => Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
  return (
    <div className="meter-row">
      <span className="meter-label">{label}</span>
      <div
        className="meter"
        role="meter"
        aria-label={`Input level ${label}`}
        aria-valuemin={-60}
        aria-valuemax={0}
        aria-valuenow={Math.round(Number.isFinite(peak) ? peak : -60)}
      >
        <div className="meter-fill" style={{ width: `${String(pct(peak))}%` }} />
        {hold !== undefined && (
          <div className="meter-hold" style={{ left: `${String(pct(hold))}%` }} />
        )}
      </div>
      <span className="meter-value">{formatDb(peak)}</span>
    </div>
  );
}
