import { useState, type SyntheticEvent } from "react";

import type { EngineClient, Job, SystemInfo } from "../api/client";
import { formatBytes, formatMs, formatTime } from "../format";
import { isActive } from "../jobs";
import { StateBadge } from "./StateBadge";

interface Props {
  client: EngineClient;
  system: SystemInfo | undefined;
  jobs: Job[];
  onChanged: () => void;
}

type SmokeBackend = "cpu" | "directml";

export function JobsPanel({ client, system, jobs, onChanged }: Props) {
  const [backend, setBackend] = useState<SmokeBackend>("directml");
  const [seconds, setSeconds] = useState(60);
  const [strict, setStrict] = useState(true);
  const [error, setError] = useState<string>();

  const directmlReady = system?.backends.find((b) => b.id === "directml")?.state === "available";

  const run = async (action: () => Promise<unknown>) => {
    setError(undefined);
    try {
      await action();
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const submit = (event: SyntheticEvent) => {
    event.preventDefault();
    void run(() => client.submitJob("smoke_inference", { backend, seconds, strict }));
  };

  return (
    <section className="panel full" aria-labelledby="jobs-title" data-testid="jobs-panel">
      <div className="panel-head">
        <h2 id="jobs-title">Jobs</h2>
      </div>
      <form className="smoke-form" onSubmit={submit}>
        <label>
          Backend
          <select
            value={backend}
            onChange={(e) => {
              setBackend(e.target.value as SmokeBackend);
            }}
            data-testid="smoke-backend"
          >
            <option value="directml" disabled={!directmlReady}>
              DirectML (GPU)
            </option>
            <option value="cpu">CPU</option>
          </select>
        </label>
        <label>
          Audio length (s)
          <input
            type="number"
            min={1}
            max={600}
            value={seconds}
            onChange={(e) => {
              setSeconds(Math.max(1, Math.min(600, Number(e.target.value) || 1)));
            }}
            data-testid="smoke-seconds"
          />
        </label>
        <label className="check">
          <input
            type="checkbox"
            checked={strict}
            onChange={(e) => {
              setStrict(e.target.checked);
            }}
          />
          Strict (no CPU fallback)
        </label>
        <button type="submit" data-testid="run-smoke">
          Run smoke inference
        </button>
        {system?.test_tasks_enabled && (
          <button
            type="button"
            className="secondary"
            data-testid="run-test-job"
            onClick={() =>
              void run(() => client.submitJob("test.steps", { steps: 300, delay_s: 0.1 }))
            }
          >
            Start slow test job
          </button>
        )}
      </form>
      <p className="muted small">
        Smoke inference runs a tiny locally generated model over synthetic audio in an isolated
        worker process. No models are downloaded and no audio leaves this machine.
      </p>
      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      {jobs.length === 0 ? (
        <p className="muted">No jobs yet.</p>
      ) : (
        <ul className="jobs" data-testid="job-list">
          {jobs.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              onCancel={() => void run(() => client.cancelJob(job.id))}
            />
          ))}
        </ul>
      )}
    </section>
  );
}

function describe(job: Job): string {
  const p = job.params;
  switch (job.kind) {
    case "detect_backends":
      return "Backend detection";
    case "probe":
      return `Probe · ${String(p.backend)}`;
    case "smoke_inference":
      return `Smoke inference · ${String(p.backend)} · ${String(p.seconds)} s`;
    case "test.steps":
      return "Test job";
    default:
      return job.kind;
  }
}

function num(result: Job["result"], key: string): number | undefined {
  const value = result?.[key];
  return typeof value === "number" ? value : undefined;
}

function JobRow({ job, onCancel }: { job: Job; onCancel: () => void }) {
  const active = isActive(job);
  const rtf = num(job.result, "realtime_factor");
  const adapter = job.result?.adapter_used;
  const passed = job.result?.passed;
  return (
    <li className="job" data-testid="job" data-job-id={job.id} data-job-kind={job.kind}>
      <div className="job-head">
        <span className="job-title">{describe(job)}</span>
        <StateBadge state={job.state} testId="job-state" />
        <span className="muted small">{formatTime(job.created_at)}</span>
        {active && (
          <button
            type="button"
            className="secondary small-btn"
            onClick={onCancel}
            disabled={job.state === "cancelling"}
            data-testid="cancel-job"
          >
            Cancel
          </button>
        )}
      </div>
      {active && (
        <div className="progress" role="progressbar" aria-valuenow={Math.round(job.progress * 100)}>
          <div className="bar" style={{ width: `${(job.progress * 100).toFixed(1)}%` }} />
        </div>
      )}
      {active && job.message && <p className="muted small">{job.message}</p>}
      {job.kind === "smoke_inference" && job.state === "succeeded" && (
        <ul className="metrics small" data-testid="smoke-metrics">
          <li>{passed === true ? "verified ✓" : "verification failed"}</li>
          {typeof adapter === "string" && <li>ran on {adapter}</li>}
          {rtf !== undefined && <li>{Math.round(rtf)}× realtime</li>}
          <li>chunk {formatMs(num(job.result, "chunk_ms_mean"))}</li>
          <li>max error {num(job.result, "max_abs_error_vs_numpy")?.toExponential(1) ?? "—"}</li>
          <li>worker peak {formatBytes(job.peak_worker_memory_bytes)}</li>
        </ul>
      )}
      {job.state === "failed" && (
        <p className="notice small" data-testid="job-error">
          {job.error}
          {job.exit_code !== null ? ` (exit code ${job.exit_code})` : ""}
        </p>
      )}
      {job.state === "cancelled" && job.forced_termination && (
        <p className="muted small">Worker did not stop in time and was terminated.</p>
      )}
    </li>
  );
}
