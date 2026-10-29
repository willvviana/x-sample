import { useState } from "react";

import type { BackendId, EngineClient, Job, ProbeRecord, SystemInfo } from "../api/client";
import { formatBytes, formatMs, formatTime } from "../format";
import { isActive } from "../jobs";
import { StateBadge } from "./StateBadge";

interface Props {
  client: EngineClient;
  system: SystemInfo | undefined;
  probes: ProbeRecord[];
  jobs: Job[];
  onChanged: () => void;
}

const LABELS: Record<BackendId, string> = {
  cpu: "CPU",
  directml: "DirectML",
  vulkan: "Vulkan",
};

export function BackendsPanel({ client, system, probes, jobs, onChanged }: Props) {
  const [error, setError] = useState<string>();
  const detection = system?.detection;
  const detecting = detection?.state === "running" || detection?.state === "pending";

  const run = async (action: () => Promise<unknown>) => {
    setError(undefined);
    try {
      await action();
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section className="panel wide" aria-labelledby="backends-title" data-testid="backends-panel">
      <div className="panel-head">
        <h2 id="backends-title">Backends</h2>
        <div className="row">
          <span className="muted small" data-testid="detection-state">
            Detection: {detection?.state ?? "…"}
            {detection?.updated_at ? ` · ${formatTime(detection.updated_at)}` : ""}
          </span>
          <button
            type="button"
            className="secondary"
            disabled={detecting}
            onClick={() => void run(() => client.detect())}
          >
            Re-detect
          </button>
        </div>
      </div>
      <p className="muted small">
        A detected backend is not assumed to run any model. Support is claimed only after a probe
        runs a reference model and matches the CPU/NumPy result.
      </p>
      {error && (
        <p className="notice" role="alert">
          {error}
        </p>
      )}
      <div className="backends">
        {(system?.backends ?? []).map((backend) => {
          const probe = probes.find((p) => p.backend === backend.id);
          const probing = jobs.some(
            (j) => j.kind === "probe" && j.params.backend === backend.id && isActive(j),
          );
          return (
            <article className="backend" key={backend.id} data-testid={`backend-${backend.id}`}>
              <header>
                <h3>{LABELS[backend.id]}</h3>
                <StateBadge state={backend.state} testId={`backend-${backend.id}-state`} />
              </header>
              {backend.devices.length > 0 && <p className="small">{backend.devices.join(", ")}</p>}
              <p className="muted small">{backend.detail ?? "—"}</p>
              <p className="small" data-testid={`backend-${backend.id}-support`}>
                Model support:{" "}
                <strong>
                  {backend.claims_model_support ? "verified by probe" : "not claimed"}
                </strong>
              </p>
              <ProbeSummary probe={probe} />
              <button
                type="button"
                disabled={probing || backend.state !== "available"}
                onClick={() => void run(() => client.runProbe(backend.id, Boolean(probe)))}
              >
                {probing ? "Probing…" : probe ? "Re-probe" : "Probe"}
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function ProbeSummary({ probe }: { probe: ProbeRecord | undefined }) {
  if (!probe) return <p className="muted small">Not probed yet.</p>;
  return (
    <div className="probe" data-testid={`probe-${probe.backend}`}>
      <div className="row">
        <span className="small">Probe</span>
        <StateBadge state={probe.status} testId={`probe-${probe.backend}-status`} />
        <span className="muted small">{formatTime(probe.probed_at)}</span>
      </div>
      {probe.status === "unsupported" || probe.detail ? (
        <p className="muted small">{probe.detail}</p>
      ) : null}
      {probe.max_abs_error !== null && probe.max_abs_error !== undefined && (
        <ul className="metrics small">
          <li>max error {probe.max_abs_error.toExponential(1)}</li>
          <li>chunk {formatMs(probe.chunk_ms_mean)}</li>
          {probe.adapter_used && <li>ran on {probe.adapter_used}</li>}
          {probe.gpu_dedicated_peak_bytes ? (
            <li>GPU mem {formatBytes(probe.gpu_dedicated_peak_bytes)}</li>
          ) : null}
          <li>
            fallback:{" "}
            {probe.fallback_nodes && probe.fallback_nodes.length > 0
              ? probe.fallback_nodes.join(", ")
              : "none"}
          </li>
        </ul>
      )}
    </div>
  );
}
