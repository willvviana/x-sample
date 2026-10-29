import type { SystemInfo } from "../api/client";
import { formatBytes } from "../format";

export function SystemPanel({ system }: { system: SystemInfo | undefined }) {
  return (
    <section className="panel" aria-labelledby="system-title" data-testid="system-panel">
      <h2 id="system-title">System</h2>
      {!system ? (
        <p className="muted">Loading…</p>
      ) : (
        <dl className="facts">
          <dt>CPU</dt>
          <dd>
            {system.cpu.name ?? "Unknown"}
            <span className="muted">
              {" "}
              · {system.cpu.cores ?? "?"} cores / {system.cpu.threads ?? "?"} threads
            </span>
          </dd>
          <dt>Memory</dt>
          <dd>
            {formatBytes(system.memory.total_bytes)}
            <span className="muted"> · {formatBytes(system.memory.available_bytes)} free</span>
          </dd>
          {system.gpus.map((gpu) => (
            <GpuRow
              key={gpu.name}
              name={gpu.name}
              vram={gpu.vram_bytes}
              driver={gpu.driver_version}
            />
          ))}
          <dt>Budgets</dt>
          <dd>
            VRAM {formatBytes(system.budgets.vram_budget_bytes)} · worker RAM{" "}
            {formatBytes(system.budgets.worker_memory_limit_bytes)} ·{" "}
            {system.budgets.max_concurrent_jobs} parallel jobs
          </dd>
          <dt>Runtime</dt>
          <dd>
            Python {system.python_version}
            {system.detection.runtime_version && (
              <> · ONNX Runtime {system.detection.runtime_version}</>
            )}
          </dd>
          <dt>OS</dt>
          <dd className="small">{system.os}</dd>
        </dl>
      )}
    </section>
  );
}

function GpuRow({
  name,
  vram,
  driver,
}: {
  name: string;
  vram: number | null;
  driver: string | null;
}) {
  return (
    <>
      <dt>GPU</dt>
      <dd data-testid="gpu">
        {name}
        <span className="muted">
          {" "}
          · {formatBytes(vram)} VRAM{driver ? ` · driver ${driver}` : ""}
        </span>
      </dd>
    </>
  );
}
