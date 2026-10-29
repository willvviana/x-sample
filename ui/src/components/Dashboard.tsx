import { useCallback, useEffect, useState } from "react";

import type { EngineClient, SystemInfo } from "../api/client";
import { isActive } from "../jobs";
import { usePolling } from "../usePolling";
import { Workspace } from "../workspace/Workspace";
import { BackendsPanel } from "./BackendsPanel";
import { JobsPanel } from "./JobsPanel";
import { SystemPanel } from "./SystemPanel";

interface Props {
  client: EngineClient;
  onUnauthorized: () => void;
}

export function Dashboard({ client, onUnauthorized }: Props) {
  const [tab, setTab] = useState<"project" | "system">("project");
  const loadSystem = useCallback(() => client.system(), [client]);
  const loadJobs = useCallback(() => client.jobs(), [client]);
  const loadProbes = useCallback(() => client.probes(), [client]);

  const jobs = usePolling(loadJobs, 600);
  const anyActive = jobs.data?.some(isActive) ?? false;
  const system = usePolling<SystemInfo>(loadSystem, anyActive ? 1500 : 5000);
  const probes = usePolling(loadProbes, anyActive ? 1500 : 5000);

  const unauthorized = [system.error, jobs.error, probes.error].some((e) => e?.status === 401);
  useEffect(() => {
    if (unauthorized) onUnauthorized();
  }, [unauthorized, onUnauthorized]);

  const unreachable = system.error?.status === 0;
  const refreshAll = useCallback(() => {
    jobs.refresh();
    system.refresh();
    probes.refresh();
  }, [jobs, system, probes]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">
            ◢◣
          </span>
          <h1>X-Sample</h1>
          <span className="muted">Phase 1 · audio projects</span>
        </div>
        <nav className="tabs" aria-label="Sections">
          {(["project", "system"] as const).map((id) => (
            <button
              key={id}
              type="button"
              className={tab === id ? "tab active" : "tab"}
              aria-current={tab === id ? "page" : undefined}
              onClick={() => {
                setTab(id);
              }}
              data-testid={`tab-${id}`}
            >
              {id === "project" ? "Project" : "System"}
            </button>
          ))}
        </nav>
        <div
          className={`connection ${unreachable ? "bad" : system.data ? "ok" : "pending"}`}
          data-testid="connection"
        >
          {unreachable
            ? "Engine unreachable"
            : system.data
              ? `Engine ${system.data.engine_version} · local`
              : "Connecting…"}
        </div>
      </header>
      {tab === "project" && (
        <main>
          <Workspace client={client} onUnauthorized={onUnauthorized} />
        </main>
      )}
      <main className="grid" hidden={tab !== "system"}>
        <SystemPanel system={system.data} />
        <BackendsPanel
          client={client}
          system={system.data}
          probes={probes.data ?? []}
          jobs={jobs.data ?? []}
          onChanged={refreshAll}
        />
        <JobsPanel
          client={client}
          system={system.data}
          jobs={jobs.data ?? []}
          onChanged={refreshAll}
        />
      </main>
    </div>
  );
}
