import type { components } from "./schema";

export type SystemInfo = components["schemas"]["SystemInfo"];
export type BackendStatus = components["schemas"]["BackendStatus"];
export type ProbeRecord = components["schemas"]["ProbeRecord"];
export type Job = components["schemas"]["JobOut"];
export type BackendId = BackendStatus["id"];
export type ProjectView = components["schemas"]["ProjectView"];
export type ProjectList = components["schemas"]["ProjectList"];
export type Track = components["schemas"]["Track"];
export type Clip = components["schemas"]["Clip"];
export type SampleItem = components["schemas"]["Sample"];
export type AudioDevices = components["schemas"]["AudioDevices"];
export type AudioDevice = components["schemas"]["AudioDevice"];
export type Peaks = components["schemas"]["PeaksOut"];
export type ChannelMode = "keep" | "mono" | "stereo";
export type AnalysisView = components["schemas"]["AnalysisView"];
export type AnalysisResult = components["schemas"]["AnalysisResult"];
export type AnalysisCorrections = components["schemas"]["AnalysisCorrections"];
export type EffectiveAnalysis = components["schemas"]["EffectiveAnalysis"];
export type ChordOut = components["schemas"]["ChordOut"];
export type TagOut = components["schemas"]["TagOut"];

// Same origin when the engine serves the built UI; the engine port during `vite` dev.
const ENGINE_URL: string =
  (import.meta.env.VITE_ENGINE_URL as string | undefined) ??
  (import.meta.env.DEV ? "http://127.0.0.1:8765" : "");

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export class EngineClient {
  constructor(private readonly token: string) {}

  private async raw(path: string, init: RequestInit = {}, contentType?: string): Promise<Response> {
    let response: Response;
    try {
      response = await fetch(`${ENGINE_URL}${path}`, {
        ...init,
        headers: {
          Authorization: `Bearer ${this.token}`,
          ...(contentType ? { "Content-Type": contentType } : {}),
        },
        credentials: "omit",
        cache: "no-store",
      });
    } catch {
      throw new ApiError(0, "Engine unreachable");
    }
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = (await response.json()) as { detail?: unknown };
        if (typeof body.detail === "string") detail = body.detail;
      } catch {
        // keep statusText
      }
      throw new ApiError(response.status, detail);
    }
    return response;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const response = await this.raw(path, init, init.body ? "application/json" : undefined);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  private post<T>(path: string, body?: unknown): Promise<T> {
    return this.request(path, {
      method: "POST",
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    });
  }

  private patch<T>(path: string, body: unknown): Promise<T> {
    return this.request(path, { method: "PATCH", body: JSON.stringify(body) });
  }

  private del<T>(path: string): Promise<T> {
    return this.request(path, { method: "DELETE" });
  }

  private put<T>(path: string, body: unknown): Promise<T> {
    return this.request(path, { method: "PUT", body: JSON.stringify(body) });
  }

  // --- projects ---
  listProjects(): Promise<ProjectList> {
    return this.request("/api/projects");
  }
  currentProject(): Promise<ProjectView | null> {
    return this.request("/api/project");
  }
  createProject(name: string, sampleRate: number): Promise<ProjectView> {
    return this.post("/api/projects", { name, sample_rate: sampleRate });
  }
  openProject(path: string): Promise<ProjectView> {
    return this.post("/api/projects/open", { path });
  }
  closeProject(): Promise<void> {
    return this.post("/api/project/close");
  }
  saveProject(): Promise<ProjectView> {
    return this.post("/api/project/save");
  }
  verifyProject(): Promise<Job> {
    return this.post("/api/project/verify");
  }
  rebuildSource(id: string): Promise<Job> {
    return this.post(`/api/project/sources/${encodeURIComponent(id)}/rebuild`);
  }

  // --- analysis ---
  analysis(sourceId: string): Promise<AnalysisView> {
    return this.request(`/api/project/sources/${encodeURIComponent(sourceId)}/analysis`);
  }
  runAnalysis(sourceId: string, force = false): Promise<AnalysisView> {
    return this.post(`/api/project/sources/${encodeURIComponent(sourceId)}/analysis`, { force });
  }
  putCorrections(sourceId: string, body: AnalysisCorrections): Promise<AnalysisView> {
    return this.put(
      `/api/project/sources/${encodeURIComponent(sourceId)}/analysis/corrections`,
      body,
    );
  }
  resetCorrections(sourceId: string): Promise<AnalysisView> {
    return this.del(`/api/project/sources/${encodeURIComponent(sourceId)}/analysis/corrections`);
  }

  // --- editing ---
  addTrack(name?: string): Promise<ProjectView> {
    return this.post("/api/project/tracks", { name: name ?? null });
  }
  patchTrack(
    id: string,
    body: { name?: string; muted?: boolean; solo?: boolean; gain?: number },
  ): Promise<ProjectView> {
    return this.patch(`/api/project/tracks/${encodeURIComponent(id)}`, body);
  }
  deleteTrack(id: string): Promise<ProjectView> {
    return this.del(`/api/project/tracks/${encodeURIComponent(id)}`);
  }
  trimClip(id: string, start: number, end: number): Promise<ProjectView> {
    return this.post(`/api/project/clips/${encodeURIComponent(id)}/trim`, { start, end });
  }
  resetTrim(id: string): Promise<ProjectView> {
    return this.post(`/api/project/clips/${encodeURIComponent(id)}/reset-trim`);
  }
  deleteClip(id: string): Promise<ProjectView> {
    return this.del(`/api/project/clips/${encodeURIComponent(id)}`);
  }
  sliceClip(id: string, start: number, end: number, name?: string): Promise<ProjectView> {
    return this.post(`/api/project/clips/${encodeURIComponent(id)}/slice`, {
      start,
      end,
      name: name ?? null,
    });
  }
  renameSample(id: string, name: string): Promise<ProjectView> {
    return this.patch(`/api/project/samples/${encodeURIComponent(id)}`, { name });
  }
  deleteSample(id: string): Promise<ProjectView> {
    return this.del(`/api/project/samples/${encodeURIComponent(id)}`);
  }
  placeSample(id: string, trackId: string | null, at: number): Promise<ProjectView> {
    return this.post(`/api/project/samples/${encodeURIComponent(id)}/place`, {
      track_id: trackId,
      at,
    });
  }

  // --- audio ---
  async importFile(file: File, channels: ChannelMode): Promise<Job> {
    const query = new URLSearchParams({ filename: file.name, channels });
    const response = await this.raw(
      `/api/project/import?${query.toString()}`,
      { method: "POST", body: file },
      "application/octet-stream",
    );
    return (await response.json()) as Job;
  }
  peaks(sourceId: string, start: number, end: number, columns: number): Promise<Peaks> {
    const q = new URLSearchParams({
      start: String(start),
      end: String(end),
      columns: String(columns),
    });
    return this.request(
      `/api/project/sources/${encodeURIComponent(sourceId)}/peaks?${q.toString()}`,
    );
  }
  async renderMix(start: number, end: number): Promise<ArrayBuffer> {
    const q = new URLSearchParams({ start: String(start), end: String(end) });
    return (await this.raw(`/api/project/render?${q.toString()}`)).arrayBuffer();
  }
  async renderSample(id: string): Promise<ArrayBuffer> {
    return (await this.raw(`/api/project/samples/${encodeURIComponent(id)}/render`)).arrayBuffer();
  }
  exportAudio(body: {
    kind: "sample" | "selection";
    sample_id?: string;
    start?: number;
    end?: number;
    name?: string;
    format?: "wav" | "flac";
    subtype?: "PCM_16" | "PCM_24" | "FLOAT";
  }): Promise<Job> {
    return this.post("/api/project/exports", body);
  }
  async downloadExport(filename: string): Promise<void> {
    const blob = await (
      await this.raw(`/api/project/exports/${encodeURIComponent(filename)}`)
    ).blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => {
      URL.revokeObjectURL(url);
    }, 10_000);
  }

  // --- devices & recording ---
  audioDevices(): Promise<AudioDevices> {
    return this.request("/api/audio/devices");
  }
  refreshDevices(): Promise<Job> {
    return this.post("/api/audio/devices/refresh");
  }
  startRecording(device: AudioDevice, maxSeconds: number): Promise<Job> {
    return this.post("/api/audio/record", {
      device_index: device.index,
      device_name: device.name,
      kind: device.kind,
      max_seconds: maxSeconds,
    });
  }
  stopJob(id: string): Promise<Job> {
    return this.post(`/api/jobs/${encodeURIComponent(id)}/stop`);
  }

  system(): Promise<SystemInfo> {
    return this.request("/api/system");
  }

  detect(): Promise<Job> {
    return this.request("/api/backends/detect", { method: "POST" });
  }

  probes(): Promise<ProbeRecord[]> {
    return this.request("/api/probes");
  }

  runProbe(backend: BackendId, force = false): Promise<Job> {
    return this.request("/api/probes", {
      method: "POST",
      body: JSON.stringify({ backend, force }),
    });
  }

  jobs(): Promise<Job[]> {
    return this.request("/api/jobs");
  }

  submitJob(kind: string, params: Record<string, unknown>): Promise<Job> {
    return this.request("/api/jobs", { method: "POST", body: JSON.stringify({ kind, params }) });
  }

  cancelJob(id: string): Promise<Job> {
    return this.request(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
  }
}
