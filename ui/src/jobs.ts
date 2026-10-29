import type { Job } from "./api/client";

const ACTIVE = new Set<Job["state"]>(["queued", "running", "cancelling"]);

export function isActive(job: Job): boolean {
  return ACTIVE.has(job.state);
}
