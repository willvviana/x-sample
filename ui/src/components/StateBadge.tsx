const TONES: Record<string, string> = {
  available: "ok",
  pass: "ok",
  succeeded: "ok",
  unavailable: "bad",
  fail: "bad",
  failed: "bad",
  unsupported: "warn",
  cancelled: "warn",
  cancelling: "warn",
  running: "info",
  queued: "info",
  pending: "info",
  unknown: "info",
};

export function StateBadge({ state, testId }: { state: string; testId?: string }) {
  return (
    <span className={`badge ${TONES[state] ?? "info"}`} data-testid={testId}>
      {state}
    </span>
  );
}
