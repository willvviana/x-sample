/** Positions are integer samples; seconds are only for display. */
export function formatPosition(samples: number, rate: number): string {
  const total = samples / rate;
  const minutes = Math.floor(total / 60);
  const seconds = total - minutes * 60;
  return `${String(minutes)}:${seconds.toFixed(3).padStart(6, "0")}`;
}

export function formatDuration(samples: number, rate: number): string {
  return `${(samples / rate).toFixed(2)} s`;
}

export function formatDb(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "−∞";
  return `${value.toFixed(1)} dB`;
}

export function clamp(value: number, low: number, high: number): number {
  return Math.min(high, Math.max(low, value));
}
