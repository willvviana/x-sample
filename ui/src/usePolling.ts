import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "./api/client";

export interface Polled<T> {
  data: T | undefined;
  error: ApiError | undefined;
  refresh: () => void;
}

/** Poll `load` every `intervalMs`; the interval may change between renders. */
export function usePolling<T>(load: () => Promise<T>, intervalMs: number): Polled<T> {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<ApiError>();
  const loadRef = useRef(load);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    loadRef.current = load;
  }, [load]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const run = async () => {
      try {
        const value = await loadRef.current();
        if (!cancelled) {
          setData(value);
          setError(undefined);
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
      }
      if (!cancelled) timer = window.setTimeout(() => void run(), intervalMs);
    };
    void run();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [intervalMs, tick]);

  const refresh = useCallback(() => {
    setTick((t) => t + 1);
  }, []);
  return { data, error, refresh };
}
