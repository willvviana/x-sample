import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Plays WAV bytes rendered by the engine through Web Audio. The UI does no mixing: the
 * engine renders the preview (16-bit WAV) and this only decodes and plays it.
 */
export interface Playback {
  playing: boolean;
  /** Seconds since the start of the current clip of audio; null when stopped. */
  elapsed: number | null;
  error: string | undefined;
  play: (bytes: ArrayBuffer) => Promise<void>;
  stop: () => void;
}

export function usePlayback(onEnded?: () => void): Playback {
  const ctxRef = useRef<AudioContext>(undefined);
  const sourceRef = useRef<AudioBufferSourceNode>(undefined);
  const startedAt = useRef(0);
  const raf = useRef(0);
  const [playing, setPlaying] = useState(false);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [error, setError] = useState<string>();
  const endedRef = useRef(onEnded);

  useEffect(() => {
    endedRef.current = onEnded;
  }, [onEnded]);

  const halt = useCallback(() => {
    cancelAnimationFrame(raf.current);
    const source = sourceRef.current;
    sourceRef.current = undefined;
    if (source) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // already stopped
      }
    }
    setPlaying(false);
    setElapsed(null);
  }, []);

  const play = useCallback(
    async (bytes: ArrayBuffer) => {
      halt();
      setError(undefined);
      try {
        const ctx = (ctxRef.current ??= new AudioContext());
        if (ctx.state === "suspended") await ctx.resume();
        const buffer = await ctx.decodeAudioData(bytes);
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        source.onended = () => {
          if (sourceRef.current === source) {
            halt();
            endedRef.current?.();
          }
        };
        sourceRef.current = source;
        startedAt.current = ctx.currentTime;
        source.start();
        setPlaying(true);
        const tick = () => {
          setElapsed(ctx.currentTime - startedAt.current);
          raf.current = requestAnimationFrame(tick);
        };
        tick();
      } catch (err) {
        halt();
        setError(err instanceof Error ? err.message : "Playback failed");
      }
    },
    [halt],
  );

  useEffect(
    () => () => {
      halt();
      void ctxRef.current?.close();
    },
    [halt],
  );

  return { playing, elapsed, error, play, stop: halt };
}
