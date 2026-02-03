/**
 * usePerformanceMonitor — Tracks FPS and auto-degrades quality.
 *
 * Uses requestAnimationFrame to count frames per second.
 * When FPS sustains below PERF_DEGRADE_FPS for PERF_DEGRADE_SECONDS,
 * sets degraded=true. Auto-recovers when FPS sustains above
 * PERF_RECOVER_FPS for PERF_RECOVER_SECONDS.
 */

import { useRef, useState, useEffect } from 'react';
import {
  PERF_SAMPLE_INTERVAL,
  PERF_DEGRADE_FPS,
  PERF_RECOVER_FPS,
  PERF_DEGRADE_SECONDS,
  PERF_RECOVER_SECONDS,
} from '../constants';

export interface PerformanceState {
  fps: number;
  degraded: boolean;
}

export function usePerformanceMonitor(): PerformanceState {
  const [state, setState] = useState<PerformanceState>({ fps: 60, degraded: false });
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(performance.now());
  const rafRef = useRef(0);
  const lowCountRef = useRef(0);
  const highCountRef = useRef(0);
  const degradedRef = useRef(false);

  useEffect(() => {
    let running = true;

    const tick = () => {
      if (!running) return;
      frameCountRef.current++;
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    const sampler = setInterval(() => {
      const now = performance.now();
      const elapsed = (now - lastTimeRef.current) / 1000;
      const fps = elapsed > 0 ? Math.round(frameCountRef.current / elapsed) : 60;
      frameCountRef.current = 0;
      lastTimeRef.current = now;

      if (fps < PERF_DEGRADE_FPS) {
        lowCountRef.current++;
        highCountRef.current = 0;
      } else if (fps > PERF_RECOVER_FPS) {
        highCountRef.current++;
        lowCountRef.current = 0;
      } else {
        lowCountRef.current = 0;
        highCountRef.current = 0;
      }

      let degraded = degradedRef.current;
      if (!degraded && lowCountRef.current >= PERF_DEGRADE_SECONDS) {
        degraded = true;
        degradedRef.current = true;
      } else if (degraded && highCountRef.current >= PERF_RECOVER_SECONDS) {
        degraded = false;
        degradedRef.current = false;
      }

      setState({ fps, degraded });
    }, PERF_SAMPLE_INTERVAL);

    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      clearInterval(sampler);
    };
  }, []);

  return state;
}
