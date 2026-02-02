/**
 * Shared utility functions for NeuralPlayground.
 *
 * Pure math/helper functions used across the neural network core,
 * visualization components, and adversarial lab.
 */

import type { ActivationFn } from './types';

// ─── Activation functions ────────────────────────────────────────────

export function activate(x: number, fn: ActivationFn): number {
  switch (fn) {
    case 'relu': return Math.max(0, x);
    case 'sigmoid': return 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500)));
    case 'tanh': return Math.tanh(x);
  }
}

export function activateDerivative(x: number, fn: ActivationFn): number {
  switch (fn) {
    case 'relu': return x > 0 ? 1 : 0;
    case 'sigmoid': {
      const s = activate(x, 'sigmoid');
      return s * (1 - s);
    }
    case 'tanh': {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
  }
}

// ─── Array helpers ───────────────────────────────────────────────────

/** Stack-safe max — avoids Math.max(...arr) RangeError on large arrays */
export function safeMax(arr: number[]): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > m) m = arr[i];
  }
  return m;
}

/** Stack-safe argmax */
export function argmax(arr: number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
  }
  return maxIdx;
}

// ─── Softmax ─────────────────────────────────────────────────────────

export function softmax(arr: number[]): number[] {
  const max = safeMax(arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0 || !isFinite(sum)) {
    // Uniform fallback on degenerate input
    return arr.map(() => 1 / arr.length);
  }
  return exps.map(x => x / sum);
}

// ─── Weight initialization ──────────────────────────────────────────

export function xavierInit(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  const u1 = Math.random();
  const u2 = Math.random();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── PRNG ────────────────────────────────────────────────────────────

/** Seeded PRNG (Mulberry32) for reproducible noise patterns */
export function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform for gaussian noise */
export function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// ─── Canvas color helpers ────────────────────────────────────────────

/** Get node color based on activation value */
export function getActivationColor(value: number, alpha = 1): string {
  if (value > 0) {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(99, 222, 255, ${intensity * alpha})`;
  } else {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(255, 99, 132, ${intensity * alpha})`;
  }
}

/** Get connection color based on weight value */
export function getWeightColor(value: number): string {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped > 0) {
    return `rgba(99, 222, 255, ${Math.abs(clamped) * 0.6})`;
  } else {
    return `rgba(255, 99, 132, ${Math.abs(clamped) * 0.6})`;
  }
}
