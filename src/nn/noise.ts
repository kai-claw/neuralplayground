/**
 * Noise generation for the Adversarial Lab.
 *
 * Pure noise-pattern generators used by the adversarial testing feature.
 * All functions are deterministic given a PRNG seed and operate on Float32Arrays.
 */

import { mulberry32, gaussianNoise } from '../utils';
import type { NoiseType } from '../types';
import { INPUT_DIM } from '../constants';

const PIXEL_COUNT = INPUT_DIM * INPUT_DIM;

// ─── Noise pattern generation ────────────────────────────────────────

/**
 * Generate a reproducible noise pattern for the given type and seed.
 *
 * Returns a new Float32Array of length 784 (28×28).
 * Values represent the raw noise signal before amplitude scaling.
 * Callers (AdversarialLab) store and reference the pattern across renders,
 * so each call must return an owned buffer.
 */
export function generateNoisePattern(
  type: NoiseType,
  seed: number,
  targetDigit: number = 0,
): Float32Array {
  const pattern = new Float32Array(PIXEL_COUNT);
  const rng = mulberry32(seed);

  switch (type) {
    case 'gaussian':
      for (let i = 0; i < PIXEL_COUNT; i++) {
        pattern[i] = gaussianNoise(rng);
      }
      break;

    case 'salt-pepper':
      for (let i = 0; i < PIXEL_COUNT; i++) {
        const r = rng();
        if (r < 0.15) pattern[i] = 1;       // salt (white)
        else if (r < 0.30) pattern[i] = -1;  // pepper (black)
        else pattern[i] = 0;
      }
      break;

    case 'adversarial': {
      const cx = INPUT_DIM / 2;
      const cy = INPUT_DIM / 2;
      const targetAngle = (targetDigit / 10) * Math.PI * 2;

      for (let i = 0; i < PIXEL_COUNT; i++) {
        const x = i % INPUT_DIM;
        const y = Math.floor(i / INPUT_DIM);
        const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) / (INPUT_DIM / 2);
        const angle = Math.atan2(y - cy, x - cx);
        const angleBias = Math.cos(angle - targetAngle);
        pattern[i] = angleBias * (1 - dist) + gaussianNoise(rng) * 0.3;
      }
      break;
    }
  }

  return pattern;
}

// ─── Noise application ───────────────────────────────────────────────

// Pre-allocated noised output buffer
let _noisedBuf: number[] = [];

/**
 * Apply a noise pattern to a clean input at the given noise level.
 *
 * @param input     784-element pixel array [0, 1]
 * @param pattern   noise pattern from generateNoisePattern
 * @param level     noise amplitude [0, 1]
 * @param type      noise type (salt-pepper uses binary flip logic)
 * @param seed      PRNG seed for salt-pepper flip decisions
 * @returns         new 784-element array with noise applied, clamped [0, 1]
 */
export function applyNoise(
  input: number[],
  pattern: Float32Array,
  level: number,
  type: NoiseType,
  seed: number,
): number[] {
  const len = Math.min(input.length, PIXEL_COUNT);
  // Reuse output buffer (avoids Array allocation per call)
  if (_noisedBuf.length !== len) _noisedBuf = new Array<number>(len);
  const noised = _noisedBuf;

  if (type === 'salt-pepper') {
    const rng = mulberry32(seed);
    for (let i = 0; i < len; i++) {
      const flip = Math.abs(pattern[i]) > 0.5;
      if (flip && rng() < level) {
        noised[i] = pattern[i] > 0 ? 1 : 0;
      } else {
        noised[i] = input[i];
      }
    }
  } else {
    for (let i = 0; i < len; i++) {
      noised[i] = Math.max(0, Math.min(1, input[i] + pattern[i] * level));
    }
  }

  return noised;
}
