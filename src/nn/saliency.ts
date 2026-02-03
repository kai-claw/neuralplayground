/**
 * Saliency Map — compute gradient-based attention heatmaps.
 *
 * Given a trained network and an input image, computes which pixels
 * matter most for the predicted class via input gradient magnitude.
 * This is the classic "what is the network looking at?" visualization.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import { computeInputGradient } from './dreams';

// Pre-allocated saliency buffer (avoids Float32Array allocation per call)
let _saliencyBuffer: Float32Array | null = null;

/**
 * Compute saliency map: absolute gradient magnitude normalized to [0, 1].
 * Returns a Float32Array of 784 values indicating pixel importance.
 *
 * NOTE: Returns a shared buffer. Callers must consume before the next call.
 */
export function computeSaliency(
  network: NeuralNetwork,
  input: number[],
  targetClass: number,
): Float32Array {
  const gradient = computeInputGradient(network, input, targetClass);
  if (!_saliencyBuffer || _saliencyBuffer.length !== gradient.length) {
    _saliencyBuffer = new Float32Array(gradient.length);
  }
  const saliency = _saliencyBuffer;

  // Absolute gradient magnitude
  let maxVal = 0;
  for (let i = 0; i < gradient.length; i++) {
    saliency[i] = Math.abs(gradient[i]);
    if (saliency[i] > maxVal) maxVal = saliency[i];
  }

  // Normalize to [0, 1]
  if (maxVal > 0) {
    const inv = 1 / maxVal;
    for (let i = 0; i < saliency.length; i++) {
      saliency[i] *= inv;
    }
  }

  return saliency;
}

/** Inferno-like colormap: black → purple → red → yellow → white */
const INFERNO_STOPS: [number, number, number, number][] = [
  [0.0, 0, 0, 4],
  [0.13, 27, 12, 65],
  [0.25, 72, 12, 104],
  [0.38, 120, 28, 109],
  [0.5, 165, 44, 96],
  [0.63, 207, 68, 70],
  [0.75, 237, 105, 37],
  [0.88, 251, 155, 6],
  [1.0, 252, 255, 164],
];

// ─── 256-entry pre-computed saliency color LUT (eliminates per-pixel allocation) ───
const SALIENCY_LUT_R = new Uint8Array(256);
const SALIENCY_LUT_G = new Uint8Array(256);
const SALIENCY_LUT_B = new Uint8Array(256);
const SALIENCY_LUT_A = new Uint8Array(256);

(function buildSaliencyLUT() {
  for (let i = 0; i < 256; i++) {
    const value = i / 255;
    if (value <= 0) {
      SALIENCY_LUT_R[i] = 0;
      SALIENCY_LUT_G[i] = 0;
      SALIENCY_LUT_B[i] = 0;
      SALIENCY_LUT_A[i] = 0;
      continue;
    }
    let lo = INFERNO_STOPS[0];
    let hi = INFERNO_STOPS[INFERNO_STOPS.length - 1];
    for (let s = 0; s < INFERNO_STOPS.length - 1; s++) {
      if (value >= INFERNO_STOPS[s][0] && value <= INFERNO_STOPS[s + 1][0]) {
        lo = INFERNO_STOPS[s];
        hi = INFERNO_STOPS[s + 1];
        break;
      }
    }
    const range = hi[0] - lo[0];
    const t = range > 0 ? (value - lo[0]) / range : 0;
    SALIENCY_LUT_R[i] = Math.round(lo[1] + (hi[1] - lo[1]) * t);
    SALIENCY_LUT_G[i] = Math.round(lo[2] + (hi[2] - lo[2]) * t);
    SALIENCY_LUT_B[i] = Math.round(lo[3] + (hi[3] - lo[3]) * t);
    SALIENCY_LUT_A[i] = Math.round(Math.min(255, value * 1.3 * 255));
  }
})();

/** Map a 0-1 saliency value to an RGBA color (inferno-ish).
 *  Returns a tuple — use saliencyLUT* for hot-path pixel rendering instead. */
export function saliencyToColor(value: number): [number, number, number, number] {
  const idx = Math.max(0, Math.min(255, Math.round(value * 255)));
  return [SALIENCY_LUT_R[idx], SALIENCY_LUT_G[idx], SALIENCY_LUT_B[idx], SALIENCY_LUT_A[idx]];
}

// Cached ImageData for saliency overlay rendering
const _saliencyCache = new Map<number, ImageData>();

/**
 * Render saliency map as an ImageData overlay.
 * Composites: dim original digit + bright saliency heatmap.
 *
 * NOTE: Returns a cached ImageData. Callers must consume before next call.
 */
export function renderSaliencyOverlay(
  input: number[],
  saliency: Float32Array,
  size: number,
): ImageData {
  const dim = 28;
  let imageData = _saliencyCache.get(size);
  if (!imageData) {
    imageData = new ImageData(size, size);
    _saliencyCache.set(size, imageData);
  }
  const data = imageData.data;
  const scale = size / dim;

  for (let py = 0; py < size; py++) {
    const sy = Math.min(dim - 1, (py / scale) | 0);
    const rowOff = sy * dim;
    for (let px = 0; px < size; px++) {
      const sx = Math.min(dim - 1, (px / scale) | 0);
      const idx = rowOff + sx;

      const pixelVal = input[idx] || 0;
      const salVal = saliency[idx] || 0;

      // Base: dim grayscale of original digit
      const gray = (pixelVal * 80 + 0.5) | 0;

      // LUT-based saliency color (zero allocation)
      const lutIdx = Math.max(0, Math.min(255, (salVal * 255 + 0.5) | 0));
      const sr = SALIENCY_LUT_R[lutIdx];
      const sg = SALIENCY_LUT_G[lutIdx];
      const sb = SALIENCY_LUT_B[lutIdx];
      const alpha = SALIENCY_LUT_A[lutIdx] / 255;

      // Composite
      const oneMinusA = 1 - alpha;
      const r = (gray * oneMinusA + sr * alpha + 0.5) | 0;
      const g = (gray * (1 - alpha * 0.9) + sg * alpha + 0.5) | 0;
      const b = (gray * (1 - alpha * 0.8) + sb * alpha + 0.5) | 0;

      const offset = (py * size + px) << 2;
      data[offset] = r < 255 ? r : 255;
      data[offset + 1] = g < 255 ? g : 255;
      data[offset + 2] = b < 255 ? b : 255;
      data[offset + 3] = 255;
    }
  }

  return imageData;
}
