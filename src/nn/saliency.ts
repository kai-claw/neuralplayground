/**
 * Saliency Map — compute gradient-based attention heatmaps.
 *
 * Given a trained network and an input image, computes which pixels
 * matter most for the predicted class via input gradient magnitude.
 * This is the classic "what is the network looking at?" visualization.
 */

import type { NeuralNetwork } from './NeuralNetwork';
import { computeInputGradient } from './dreams';

/**
 * Compute saliency map: absolute gradient magnitude normalized to [0, 1].
 * Returns a Float32Array of 784 values indicating pixel importance.
 */
export function computeSaliency(
  network: NeuralNetwork,
  input: number[],
  targetClass: number,
): Float32Array {
  const gradient = computeInputGradient(network, input, targetClass);
  const saliency = new Float32Array(gradient.length);

  // Absolute gradient magnitude
  let maxVal = 0;
  for (let i = 0; i < gradient.length; i++) {
    saliency[i] = Math.abs(gradient[i]);
    if (saliency[i] > maxVal) maxVal = saliency[i];
  }

  // Normalize to [0, 1]
  if (maxVal > 0) {
    for (let i = 0; i < saliency.length; i++) {
      saliency[i] /= maxVal;
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

/** Map a 0-1 saliency value to an RGBA color (inferno-ish). */
export function saliencyToColor(value: number): [number, number, number, number] {
  if (value <= 0) return [0, 0, 0, 0];

  // Find surrounding stops
  let lo = INFERNO_STOPS[0];
  let hi = INFERNO_STOPS[INFERNO_STOPS.length - 1];
  for (let i = 0; i < INFERNO_STOPS.length - 1; i++) {
    if (value >= INFERNO_STOPS[i][0] && value <= INFERNO_STOPS[i + 1][0]) {
      lo = INFERNO_STOPS[i];
      hi = INFERNO_STOPS[i + 1];
      break;
    }
  }

  const range = hi[0] - lo[0];
  const t = range > 0 ? (value - lo[0]) / range : 0;

  const r = Math.round(lo[1] + (hi[1] - lo[1]) * t);
  const g = Math.round(lo[2] + (hi[2] - lo[2]) * t);
  const b = Math.round(lo[3] + (hi[3] - lo[3]) * t);
  // Alpha ramps from 0 at low values to full at high
  const a = Math.round(Math.min(255, value * 1.3 * 255));

  return [r, g, b, a];
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
    for (let px = 0; px < size; px++) {
      const sx = Math.min(dim - 1, Math.floor(px / scale));
      const sy = Math.min(dim - 1, Math.floor(py / scale));
      const idx = sy * dim + sx;

      const pixelVal = input[idx] || 0;
      const salVal = saliency[idx] || 0;

      // Base: dim grayscale of original digit
      const gray = Math.round(pixelVal * 80);

      // Overlay: saliency heatmap
      const [sr, sg, sb, sa] = saliencyToColor(salVal);
      const alpha = sa / 255;

      // Composite
      const r = Math.round(gray * (1 - alpha) + sr * alpha);
      const g = Math.round(gray * (1 - alpha * 0.9) + sg * alpha);
      const b = Math.round(gray * (1 - alpha * 0.8) + sb * alpha);

      const offset = (py * size + px) * 4;
      data[offset] = Math.min(255, r);
      data[offset + 1] = Math.min(255, g);
      data[offset + 2] = Math.min(255, b);
      data[offset + 3] = 255;
    }
  }

  return imageData;
}
