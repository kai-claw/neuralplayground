/**
 * Shared rendering helpers — pure image-data generation.
 *
 * Extracted from FeatureMaps.tsx and DigitMorph.tsx.
 * These functions produce ImageData or pixel arrays without touching the DOM.
 */

import { INPUT_DIM } from './constants';

// ─── Diverging colormap (cyan ← dark → red) ─────────────────────────

/**
 * Render a weight vector as a diverging-colormap ImageData.
 *
 * Positive weights → cyan/blue.  Negative weights → red/orange.
 * Weights are contrast-normalized to [0, 1] then mapped through the colormap.
 *
 * @param weights  Flat array of weights (typically 784 for first-layer neurons)
 * @param size     Output image size in pixels (square)
 */
export function weightsToImageData(
  weights: number[],
  size: number,
): ImageData {
  const imgData = new ImageData(size, size);
  const data = imgData.data;

  // Contrast normalization
  let wMin = Infinity;
  let wMax = -Infinity;
  const len = Math.min(weights.length, INPUT_DIM * INPUT_DIM);
  for (let i = 0; i < len; i++) {
    if (weights[i] < wMin) wMin = weights[i];
    if (weights[i] > wMax) wMax = weights[i];
  }
  const range = wMax - wMin;

  const scale = size / INPUT_DIM;

  for (let py = 0; py < size; py++) {
    const sy = Math.min(Math.floor(py / scale), INPUT_DIM - 1);
    for (let px = 0; px < size; px++) {
      const sx = Math.min(Math.floor(px / scale), INPUT_DIM - 1);
      const wi = sy * INPUT_DIM + sx;
      // When all weights are identical (range=0), map based on sign:
      // positive → 1.0 (cyan), negative → 0.0 (red), zero → 0.5 (neutral)
      const norm = wi >= len ? 0.5
        : range === 0 ? (wMax > 0 ? 1.0 : wMax < 0 ? 0.0 : 0.5)
        : (weights[wi] - wMin) / range;

      const idx = (py * size + px) * 4;
      if (norm >= 0.5) {
        const t = (norm - 0.5) * 2;
        data[idx]     = Math.round(20 + t * 79);    // R
        data[idx + 1] = Math.round(40 + t * 182);   // G
        data[idx + 2] = Math.round(60 + t * 195);   // B
      } else {
        const t = (0.5 - norm) * 2;
        data[idx]     = Math.round(20 + t * 235);   // R
        data[idx + 1] = Math.round(40 + t * 59);    // G
        data[idx + 2] = Math.round(60 + t * 72);    // B
      }
      data[idx + 3] = 255;
    }
  }

  return imgData;
}

// ─── Grayscale pixel rendering ───────────────────────────────────────

/**
 * Render a 784-element [0, 1] pixel array as a grayscale ImageData.
 *
 * Used by DigitMorph and AdversarialLab to preview processed images.
 *
 * @param pixels  28×28 flat array of pixel values [0, 1]
 * @param size    Output display size in pixels (square, will be upscaled)
 */
export function pixelsToImageData(
  pixels: number[],
  size: number,
): ImageData {
  const imgData = new ImageData(size, size);
  const data = imgData.data;
  const scale = size / INPUT_DIM;

  for (let py = 0; py < size; py++) {
    const sy = Math.min(Math.floor(py / scale), INPUT_DIM - 1);
    for (let px = 0; px < size; px++) {
      const sx = Math.min(Math.floor(px / scale), INPUT_DIM - 1);
      const v = Math.round((pixels[sy * INPUT_DIM + sx] || 0) * 255);
      const idx = (py * size + px) * 4;
      data[idx] = v;
      data[idx + 1] = v;
      data[idx + 2] = v;
      data[idx + 3] = 255;
    }
  }

  return imgData;
}

// ─── Pixel interpolation ─────────────────────────────────────────────

/**
 * Linearly interpolate between two pixel arrays.
 *
 * @param a  First pixel array (784 elements, [0, 1])
 * @param b  Second pixel array (784 elements, [0, 1])
 * @param t  Blend factor [0, 1]: 0 = a, 1 = b
 */
export function lerpPixels(a: number[], b: number[], t: number): number[] {
  const len = Math.min(a.length, b.length);
  const result = new Array<number>(len);
  const oneMinusT = 1 - t;
  for (let i = 0; i < len; i++) {
    result[i] = a[i] * oneMinusT + b[i] * t;
  }
  return result;
}
