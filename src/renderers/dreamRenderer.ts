/**
 * Dream image rendering — pure canvas drawing functions.
 *
 * Extracted from NetworkDreams.tsx for independent testing and reuse.
 * All functions produce pixel data or draw to a provided context
 * without React dependency.
 */

import { INPUT_DIM, DREAM_STEPS } from '../constants';

// ─── Colorize helpers ────────────────────────────────────────────────

/**
 * Convert a grayscale pixel value [0, 1] to a cyan-tinted RGB tuple.
 * Higher brightness → stronger cyan tint (signature dream look).
 */
export function dreamPixelToRGB(value: number): [number, number, number] {
  const v = Math.round(Math.max(0, Math.min(1, value)) * 255);
  return [
    Math.round(v * 0.4),   // R — suppressed for cyan bias
    Math.round(v * 0.87),  // G — near-full
    v,                      // B — full
  ];
}

// ─── Main dream canvas ──────────────────────────────────────────────

/**
 * Render a dream image onto a canvas context with an overlay showing
 * step count and confidence.
 *
 * @param ctx            2D canvas context (already DPR-scaled)
 * @param dreamImage     784-element pixel array [0, 1]
 * @param displaySize    Target display size in CSS pixels
 * @param dreamStep      Current gradient-ascent step
 * @param dreamConfidence Current target-class probability
 */
export function renderDreamImage(
  ctx: CanvasRenderingContext2D,
  dreamImage: number[],
  displaySize: number,
  dreamStep: number,
  dreamConfidence: number,
): void {
  const scale = displaySize / INPUT_DIM;

  for (let y = 0; y < INPUT_DIM; y++) {
    for (let x = 0; x < INPUT_DIM; x++) {
      const [r, g, b] = dreamPixelToRGB(dreamImage[y * INPUT_DIM + x] || 0);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x * scale, y * scale, scale + 0.5, scale + 0.5);
    }
  }

  // Overlay bar
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fillRect(0, displaySize - 22, displaySize, 22);
  ctx.fillStyle = '#63deff';
  ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(
    `Step ${dreamStep}/${DREAM_STEPS} — ${(dreamConfidence * 100).toFixed(1)}%`,
    displaySize / 2,
    displaySize - 7,
  );
}

// ─── Gallery rendering ───────────────────────────────────────────────

const GALLERY_CELL_SIZE = 40;
const GALLERY_GAP = 4;
const GALLERY_COLS = 5;
const GALLERY_ROWS = 2;
const GALLERY_LABEL_HEIGHT = 14;

/** Computed gallery canvas dimensions */
export const GALLERY_DIMS = {
  width: GALLERY_COLS * (GALLERY_CELL_SIZE + GALLERY_GAP) - GALLERY_GAP,
  height: GALLERY_ROWS * (GALLERY_CELL_SIZE + GALLERY_GAP + GALLERY_LABEL_HEIGHT) - GALLERY_GAP,
  cols: GALLERY_COLS,
  rows: GALLERY_ROWS,
  cellSize: GALLERY_CELL_SIZE,
  gap: GALLERY_GAP,
} as const;

/**
 * Render the 10-digit dream gallery grid.
 *
 * @param ctx      2D canvas context (already DPR-scaled)
 * @param gallery  Array of 10 dream images (null = not yet generated)
 */
export function renderDreamGallery(
  ctx: CanvasRenderingContext2D,
  gallery: (number[] | null)[],
): void {
  const { width, height, cellSize, gap, cols } = GALLERY_DIMS;
  ctx.clearRect(0, 0, width, height);

  for (let d = 0; d < 10; d++) {
    const col = d % cols;
    const row = Math.floor(d / cols);
    const x = col * (cellSize + gap);
    const y = row * (cellSize + gap + GALLERY_LABEL_HEIGHT);

    // Digit label
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(String(d), x + cellSize / 2, y + 10);

    const img = gallery[d];
    if (img) {
      const scale = cellSize / INPUT_DIM;
      for (let py = 0; py < INPUT_DIM; py++) {
        for (let px = 0; px < INPUT_DIM; px++) {
          const [r, g, b] = dreamPixelToRGB(img[py * INPUT_DIM + px] || 0);
          ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
          ctx.fillRect(
            x + px * scale,
            y + GALLERY_LABEL_HEIGHT - 2 + py * scale,
            scale + 0.5,
            scale + 0.5,
          );
        }
      }
    } else {
      // Empty placeholder
      ctx.fillStyle = '#1f2937';
      ctx.fillRect(x, y + GALLERY_LABEL_HEIGHT - 2, cellSize, cellSize);
      ctx.fillStyle = '#4b5563';
      ctx.font = '18px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('?', x + cellSize / 2, y + GALLERY_LABEL_HEIGHT - 2 + cellSize / 2 + 6);
    }
  }
}
