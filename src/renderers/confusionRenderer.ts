/**
 * Confusion Matrix Canvas Renderer.
 *
 * Draws a 10×10 heatmap with cell counts, axis labels,
 * hover highlight, and precision/recall sidebar.
 */

import type { ConfusionData } from '../nn/confusion';

const CELL_SIZE = 36;
const LABEL_SIZE = 24;
const PADDING = 4;
const FONT = '11px Inter, sans-serif';
const FONT_BOLD = '600 12px Inter, sans-serif';
const FONT_SMALL = '9px Inter, sans-serif';

/** Total canvas width/height */
export const CONFUSION_CANVAS_SIZE = LABEL_SIZE + CELL_SIZE * 10 + PADDING * 2;

function heatColor(value: number, maxVal: number, isDiagonal: boolean): string {
  if (maxVal === 0) return 'rgba(42, 48, 66, 0.6)';
  const t = Math.min(value / Math.max(maxVal, 1), 1);
  if (isDiagonal) {
    // Green for correct: dim → bright
    const r = Math.round(10 + (16 - 10) * (1 - t));
    const g = Math.round(30 + (185 - 30) * t);
    const b = Math.round(40 + (129 - 40) * t);
    const a = 0.3 + t * 0.7;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  } else {
    // Red for errors: dim → bright
    const r = Math.round(60 + (255 - 60) * t);
    const g = Math.round(30 + (99 - 30) * (1 - t));
    const b = Math.round(40 + (132 - 40) * (1 - t));
    const a = 0.2 + t * 0.8;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }
}

export function drawConfusionMatrix(
  ctx: CanvasRenderingContext2D,
  data: ConfusionData,
  hoverCell: { row: number; col: number } | null,
): void {
  const size = CONFUSION_CANVAS_SIZE;
  ctx.clearRect(0, 0, size, size);

  // Find max value for color scaling
  let maxVal = 0;
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      if (data.matrix[r][c] > maxVal) maxVal = data.matrix[r][c];
    }
  }

  const ox = LABEL_SIZE; // x offset
  const oy = LABEL_SIZE; // y offset

  // Draw axis labels
  ctx.font = FONT_BOLD;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  for (let i = 0; i < 10; i++) {
    const x = ox + i * CELL_SIZE + CELL_SIZE / 2;
    const y = oy + i * CELL_SIZE + CELL_SIZE / 2;

    // Top labels (predicted)
    const isHoverCol = hoverCell && hoverCell.col === i;
    ctx.fillStyle = isHoverCol ? '#63deff' : '#9ca3af';
    ctx.fillText(String(i), x, oy / 2);

    // Left labels (actual)
    const isHoverRow = hoverCell && hoverCell.row === i;
    ctx.fillStyle = isHoverRow ? '#63deff' : '#9ca3af';
    ctx.fillText(String(i), ox / 2, y);
  }

  // Axis titles
  ctx.font = FONT_SMALL;
  ctx.fillStyle = '#6b7280';
  ctx.fillText('Predicted →', ox + CELL_SIZE * 5, 6);
  ctx.save();
  ctx.translate(6, oy + CELL_SIZE * 5);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Actual →', 0, 0);
  ctx.restore();

  // Draw cells
  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const x = ox + c * CELL_SIZE;
      const y = oy + r * CELL_SIZE;
      const val = data.matrix[r][c];
      const isDiag = r === c;

      // Cell background
      ctx.fillStyle = heatColor(val, maxVal, isDiag);
      ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);

      // Hover highlight
      if (hoverCell && hoverCell.row === r && hoverCell.col === c) {
        ctx.strokeStyle = '#63deff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
      }

      // Row/column highlight on hover
      if (hoverCell && (hoverCell.row === r || hoverCell.col === c) &&
          !(hoverCell.row === r && hoverCell.col === c)) {
        ctx.fillStyle = 'rgba(99, 222, 255, 0.05)';
        ctx.fillRect(x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2);
      }

      // Count text
      if (val > 0) {
        ctx.font = isDiag ? FONT_BOLD : FONT;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const brightness = val / maxVal;
        ctx.fillStyle = brightness > 0.5 ? '#ffffff' : '#9ca3af';
        ctx.fillText(String(val), x + CELL_SIZE / 2, y + CELL_SIZE / 2);
      }
    }
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(42, 48, 66, 0.5)';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 10; i++) {
    ctx.beginPath();
    ctx.moveTo(ox + i * CELL_SIZE, oy);
    ctx.lineTo(ox + i * CELL_SIZE, oy + 10 * CELL_SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ox, oy + i * CELL_SIZE);
    ctx.lineTo(ox + 10 * CELL_SIZE, oy + i * CELL_SIZE);
    ctx.stroke();
  }
}

/**
 * Convert canvas coordinates to matrix cell.
 */
export function hitTestConfusion(
  x: number,
  y: number,
): { row: number; col: number } | null {
  const ox = LABEL_SIZE;
  const oy = LABEL_SIZE;
  const col = Math.floor((x - ox) / CELL_SIZE);
  const row = Math.floor((y - oy) / CELL_SIZE);
  if (row >= 0 && row < 10 && col >= 0 && col < 10) {
    return { row, col };
  }
  return null;
}
