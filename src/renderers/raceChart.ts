/**
 * Race chart rendering — pure canvas drawing function.
 *
 * Extracted from TrainingRace.tsx for independent testing and cleaner separation
 * of rendering logic from React component state.
 */

import { RACE_EPOCHS } from '../constants';

export interface RaceChartData {
  accA: number[];
  accB: number[];
  epoch: number;
  winner: 'A' | 'B' | 'tie' | null;
}

export interface RacerVisuals {
  name: string;
  color: string;
}

const CHART_WIDTH = 420;
const PAD = { top: 20, right: 50, bottom: 24, left: 40 } as const;

/**
 * Draw the training race accuracy chart.
 *
 * Pure rendering function — no React dependency, no side effects beyond
 * drawing to the provided canvas context.
 */
export function drawRaceChart(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  data: RaceChartData,
  racerA: RacerVisuals,
  racerB: RacerVisuals,
): void {
  ctx.clearRect(0, 0, width, height);

  const plotW = width - PAD.left - PAD.right;
  const plotH = height - PAD.top - PAD.bottom;

  // Grid lines
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.top + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(PAD.left, y);
    ctx.lineTo(PAD.left + plotW, y);
    ctx.stroke();
  }

  const { accA, accB, epoch, winner } = data;

  if (accA.length === 0 && accB.length === 0) {
    ctx.fillStyle = '#6b7280';
    ctx.font = '12px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Configure networks and click Race!', width / 2, height / 2);
    return;
  }

  const maxEpochs = Math.max(accA.length, accB.length, RACE_EPOCHS);

  drawAccuracyLine(ctx, accA, racerA.color, maxEpochs, plotW, plotH);
  drawAccuracyLine(ctx, accB, racerB.color, maxEpochs, plotW, plotH);

  // Y-axis labels
  ctx.fillStyle = '#9ca3af';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {
    const val = (1 / 4) * (4 - i);
    const y = PAD.top + (plotH / 4) * i;
    ctx.fillText(`${(val * 100).toFixed(0)}%`, PAD.left - 4, y + 3);
  }

  // Legend
  ctx.font = 'bold 10px Inter, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillStyle = racerA.color;
  ctx.fillText(
    `A: ${accA.length > 0 ? (accA[accA.length - 1] * 100).toFixed(1) : 0}%`,
    PAD.left + 4,
    PAD.top - 6,
  );
  ctx.fillStyle = racerB.color;
  ctx.fillText(
    `B: ${accB.length > 0 ? (accB[accB.length - 1] * 100).toFixed(1) : 0}%`,
    PAD.left + plotW / 2,
    PAD.top - 6,
  );

  // Epoch label
  ctx.fillStyle = '#6b7280';
  ctx.font = '9px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`Epoch ${epoch}/${RACE_EPOCHS}`, width / 2, height - 4);

  // Winner
  if (winner) {
    ctx.fillStyle = winner === 'A' ? racerA.color : winner === 'B' ? racerB.color : '#9ca3af';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.textAlign = 'right';
    const label = winner === 'tie' ? '🤝 Tie!' :
      `🏆 ${winner === 'A' ? racerA.name : racerB.name} wins!`;
    ctx.fillText(label, PAD.left + plotW, PAD.top - 6);
  }
}

/** Draw a single accuracy line + filled area */
function drawAccuracyLine(
  ctx: CanvasRenderingContext2D,
  data: number[],
  color: string,
  maxEpochs: number,
  plotW: number,
  plotH: number,
): void {
  if (data.length < 2) return;

  const xScale = 1 / Math.max(maxEpochs - 1, 1);

  // Stroke
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = PAD.left + (i * xScale) * plotW;
    const y = PAD.top + plotH - (Math.min(data[i], 1) / 1) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fill area
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = PAD.left + (i * xScale) * plotW;
    const y = PAD.top + plotH - (Math.min(data[i], 1) / 1) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  const lastX = PAD.left + ((data.length - 1) * xScale) * plotW;
  ctx.lineTo(lastX, PAD.top + plotH);
  ctx.lineTo(PAD.left, PAD.top + plotH);
  ctx.closePath();
  ctx.globalAlpha = 0.06;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.globalAlpha = 1;
}

export { CHART_WIDTH };
