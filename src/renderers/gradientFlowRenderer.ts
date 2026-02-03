/**
 * Gradient Flow Monitor Canvas Renderer.
 *
 * Draws per-layer gradient magnitude bars with health indicators,
 * plus a mini sparkline history showing gradient evolution over time.
 */

import type { GradientFlowSnapshot } from '../nn/gradientFlow';

const BAR_HEIGHT = 22;
const BAR_GAP = 6;
const LABEL_WIDTH = 52;
const VALUE_WIDTH = 56;
const PADDING = 8;
const SPARKLINE_HEIGHT = 40;
const SPARKLINE_GAP = 8;
const FONT = '10px Inter, sans-serif';
const FONT_BOLD = '600 11px Inter, sans-serif';

export const GRADIENT_FLOW_WIDTH = 340;

export function getGradientFlowHeight(numLayers: number): number {
  return PADDING * 2 + numLayers * (BAR_HEIGHT + BAR_GAP) + SPARKLINE_GAP + SPARKLINE_HEIGHT + 14;
}

function gradColor(meanGrad: number): string {
  // Green (healthy) → Yellow (warning) → Red (exploding/vanishing)
  if (meanGrad < 1e-6) return '#6b7280'; // dead gray
  if (meanGrad < 0.001) return '#3b82f6'; // vanishing blue
  if (meanGrad < 0.01) return '#06b6d4'; // low cyan
  if (meanGrad < 0.1) return '#10b981'; // healthy green
  if (meanGrad < 1.0) return '#fbbf24'; // warning yellow
  if (meanGrad < 10) return '#f97316'; // hot orange
  return '#ef4444'; // exploding red
}

function healthEmoji(health: GradientFlowSnapshot['health']): string {
  switch (health) {
    case 'healthy': return '✅';
    case 'vanishing': return '❄️';
    case 'exploding': return '🔥';
  }
}

function healthLabel(health: GradientFlowSnapshot['health']): string {
  switch (health) {
    case 'healthy': return 'Healthy';
    case 'vanishing': return 'Vanishing';
    case 'exploding': return 'Exploding';
  }
}

function healthColor(health: GradientFlowSnapshot['health']): string {
  switch (health) {
    case 'healthy': return '#10b981';
    case 'vanishing': return '#3b82f6';
    case 'exploding': return '#ef4444';
  }
}

export function drawGradientFlow(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  snapshot: GradientFlowSnapshot | null,
  history: GradientFlowSnapshot[],
): void {
  ctx.clearRect(0, 0, width, height);

  if (!snapshot || snapshot.layers.length === 0) {
    ctx.font = FONT;
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Train to see gradient flow…', width / 2, height / 2);
    return;
  }

  const layers = snapshot.layers;
  const barWidth = width - LABEL_WIDTH - VALUE_WIDTH - PADDING * 2;

  // Find scale: log10 of max gradient across all layers
  let globalMax = 0;
  for (const l of layers) {
    if (l.meanAbsGrad > globalMax) globalMax = l.meanAbsGrad;
  }
  // Use log scale for better visibility
  const logMax = globalMax > 0 ? Math.log10(globalMax) + 2 : 1; // shift so small values visible

  // Draw bars
  let y = PADDING;
  for (let i = 0; i < layers.length; i++) {
    const l = layers[i];
    const isOutput = i === layers.length - 1;
    const label = isOutput ? 'Output' : `Hidden ${i + 1}`;

    // Label
    ctx.font = FONT;
    ctx.fillStyle = '#9ca3af';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, LABEL_WIDTH - 4, y + BAR_HEIGHT / 2);

    // Bar background
    const bx = LABEL_WIDTH;
    ctx.fillStyle = 'rgba(42, 48, 66, 0.5)';
    ctx.beginPath();
    ctx.roundRect(bx, y, barWidth, BAR_HEIGHT, 4);
    ctx.fill();

    // Bar fill
    const logVal = l.meanAbsGrad > 0 ? Math.log10(l.meanAbsGrad) + 2 : 0;
    const fillFrac = Math.max(0, Math.min(1, logVal / logMax));
    const fillW = fillFrac * barWidth;

    if (fillW > 1) {
      const color = gradColor(l.meanAbsGrad);
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, y, fillW, BAR_HEIGHT, 4);
      ctx.fill();

      // Glow
      ctx.shadowColor = color;
      ctx.shadowBlur = 6;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(bx, y, fillW, BAR_HEIGHT, 4);
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Dead neuron indicator
    if (l.deadFraction > 0.5) {
      const deadW = l.deadFraction * barWidth;
      ctx.fillStyle = 'rgba(107, 114, 128, 0.3)';
      ctx.beginPath();
      ctx.roundRect(bx + barWidth - deadW, y, deadW, BAR_HEIGHT, 4);
      ctx.fill();
    }

    // Value text
    ctx.font = FONT;
    ctx.fillStyle = gradColor(l.meanAbsGrad);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    const valStr = l.meanAbsGrad >= 0.01
      ? l.meanAbsGrad.toFixed(3)
      : l.meanAbsGrad.toExponential(1);
    ctx.fillText(valStr, LABEL_WIDTH + barWidth + 4, y + BAR_HEIGHT / 2);

    y += BAR_HEIGHT + BAR_GAP;
  }

  // Health badge
  ctx.font = FONT_BOLD;
  ctx.fillStyle = healthColor(snapshot.health);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'top';
  ctx.fillText(
    `${healthEmoji(snapshot.health)} ${healthLabel(snapshot.health)}`,
    width - PADDING,
    y + 2,
  );

  // Sparkline of gradient history (mean across all layers over time)
  const sparkY = y + SPARKLINE_GAP + 14;
  const sparkW = width - PADDING * 2;

  if (history.length > 1) {
    // Compute per-epoch mean gradient
    const values: number[] = [];
    for (const snap of history) {
      let sum = 0;
      for (const l of snap.layers) sum += l.meanAbsGrad;
      values.push(sum / snap.layers.length);
    }

    // Log-scale sparkline (in-place to avoid allocation)
    const logVals = values;
    for (let i = 0; i < values.length; i++) {
      logVals[i] = values[i] > 0 ? Math.log10(values[i]) : -10;
    }
    let minLog = logVals[0], maxLog = logVals[0];
    for (const v of logVals) {
      if (v < minLog) minLog = v;
      if (v > maxLog) maxLog = v;
    }
    const range = maxLog - minLog || 1;

    // Label
    ctx.font = FONT;
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillText('Gradient history (log scale)', PADDING, sparkY - 12);

    // Background
    ctx.fillStyle = 'rgba(42, 48, 66, 0.3)';
    ctx.beginPath();
    ctx.roundRect(PADDING, sparkY, sparkW, SPARKLINE_HEIGHT, 4);
    ctx.fill();

    // Line
    ctx.beginPath();
    for (let i = 0; i < logVals.length; i++) {
      const x = PADDING + (i / (logVals.length - 1)) * sparkW;
      const sy = sparkY + SPARKLINE_HEIGHT - ((logVals[i] - minLog) / range) * (SPARKLINE_HEIGHT - 4) - 2;
      if (i === 0) ctx.moveTo(x, sy);
      else ctx.lineTo(x, sy);
    }
    ctx.strokeStyle = healthColor(snapshot.health);
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Fill area
    const lastX = PADDING + sparkW;
    ctx.lineTo(lastX, sparkY + SPARKLINE_HEIGHT);
    ctx.lineTo(PADDING, sparkY + SPARKLINE_HEIGHT);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, sparkY, 0, sparkY + SPARKLINE_HEIGHT);
    const hc = healthColor(snapshot.health);
    grad.addColorStop(0, hc.replace(')', ', 0.2)').replace('rgb', 'rgba'));
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.fill();
  }
}
