/**
 * ActivationSpace — 2D PCA projection of hidden-layer activations.
 *
 * Visualizes how the network organizes different digits in its
 * internal representation space. Each dot is a training sample,
 * colored by its digit class. Watch clusters form and separate
 * as the network learns.
 *
 * The user's drawn digit appears as a large highlighted marker
 * showing where it falls relative to the training data.
 */

import { useRef, useEffect, useCallback, useState } from 'react';
import { useContainerDims } from '../hooks/useContainerDims';
import {
  ACTIVATION_SPACE_DEFAULT,
  ACTIVATION_SPACE_ASPECT,
} from '../constants';

/** Digit class colors — 10 distinct, vibrant hues. */
const DIGIT_COLORS = [
  '#ff6384', // 0 — red
  '#36a2eb', // 1 — blue
  '#ffce56', // 2 — yellow
  '#4bc0c0', // 3 — teal
  '#9966ff', // 4 — purple
  '#ff9f40', // 5 — orange
  '#10b981', // 6 — green
  '#f472b6', // 7 — pink
  '#63deff', // 8 — cyan
  '#a78bfa', // 9 — lavender
] as const;

/** Pre-computed projection data passed from parent. */
// ProjectionData is defined in types.ts (canonical source)
import type { ProjectionData } from '../types';
export type { ProjectionData };

interface ActivationSpaceProps {
  /** Pre-computed projection data or null */
  projection: ProjectionData | null;
  /** Current epoch count */
  epoch: number;
  /** Predicted label for the drawn digit */
  predictedLabel: number | null;
}

export default function ActivationSpace({
  projection,
  epoch,
  predictedLabel,
}: ActivationSpaceProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { containerRef, dims } = useContainerDims({
    defaultWidth: ACTIVATION_SPACE_DEFAULT.width,
    defaultHeight: ACTIVATION_SPACE_DEFAULT.height,
    aspectRatio: ACTIVATION_SPACE_ASPECT,
  });
  const [hoveredDigit, setHoveredDigit] = useState<number | null>(null);

  const { width, height } = dims;

  // Render
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const pad = 30;
    const plotW = width - pad * 2;
    const plotH = height - pad * 2 - 10;

    if (!projection || projection.points.length === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        epoch === 0
          ? 'Train to see activation clusters form'
          : 'Computing projection…',
        width / 2,
        height / 2,
      );
      return;
    }

    const { points, labels, userProjection } = projection;

    // Compute bounds
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    for (const [x, y] of points) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    if (userProjection) {
      const [ux, uy] = userProjection;
      if (ux < minX) minX = ux;
      if (ux > maxX) maxX = ux;
      if (uy < minY) minY = uy;
      if (uy > maxY) maxY = uy;
    }
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;
    const mx = rangeX * 0.1;
    const my = rangeY * 0.1;
    minX -= mx; maxX += mx;
    minY -= my; maxY += my;
    const spanX = maxX - minX;
    const spanY = maxY - minY;

    const toCanvas = (px: number, py: number): [number, number] => [
      pad + ((px - minX) / spanX) * plotW,
      pad + 10 + ((py - minY) / spanY) * plotH,
    ];

    // Subtle grid
    ctx.strokeStyle = 'rgba(75, 85, 99, 0.2)';
    ctx.lineWidth = 0.5;
    const gridN = 5;
    for (let i = 0; i <= gridN; i++) {
      const gx = pad + (i / gridN) * plotW;
      const gy = pad + 10 + (i / gridN) * plotH;
      ctx.beginPath();
      ctx.moveTo(gx, pad + 10);
      ctx.lineTo(gx, pad + 10 + plotH);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad, gy);
      ctx.lineTo(pad + plotW, gy);
      ctx.stroke();
    }

    // Axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('PC1', width / 2, height - 2);
    ctx.save();
    ctx.translate(8, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('PC2', 0, 0);
    ctx.restore();

    // Draw training sample points
    for (let i = 0; i < points.length; i++) {
      const [px, py] = toCanvas(points[i][0], points[i][1]);
      const label = labels[i];
      const color = DIGIT_COLORS[label];

      const dimmed = hoveredDigit !== null && label !== hoveredDigit;

      // Glow for hovered class
      if (hoveredDigit === label) {
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(px, py, 7, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.globalAlpha = dimmed ? 0.12 : 0.75;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, dimmed ? 2.5 : 3.5, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = 1;

    // Draw user's digit as a large highlighted marker
    if (userProjection && predictedLabel !== null) {
      const [ux, uy] = toCanvas(userProjection[0], userProjection[1]);
      const color = DIGIT_COLORS[predictedLabel];

      // Outer glow ring
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.arc(ux, uy, 12, 0, Math.PI * 2);
      ctx.stroke();

      // Radial glow
      const grad = ctx.createRadialGradient(ux, uy, 0, ux, uy, 16);
      grad.addColorStop(0, color + '40');
      grad.addColorStop(1, color + '00');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(ux, uy, 16, 0, Math.PI * 2);
      ctx.fill();

      // Core dot
      ctx.globalAlpha = 1;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(ux, uy, 6, 0, Math.PI * 2);
      ctx.fill();

      // White border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(ux, uy, 6, 0, Math.PI * 2);
      ctx.stroke();

      // Label
      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(String(predictedLabel), ux, uy + 3);
    }
  }, [width, height, projection, epoch, hoveredDigit, predictedLabel]);

  useEffect(() => {
    render();
  }, [render]);

  const handleLegendHover = useCallback((digit: number | null) => {
    setHoveredDigit(digit);
  }, []);

  return (
    <div
      className="activation-space"
      ref={containerRef}
      role="group"
      aria-label="Activation space — 2D PCA projection of hidden layer"
    >
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🌌</span>
        <span>Activation Space</span>
        {epoch > 0 && (
          <span className="activation-space-epoch">
            Epoch {epoch}
          </span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="activation-space-canvas"
        role="img"
        aria-label={
          projection
            ? `2D PCA scatter plot of ${projection.points.length} training samples across 10 digit classes at epoch ${epoch}`
            : 'Activation space — train to see clusters'
        }
      />

      {/* Digit color legend */}
      <div className="activation-space-legend" role="list" aria-label="Digit class colors">
        {DIGIT_COLORS.map((color, i) => (
          <button
            key={i}
            className={`legend-dot ${hoveredDigit === i ? 'active' : ''}`}
            style={{ '--dot-color': color } as React.CSSProperties}
            onMouseEnter={() => handleLegendHover(i)}
            onMouseLeave={() => handleLegendHover(null)}
            role="listitem"
            aria-label={`Digit ${i}`}
          >
            {i}
          </button>
        ))}
      </div>

      {projection && (
        <p className="activation-space-hint">
          Hover digits to highlight. Large dot = your drawing.
        </p>
      )}
    </div>
  );
}
