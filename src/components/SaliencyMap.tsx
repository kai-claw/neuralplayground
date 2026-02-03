/**
 * SaliencyMap — "Where is the network looking?"
 *
 * Renders a gradient-based attention heatmap showing which pixels
 * matter most for the current prediction. Hot pixels = high importance.
 * Uses inferno-like colormap overlaid on the original digit.
 *
 * This is one of the most illuminating ML visualizations — watch
 * where the network focuses attention as it learns.
 */

import { useRef, useEffect, useMemo, useState } from 'react';
import { renderSaliencyOverlay, saliencyToColor } from '../nn/saliency';
import { SALIENCY_DISPLAY_SIZE, SALIENCY_HOT_THRESHOLD } from '../constants';

interface SaliencyMapProps {
  /** Pre-computed saliency values (784 floats in [0,1]) or null */
  saliency: Float32Array | null;
  /** Current drawing as 784 pixel values */
  currentInput: number[] | null;
  /** Predicted digit class */
  predictedLabel: number | null;
  /** Network has been trained */
  hasTrained: boolean;
}

/** Pre-rendered legend bar for the colormap. */
function ColorLegend() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = 120;
    const h = 10;
    canvas.width = w * 2;
    canvas.height = h * 2;
    ctx.scale(2, 2);

    for (let x = 0; x < w; x++) {
      const t = x / (w - 1);
      const [r, g, b] = saliencyToColor(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(x, 0, 1, h);
    }
  }, []);

  return (
    <div className="saliency-legend">
      <span className="legend-label">Low</span>
      <canvas
        ref={canvasRef}
        style={{ width: 120, height: 10, borderRadius: 3 }}
        aria-hidden="true"
      />
      <span className="legend-label">High</span>
    </div>
  );
}

export default function SaliencyMap({
  saliency,
  currentInput,
  predictedLabel,
  hasTrained,
}: SaliencyMapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Reuse offscreen canvas across renders (avoid createElement every time)
  const offscreenRef = useRef<HTMLCanvasElement | null>(null);
  const offscreenCtxRef = useRef<CanvasRenderingContext2D | null>(null);
  const [topPixels, setTopPixels] = useState<number>(0);

  // Render the overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !currentInput || !saliency) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = SALIENCY_DISPLAY_SIZE;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const imageData = renderSaliencyOverlay(currentInput, saliency, size);
    // Reuse offscreen canvas (avoids document.createElement per render)
    if (!offscreenRef.current || offscreenRef.current.width !== size) {
      offscreenRef.current = document.createElement('canvas');
      offscreenRef.current.width = size;
      offscreenRef.current.height = size;
      offscreenCtxRef.current = offscreenRef.current.getContext('2d');
    }
    const offCtx = offscreenCtxRef.current;
    if (offCtx) {
      offCtx.putImageData(imageData, 0, 0);
      ctx.drawImage(offscreenRef.current!, 0, 0, size, size);
    }

    // Count "hot" pixels (above threshold) as percentage
    let hot = 0;
    for (let i = 0; i < saliency.length; i++) {
      if (saliency[i] > SALIENCY_HOT_THRESHOLD) hot++;
    }
    setTopPixels(Math.round((hot / saliency.length) * 100));
  }, [currentInput, saliency]);

  // Peak saliency value
  const peakSaliency = useMemo(() => {
    if (!saliency) return 0;
    let max = 0;
    for (let i = 0; i < saliency.length; i++) {
      if (saliency[i] > max) max = saliency[i];
    }
    return max;
  }, [saliency]);

  const hasData = currentInput && saliency && hasTrained;

  return (
    <div className="saliency-map" role="group" aria-label="Saliency map — attention heatmap">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔥</span>
        <span>Attention Map</span>
      </div>

      {!hasData ? (
        <div className="saliency-empty">
          <p>Draw a digit to see where the network focuses attention.</p>
        </div>
      ) : (
        <div className="saliency-content">
          <div className="saliency-canvas-wrapper">
            <canvas
              ref={canvasRef}
              style={{ width: SALIENCY_DISPLAY_SIZE, height: SALIENCY_DISPLAY_SIZE }}
              className="saliency-canvas"
              role="img"
              aria-label={`Saliency heatmap showing attention for digit ${predictedLabel}. ${topPixels}% of pixels are important.`}
            />
            {/* Pulsing hotspot indicator */}
            <div className="saliency-pulse" aria-hidden="true" />
          </div>

          <div className="saliency-stats">
            <div className="saliency-stat">
              <span className="stat-label">Focus</span>
              <span className="stat-value stat-glow">{topPixels}%</span>
              <span className="stat-desc">of pixels matter</span>
            </div>
            <div className="saliency-stat">
              <span className="stat-label">Peak</span>
              <span className="stat-value stat-glow">{peakSaliency.toFixed(2)}</span>
              <span className="stat-desc">max gradient</span>
            </div>
          </div>

          <ColorLegend />

          <p className="saliency-hint">
            Bright pixels = what the network "sees" for digit <strong>{predictedLabel}</strong>
          </p>
        </div>
      )}
    </div>
  );
}
