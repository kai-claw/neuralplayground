/**
 * DecisionBoundary — Visualize how the network separates two digit classes.
 *
 * Generates a 2D heatmap showing the network's confidence landscape
 * as it transitions between two digit exemplars. The boundary between
 * colors reveals where the network is uncertain — the "decision frontier".
 *
 * Beautiful gradient visualization that makes an abstract concept tangible.
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { useContainerDims } from '../hooks/useContainerDims';
import { computeDecisionBoundary, renderDecisionBoundary } from '../nn/decisionBoundary';
import type { DecisionBoundaryResult } from '../nn/decisionBoundary';
import {
  DECISION_BOUNDARY_DISPLAY,
  DECISION_BOUNDARY_RESOLUTION,
} from '../constants';

// Use the predict method signature — avoids importing NeuralNetwork type directly
interface Predictable {
  predict(input: number[]): { label: number; probabilities: number[]; layers: unknown[] };
}

/** Digit pair presets that show interesting boundaries */
const INTERESTING_PAIRS: [number, number, string][] = [
  [3, 8, '3 vs 8 — subtle curve difference'],
  [1, 7, '1 vs 7 — the crossbar question'],
  [4, 9, '4 vs 9 — loop vs angle'],
  [5, 6, '5 vs 6 — open vs closed'],
  [0, 6, '0 vs 6 — tail detection'],
  [2, 7, '2 vs 7 — curve vs straight'],
];

interface DecisionBoundaryProps {
  /** Reference to the live neural network */
  networkRef: React.RefObject<Predictable | null>;
  /** Current epoch (triggers recomputation) */
  epoch: number;
  /** Whether training is active */
  isTraining: boolean;
}

export default function DecisionBoundary({
  networkRef,
  epoch,
  isTraining,
}: DecisionBoundaryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [digitA, setDigitA] = useState(3);
  const [digitB, setDigitB] = useState(8);
  const [boundaryResult, setBoundaryResult] = useState<DecisionBoundaryResult | null>(null);
  const [isComputing, setIsComputing] = useState(false);
  const [hoveredCell, setHoveredCell] = useState<{ x: number; y: number } | null>(null);
  const lastComputedEpochRef = useRef<number>(-1);

  const { containerRef, dims } = useContainerDims({
    defaultWidth: DECISION_BOUNDARY_DISPLAY.width,
    defaultHeight: DECISION_BOUNDARY_DISPLAY.width, // square
    aspectRatio: 1,
  });

  const displaySize = Math.min(dims.width, DECISION_BOUNDARY_DISPLAY.width);

  // Compute boundary when epoch changes or digits change (throttled)
  useEffect(() => {
    if (!networkRef.current || epoch === 0) return;
    if (isTraining && epoch - lastComputedEpochRef.current < 5) return; // throttle during training

    lastComputedEpochRef.current = epoch;
    setIsComputing(true);

    // Use requestIdleCallback or setTimeout to avoid blocking
    const timer = setTimeout(() => {
      if (!networkRef.current) return;
      const result = computeDecisionBoundary(
        networkRef.current,
        digitA,
        digitB,
        DECISION_BOUNDARY_RESOLUTION,
      );
      setBoundaryResult(result);
      setIsComputing(false);
    }, 50);

    return () => clearTimeout(timer);
  }, [epoch, digitA, digitB, networkRef, isTraining]);

  // Render the heatmap
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const size = displaySize;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    if (!boundaryResult) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(
        epoch === 0
          ? 'Train to see decision boundaries form'
          : isComputing ? 'Computing boundary…' : 'Select digit pair',
        size / 2,
        size / 2,
      );
      return;
    }

    // Render the boundary heatmap
    const imageData = renderDecisionBoundary(boundaryResult, size);
    ctx.putImageData(imageData, 0, 0);

    // Axis labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.textAlign = 'center';

    // X axis
    ctx.fillText(`← ${digitA}`, 30, size - 6);
    ctx.fillText(`${digitB} →`, size - 30, size - 6);

    // Y axis (rotated)
    ctx.save();
    ctx.translate(12, size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Variation', 0, 0);
    ctx.restore();

    // Hover tooltip
    if (hoveredCell) {
      const { grid, resolution } = boundaryResult;
      const cellSize = size / resolution;
      const gx = Math.min(resolution - 1, Math.floor(hoveredCell.x / cellSize));
      const gy = Math.min(resolution - 1, Math.floor(hoveredCell.y / cellSize));
      const cell = grid[gy]?.[gx];

      if (cell) {
        const tooltipX = Math.min(size - 110, hoveredCell.x + 10);
        const tooltipY = Math.max(50, hoveredCell.y - 10);

        // Background
        ctx.fillStyle = 'rgba(17, 24, 39, 0.92)';
        ctx.strokeStyle = 'rgba(99, 222, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY - 44, 100, 44, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#e5e7eb';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`Predicts: ${cell.label}`, tooltipX + 8, tooltipY - 28);
        ctx.fillText(
          `${digitA}: ${(cell.confA * 100).toFixed(0)}% | ${digitB}: ${(cell.confB * 100).toFixed(0)}%`,
          tooltipX + 8,
          tooltipY - 14,
        );

        // Confidence indicator
        const confBarY = tooltipY - 6;
        ctx.fillStyle = 'rgba(54, 162, 235, 0.6)';
        ctx.fillRect(tooltipX + 8, confBarY, cell.confA * 84, 3);
        ctx.fillStyle = 'rgba(255, 99, 132, 0.6)';
        ctx.fillRect(tooltipX + 8 + cell.confA * 84, confBarY, cell.confB * 84, 3);
      }
    }
  }, [displaySize, boundaryResult, epoch, isComputing, hoveredCell, digitA, digitB]);

  useEffect(() => {
    render();
  }, [render]);

  // Mouse tracking for tooltip
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    setHoveredCell({
      x: (e.clientX - rect.left) * (displaySize / rect.width),
      y: (e.clientY - rect.top) * (displaySize / rect.height),
    });
  }, [displaySize]);

  const handleMouseLeave = useCallback(() => setHoveredCell(null), []);

  // Boundary stats
  const stats = useMemo(() => {
    if (!boundaryResult) return null;
    const { grid, resolution } = boundaryResult;
    let aCount = 0;
    let bCount = 0;
    let otherCount = 0;
    let uncertainCells = 0;

    for (let y = 0; y < resolution; y++) {
      for (let x = 0; x < resolution; x++) {
        const cell = grid[y][x];
        if (cell.label === boundaryResult.digitA) aCount++;
        else if (cell.label === boundaryResult.digitB) bCount++;
        else otherCount++;
        if (Math.abs(cell.confA - cell.confB) < 0.15) uncertainCells++;
      }
    }

    const total = resolution * resolution;
    return {
      aPercent: ((aCount / total) * 100).toFixed(0),
      bPercent: ((bCount / total) * 100).toFixed(0),
      otherPercent: ((otherCount / total) * 100).toFixed(0),
      uncertainPercent: ((uncertainCells / total) * 100).toFixed(0),
    };
  }, [boundaryResult]);

  return (
    <div
      className="decision-boundary"
      ref={containerRef}
      role="group"
      aria-label="Decision boundary map between two digits"
    >
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🗺️</span>
        <span>Decision Boundary</span>
        {isComputing && <span className="boundary-computing">Computing…</span>}
      </div>

      {/* Digit pair selector */}
      <div className="boundary-selector">
        <div className="boundary-digit-pair">
          <select
            value={digitA}
            onChange={e => setDigitA(parseInt(e.target.value, 10))}
            className="boundary-select"
            aria-label="Digit A"
          >
            {Array.from({ length: 10 }, (_, i) => (
              <option key={i} value={i} disabled={i === digitB}>{i}</option>
            ))}
          </select>
          <span className="boundary-vs">vs</span>
          <select
            value={digitB}
            onChange={e => setDigitB(parseInt(e.target.value, 10))}
            className="boundary-select"
            aria-label="Digit B"
          >
            {Array.from({ length: 10 }, (_, i) => (
              <option key={i} value={i} disabled={i === digitA}>{i}</option>
            ))}
          </select>
        </div>

        {/* Quick pair presets */}
        <div className="boundary-presets">
          {INTERESTING_PAIRS.map(([a, b, label]) => (
            <button
              key={`${a}-${b}`}
              className={`boundary-preset-btn ${digitA === a && digitB === b ? 'active' : ''}`}
              onClick={() => { setDigitA(a); setDigitB(b); }}
              title={label}
              aria-label={label}
            >
              {a}↔{b}
            </button>
          ))}
        </div>
      </div>

      <canvas
        ref={canvasRef}
        style={{ width: displaySize, height: displaySize }}
        className="boundary-canvas"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        role="img"
        aria-label={
          boundaryResult
            ? `Decision boundary heatmap between digits ${digitA} and ${digitB}`
            : 'Decision boundary — train network first'
        }
      />

      {/* Stats row */}
      {stats && (
        <div className="boundary-stats">
          <span className="stat-a">
            <span className="stat-dot" style={{ background: '#36a2eb' }} />
            {digitA}: {stats.aPercent}%
          </span>
          <span className="stat-b">
            <span className="stat-dot" style={{ background: '#ff6384' }} />
            {digitB}: {stats.bPercent}%
          </span>
          <span className="stat-uncertain">
            ⚡ Boundary: {stats.uncertainPercent}%
          </span>
        </div>
      )}

      {!boundaryResult && epoch === 0 && (
        <p className="boundary-hint">
          Train the network, then pick two digits to visualize how it separates them.
        </p>
      )}
    </div>
  );
}
