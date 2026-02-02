import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';
import { weightsToImageData } from '../rendering';
import {
  FEATURE_MAP_CELL_SIZE,
  FEATURE_MAP_CELL_GAP,
  FEATURE_MAP_MAGNIFIER_SIZE,
  FEATURE_MAP_MAX_COLS,
} from '../constants';

interface FeatureMapsProps {
  layers: LayerState[] | null;
}

/**
 * Feature Maps — visualize what first-layer neurons have learned.
 *
 * Each neuron in the first hidden layer has 784 input weights.
 * Reshaping those weights into a 28×28 grid reveals what visual features
 * the neuron has learned to detect (edges, curves, blobs, etc).
 *
 * This is THE classic "whoa" moment in neural network education.
 */
export function FeatureMaps({ layers }: FeatureMapsProps) {
  const gridCanvasRef = useRef<HTMLCanvasElement>(null);
  const magCanvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredNeuron, setHoveredNeuron] = useState<number | null>(null);

  const firstLayer = layers && layers.length > 0 ? layers[0] : null;
  const numNeurons = firstLayer ? firstLayer.weights.length : 0;

  // Compute grid layout (derived state — no effect needed)
  const gridDims = useMemo(() => {
    if (numNeurons === 0) return { cols: 0, rows: 0, total: 0 };
    const cols = Math.min(numNeurons, FEATURE_MAP_MAX_COLS);
    const rows = Math.ceil(numNeurons / cols);
    return { cols, rows, total: numNeurons };
  }, [numNeurons]);

  /** Render a single neuron's weights as a diverging colormap ImageData */
  const renderNeuronToImageData = useCallback(weightsToImageData, []);

  // Render the grid of mini feature maps
  useEffect(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas || !firstLayer || gridDims.total === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { cols, rows } = gridDims;
    const gridW = cols * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP) - FEATURE_MAP_CELL_GAP;
    const gridH = rows * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP) - FEATURE_MAP_CELL_GAP;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = gridW * dpr;
    canvas.height = gridH * dpr;
    canvas.style.width = `${gridW}px`;
    canvas.style.height = `${gridH}px`;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, gridW, gridH);

    for (let n = 0; n < firstLayer.weights.length; n++) {
      const col = n % cols;
      const row = Math.floor(n / cols);
      const x = col * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP);
      const y = row * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP);

      const imgData = renderNeuronToImageData(firstLayer.weights[n], FEATURE_MAP_CELL_SIZE);

      // Draw onto an offscreen canvas first (putImageData ignores scale)
      const offscreen = document.createElement('canvas');
      offscreen.width = FEATURE_MAP_CELL_SIZE;
      offscreen.height = FEATURE_MAP_CELL_SIZE;
      const offCtx = offscreen.getContext('2d');
      if (offCtx) {
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, x, y, FEATURE_MAP_CELL_SIZE, FEATURE_MAP_CELL_SIZE);
      }

      // Hover highlight
      if (hoveredNeuron === n) {
        ctx.strokeStyle = 'rgba(99, 222, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, FEATURE_MAP_CELL_SIZE + 2, FEATURE_MAP_CELL_SIZE + 2);
      }

      // Neuron index label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(x, y + FEATURE_MAP_CELL_SIZE - 12, 18, 12);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`${n}`, x + 2, y + FEATURE_MAP_CELL_SIZE - 3);

      // Activation bar
      const act = firstLayer.activations[n] || 0;
      const barW = Math.abs(act) / 2 * FEATURE_MAP_CELL_SIZE; // scale
      ctx.fillStyle = act > 0
        ? `rgba(99, 222, 255, ${0.5 + Math.abs(act) * 0.3})`
        : `rgba(255, 99, 132, ${0.5 + Math.abs(act) * 0.3})`;
      ctx.fillRect(x, y, Math.min(barW, FEATURE_MAP_CELL_SIZE), 2);
    }
  }, [firstLayer, gridDims, hoveredNeuron, renderNeuronToImageData]);

  // Render magnified view of hovered neuron
  useEffect(() => {
    const canvas = magCanvasRef.current;
    if (!canvas || !firstLayer || hoveredNeuron === null) return;
    if (hoveredNeuron >= firstLayer.weights.length) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = FEATURE_MAP_MAGNIFIER_SIZE * dpr;
    canvas.height = FEATURE_MAP_MAGNIFIER_SIZE * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, FEATURE_MAP_MAGNIFIER_SIZE, FEATURE_MAP_MAGNIFIER_SIZE);

    const imgData = renderNeuronToImageData(firstLayer.weights[hoveredNeuron], FEATURE_MAP_MAGNIFIER_SIZE);
    const offscreen = document.createElement('canvas');
    offscreen.width = FEATURE_MAP_MAGNIFIER_SIZE;
    offscreen.height = FEATURE_MAP_MAGNIFIER_SIZE;
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(offscreen, 0, 0, FEATURE_MAP_MAGNIFIER_SIZE, FEATURE_MAP_MAGNIFIER_SIZE);
    }

    // Label
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, FEATURE_MAP_MAGNIFIER_SIZE - 22, FEATURE_MAP_MAGNIFIER_SIZE, 22);
    ctx.fillStyle = '#e5e7eb';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const act = firstLayer.activations[hoveredNeuron] || 0;
    ctx.fillText(`Neuron ${hoveredNeuron} — activation: ${act.toFixed(3)}`, FEATURE_MAP_MAGNIFIER_SIZE / 2, FEATURE_MAP_MAGNIFIER_SIZE - 7);
  }, [firstLayer, hoveredNeuron, renderNeuronToImageData]);

  // Handle hover detection on grid canvas
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (gridDims.total === 0) return;
    const canvas = gridCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const col = Math.floor(mx / (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP));
    const row = Math.floor(my / (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP));
    const idx = row * gridDims.cols + col;

    // Check if cursor is actually within a cell (not in the gap)
    const cellX = mx - col * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP);
    const cellY = my - row * (FEATURE_MAP_CELL_SIZE + FEATURE_MAP_CELL_GAP);

    if (cellX >= 0 && cellX < FEATURE_MAP_CELL_SIZE && cellY >= 0 && cellY < FEATURE_MAP_CELL_SIZE && idx < gridDims.total) {
      setHoveredNeuron(idx);
    } else {
      setHoveredNeuron(null);
    }
  }, [gridDims]);

  const handleMouseLeave = useCallback(() => {
    setHoveredNeuron(null);
  }, []);

  if (!firstLayer || numNeurons === 0) {
    return (
      <div className="feature-maps" role="group" aria-label="Feature maps">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">🔬</span>
          <span>What Neurons See</span>
        </div>
        <div className="feature-maps-empty">
          <p>Train the network to see what features each neuron learns to detect.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="feature-maps" role="group" aria-label="Feature maps — first layer weight visualization">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔬</span>
        <span>What Neurons See</span>
      </div>
      <div className="feature-maps-content">
        <div className="feature-maps-grid-row">
          <canvas
            ref={gridCanvasRef}
            className="feature-maps-canvas"
            role="img"
            aria-label={`Grid of ${numNeurons} neuron feature maps from the first hidden layer`}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
          />
          {hoveredNeuron !== null && (
            <div className="feature-maps-magnifier">
              <canvas
                ref={magCanvasRef}
                style={{ width: FEATURE_MAP_MAGNIFIER_SIZE, height: FEATURE_MAP_MAGNIFIER_SIZE }}
                className="feature-maps-mag-canvas"
                role="img"
                aria-label={`Enlarged feature map for neuron ${hoveredNeuron}`}
              />
            </div>
          )}
        </div>
        <p className="feature-maps-hint">
          Each tile shows what a first-layer neuron has learned to detect. Hover to magnify.
          <br />
          <span className="feature-maps-legend">
            <span className="legend-item legend-cyan">● Positive weights</span>
            <span className="legend-item legend-red">● Negative weights</span>
          </span>
        </p>
      </div>
    </div>
  );
}

export default FeatureMaps;
