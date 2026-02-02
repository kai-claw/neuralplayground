import { useRef, useEffect, useState, useCallback } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';

interface FeatureMapsProps {
  layers: LayerState[] | null;
}

/** Size of each mini feature map in CSS pixels */
const CELL_SIZE = 38;
const CELL_GAP = 3;
const MAGNIFIER_SIZE = 140;
const INPUT_DIM = 28; // 28×28 MNIST

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
  const [gridDims, setGridDims] = useState({ cols: 0, rows: 0, total: 0 });

  const firstLayer = layers && layers.length > 0 ? layers[0] : null;
  const numNeurons = firstLayer ? firstLayer.weights.length : 0;

  // Compute grid layout
  useEffect(() => {
    if (numNeurons === 0) return;
    const cols = Math.min(numNeurons, 8);
    const rows = Math.ceil(numNeurons / cols);
    setGridDims({ cols, rows, total: numNeurons });
  }, [numNeurons]);

  /** Render a single neuron's weights as a 28×28 grayscale image onto an offscreen ImageData */
  const renderNeuronToImageData = useCallback((
    weights: number[],
    size: number,
  ): ImageData => {
    const imgData = new ImageData(size, size);
    const data = imgData.data;

    // Find weight range for contrast normalization
    let wMin = Infinity;
    let wMax = -Infinity;
    const len = Math.min(weights.length, INPUT_DIM * INPUT_DIM);
    for (let i = 0; i < len; i++) {
      if (weights[i] < wMin) wMin = weights[i];
      if (weights[i] > wMax) wMax = weights[i];
    }
    const range = wMax - wMin || 1;

    // Scale factor for rendering at non-native sizes
    const scale = size / INPUT_DIM;

    for (let py = 0; py < size; py++) {
      const sy = Math.min(Math.floor(py / scale), INPUT_DIM - 1);
      for (let px = 0; px < size; px++) {
        const sx = Math.min(Math.floor(px / scale), INPUT_DIM - 1);
        const wi = sy * INPUT_DIM + sx;
        const norm = wi < len ? (weights[wi] - wMin) / range : 0.5;

        // Diverging colormap: negative (red/orange) ↔ zero (dark) ↔ positive (cyan/blue)
        const idx = (py * size + px) * 4;
        if (norm >= 0.5) {
          const t = (norm - 0.5) * 2; // 0→1
          data[idx]     = Math.round(20 + t * 79);    // R: 20→99
          data[idx + 1] = Math.round(40 + t * 182);   // G: 40→222
          data[idx + 2] = Math.round(60 + t * 195);   // B: 60→255
        } else {
          const t = (0.5 - norm) * 2; // 0→1
          data[idx]     = Math.round(20 + t * 235);   // R: 20→255
          data[idx + 1] = Math.round(40 + t * 59);    // G: 40→99
          data[idx + 2] = Math.round(60 + t * 72);    // B: 60→132
        }
        data[idx + 3] = 255;
      }
    }

    return imgData;
  }, []);

  // Render the grid of mini feature maps
  useEffect(() => {
    const canvas = gridCanvasRef.current;
    if (!canvas || !firstLayer || gridDims.total === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { cols, rows } = gridDims;
    const gridW = cols * (CELL_SIZE + CELL_GAP) - CELL_GAP;
    const gridH = rows * (CELL_SIZE + CELL_GAP) - CELL_GAP;

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
      const x = col * (CELL_SIZE + CELL_GAP);
      const y = row * (CELL_SIZE + CELL_GAP);

      const imgData = renderNeuronToImageData(firstLayer.weights[n], CELL_SIZE);

      // Draw onto an offscreen canvas first (putImageData ignores scale)
      const offscreen = document.createElement('canvas');
      offscreen.width = CELL_SIZE;
      offscreen.height = CELL_SIZE;
      const offCtx = offscreen.getContext('2d');
      if (offCtx) {
        offCtx.putImageData(imgData, 0, 0);
        ctx.drawImage(offscreen, x, y, CELL_SIZE, CELL_SIZE);
      }

      // Hover highlight
      if (hoveredNeuron === n) {
        ctx.strokeStyle = 'rgba(99, 222, 255, 0.8)';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, CELL_SIZE + 2, CELL_SIZE + 2);
      }

      // Neuron index label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
      ctx.fillRect(x, y + CELL_SIZE - 12, 18, 12);
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      ctx.font = '8px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(`${n}`, x + 2, y + CELL_SIZE - 3);

      // Activation bar
      const act = firstLayer.activations[n] || 0;
      const barW = Math.abs(act) / 2 * CELL_SIZE; // scale
      ctx.fillStyle = act > 0
        ? `rgba(99, 222, 255, ${0.5 + Math.abs(act) * 0.3})`
        : `rgba(255, 99, 132, ${0.5 + Math.abs(act) * 0.3})`;
      ctx.fillRect(x, y, Math.min(barW, CELL_SIZE), 2);
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
    canvas.width = MAGNIFIER_SIZE * dpr;
    canvas.height = MAGNIFIER_SIZE * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, MAGNIFIER_SIZE, MAGNIFIER_SIZE);

    const imgData = renderNeuronToImageData(firstLayer.weights[hoveredNeuron], MAGNIFIER_SIZE);
    const offscreen = document.createElement('canvas');
    offscreen.width = MAGNIFIER_SIZE;
    offscreen.height = MAGNIFIER_SIZE;
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(imgData, 0, 0);
      ctx.drawImage(offscreen, 0, 0, MAGNIFIER_SIZE, MAGNIFIER_SIZE);
    }

    // Label
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(0, MAGNIFIER_SIZE - 22, MAGNIFIER_SIZE, 22);
    ctx.fillStyle = '#e5e7eb';
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const act = firstLayer.activations[hoveredNeuron] || 0;
    ctx.fillText(`Neuron ${hoveredNeuron} — activation: ${act.toFixed(3)}`, MAGNIFIER_SIZE / 2, MAGNIFIER_SIZE - 7);
  }, [firstLayer, hoveredNeuron, renderNeuronToImageData]);

  // Handle hover detection on grid canvas
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (gridDims.total === 0) return;
    const canvas = gridCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const col = Math.floor(mx / (CELL_SIZE + CELL_GAP));
    const row = Math.floor(my / (CELL_SIZE + CELL_GAP));
    const idx = row * gridDims.cols + col;

    // Check if cursor is actually within a cell (not in the gap)
    const cellX = mx - col * (CELL_SIZE + CELL_GAP);
    const cellY = my - row * (CELL_SIZE + CELL_GAP);

    if (cellX >= 0 && cellX < CELL_SIZE && cellY >= 0 && cellY < CELL_SIZE && idx < gridDims.total) {
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
                style={{ width: MAGNIFIER_SIZE, height: MAGNIFIER_SIZE }}
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
