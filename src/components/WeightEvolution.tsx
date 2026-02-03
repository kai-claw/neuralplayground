import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import type { WeightFrame } from '../nn/weightEvolution';
import { renderNeuronWeights } from '../nn/weightEvolution';
import {
  WEIGHT_EVOLUTION_CELL_SIZE,
  WEIGHT_EVOLUTION_MAX_NEURONS,
  WEIGHT_EVOLUTION_PLAYBACK_INTERVAL,
} from '../constants';

interface WeightEvolutionProps {
  frames: WeightFrame[];
}

/**
 * Weight Evolution Filmstrip — watch weights morph from random noise
 * into learned feature detectors over training epochs.
 *
 * Shows a timeline scrubber + grid of first-hidden-layer neuron weights
 * rendered as diverging colormaps (cyan = negative, amber = positive).
 */
export default function WeightEvolution({ frames }: WeightEvolutionProps) {
  const [selectedFrame, setSelectedFrame] = useState<number>(-1); // -1 = latest
  const [playing, setPlaying] = useState(false);
  const [selectedNeuron, setSelectedNeuron] = useState<number | null>(null);
  const gridRef = useRef<HTMLCanvasElement>(null);
  const magnifierRef = useRef<HTMLCanvasElement>(null);
  const playRef = useRef(false);

  const totalFrames = frames.length;
  const activeIdx = selectedFrame < 0 ? totalFrames - 1 : Math.min(selectedFrame, totalFrames - 1);
  const frame = totalFrames > 0 ? frames[activeIdx] : null;

  const neuronCount = frame?.neuronCount ?? 0;
  const inputSize = neuronCount > 0 ? frame!.weights.length / neuronCount : 784;
  const displayNeurons = Math.min(neuronCount, WEIGHT_EVOLUTION_MAX_NEURONS);
  const dim = Math.round(Math.sqrt(inputSize));
  const cellSize = WEIGHT_EVOLUTION_CELL_SIZE;

  // Grid layout
  const cols = Math.min(displayNeurons, 8);
  const rows = Math.ceil(displayNeurons / cols);

  // Pre-allocated ImageData for rendering
  const imageDataRef = useRef<ImageData | null>(null);

  // Render the neuron weight grid
  useEffect(() => {
    const canvas = gridRef.current;
    if (!canvas || !frame) return;

    const w = cols * (cellSize + 2) + 2;
    const h = rows * (cellSize + 2) + 2;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Reuse or create ImageData for a single neuron tile
    if (!imageDataRef.current || imageDataRef.current.width !== dim) {
      imageDataRef.current = new ImageData(dim, dim);
    }
    const imgData = imageDataRef.current;

    for (let n = 0; n < displayNeurons; n++) {
      renderNeuronWeights(frame.weights, n, inputSize, imgData);

      const col = n % cols;
      const row = Math.floor(n / cols);
      const x = col * (cellSize + 2) + 2;
      const y = row * (cellSize + 2) + 2;

      // Scale up the dim×dim image to cellSize×cellSize
      const tmpCanvas = document.createElement('canvas');
      tmpCanvas.width = dim;
      tmpCanvas.height = dim;
      const tmpCtx = tmpCanvas.getContext('2d')!;
      tmpCtx.putImageData(imgData, 0, 0);

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmpCanvas, x, y, cellSize, cellSize);

      // Highlight selected neuron
      if (selectedNeuron === n) {
        ctx.strokeStyle = '#63deff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, cellSize + 2, cellSize + 2);
      }
    }
  }, [frame, displayNeurons, cols, rows, cellSize, dim, inputSize, selectedNeuron]);

  // Render magnified view of selected neuron
  useEffect(() => {
    const canvas = magnifierRef.current;
    if (!canvas || !frame || selectedNeuron === null) return;

    const magSize = 112;
    canvas.width = magSize;
    canvas.height = magSize;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!imageDataRef.current || imageDataRef.current.width !== dim) {
      imageDataRef.current = new ImageData(dim, dim);
    }
    renderNeuronWeights(frame.weights, selectedNeuron, inputSize, imageDataRef.current);

    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = dim;
    tmpCanvas.height = dim;
    const tmpCtx = tmpCanvas.getContext('2d')!;
    tmpCtx.putImageData(imageDataRef.current, 0, 0);

    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmpCanvas, 0, 0, magSize, magSize);
  }, [frame, selectedNeuron, dim, inputSize]);

  // Playback timer
  useEffect(() => {
    if (!playing || totalFrames < 2) return;
    playRef.current = true;
    let idx = activeIdx;

    const timer = setInterval(() => {
      if (!playRef.current) return;
      idx++;
      if (idx >= totalFrames) {
        idx = 0;
      }
      setSelectedFrame(idx);
    }, WEIGHT_EVOLUTION_PLAYBACK_INTERVAL);

    return () => {
      playRef.current = false;
      clearInterval(timer);
    };
  }, [playing, totalFrames]); // eslint-disable-line react-hooks/exhaustive-deps

  const togglePlayback = useCallback(() => {
    if (totalFrames < 2) return;
    setPlaying(prev => !prev);
  }, [totalFrames]);

  // Handle neuron click on grid
  const handleGridClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = gridRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const col = Math.floor((x - 2) / (cellSize + 2));
    const row = Math.floor((y - 2) / (cellSize + 2));
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
      const n = row * cols + col;
      if (n < displayNeurons) {
        setSelectedNeuron(prev => prev === n ? null : n);
      }
    }
  }, [cellSize, cols, rows, displayNeurons]);

  // Change magnitude for timeline sparkline
  const changeIntensity = useMemo(() => {
    if (totalFrames < 2) return [];
    const intensities: number[] = [0];
    for (let i = 1; i < totalFrames; i++) {
      const a = frames[i - 1];
      const b = frames[i];
      const len = Math.min(a.weights.length, b.weights.length);
      let sum = 0;
      for (let j = 0; j < len; j++) {
        sum += Math.abs(b.weights[j] - a.weights[j]);
      }
      intensities.push(sum / len);
    }
    return intensities;
  }, [frames, totalFrames]);

  if (totalFrames === 0) {
    return (
      <div className="weight-evolution panel">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">🎞️</span>
          <span>Weight Evolution</span>
        </div>
        <p className="empty-hint">Train the network to see weights evolve…</p>
      </div>
    );
  }

  return (
    <div className="weight-evolution panel">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🎞️</span>
        <span>Weight Evolution</span>
        <span className="panel-badge">{totalFrames} frames</span>
      </div>

      {/* Timeline scrubber */}
      <div className="we-timeline" role="group" aria-label="Timeline scrubber">
        <button
          className={`we-play-btn ${playing ? 'playing' : ''}`}
          onClick={togglePlayback}
          aria-label={playing ? 'Pause playback' : 'Play evolution'}
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? '⏸' : '▶'}
        </button>
        <input
          type="range"
          min={0}
          max={totalFrames - 1}
          value={activeIdx}
          onChange={e => { setPlaying(false); setSelectedFrame(Number(e.target.value)); }}
          className="we-scrubber"
          aria-label={`Epoch ${frame?.epoch ?? 0}`}
        />
        <span className="we-epoch-label">
          E{frame?.epoch ?? 0}
        </span>
      </div>

      {/* Metrics for current frame */}
      <div className="we-metrics">
        <span className="we-metric">
          Loss: <strong className={frame && frame.loss < 0.5 ? 'stat-good' : ''}>{frame?.loss.toFixed(3) ?? '—'}</strong>
        </span>
        <span className="we-metric">
          Acc: <strong className={frame && frame.accuracy > 0.8 ? 'stat-good' : ''}>{frame ? (frame.accuracy * 100).toFixed(1) + '%' : '—'}</strong>
        </span>
      </div>

      {/* Change intensity sparkline */}
      {changeIntensity.length > 1 && (
        <div className="we-sparkline" aria-hidden="true">
          <svg viewBox={`0 0 ${changeIntensity.length} 20`} preserveAspectRatio="none">
            {(() => {
              let max = 0;
              for (const v of changeIntensity) if (v > max) max = v;
              if (max === 0) max = 1;
              const pts = changeIntensity.map((v, i) =>
                `${i},${20 - (v / max) * 18}`
              ).join(' ');
              return (
                <>
                  <polyline
                    points={pts}
                    fill="none"
                    stroke="rgba(99,222,255,0.5)"
                    strokeWidth="0.8"
                  />
                  {/* Current position marker */}
                  <line
                    x1={activeIdx}
                    y1={0}
                    x2={activeIdx}
                    y2={20}
                    stroke="rgba(99,222,255,0.8)"
                    strokeWidth="0.5"
                  />
                </>
              );
            })()}
          </svg>
          <span className="we-sparkline-label">Δ weight change</span>
        </div>
      )}

      {/* Neuron weight grid */}
      <div className="we-grid-container">
        <canvas
          ref={gridRef}
          className="we-grid"
          onClick={handleGridClick}
          role="img"
          aria-label={`First hidden layer neuron weights at epoch ${frame?.epoch ?? 0}`}
          style={{ cursor: 'pointer' }}
        />

        {/* Magnified view */}
        {selectedNeuron !== null && (
          <div className="we-magnifier">
            <canvas
              ref={magnifierRef}
              className="we-magnifier-canvas"
              role="img"
              aria-label={`Neuron ${selectedNeuron} weight detail`}
            />
            <span className="we-magnifier-label">
              Neuron {selectedNeuron}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
