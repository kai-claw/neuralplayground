import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import type { LayerState, NeuronStatus } from '../nn/NeuralNetwork';
import {
  SURGERY_NODE_RADIUS,
  SURGERY_NODE_SPACING,
  SURGERY_MAX_DISPLAY_NEURONS,
} from '../constants';
import { getActivationColor, mulberry32 } from '../utils';

interface NeuronSurgeryProps {
  layers: LayerState[] | null;
  onSetNeuronStatus: (layerIdx: number, neuronIdx: number, status: NeuronStatus) => void;
  onGetNeuronStatus: (layerIdx: number, neuronIdx: number) => NeuronStatus;
  onClearAll: () => void;
  /** Called to trigger re-prediction after surgery */
  onSurgeryChange: () => void;
  /** Pre-surgery prediction confidence (digit → probability) */
  currentPrediction: number[] | null;
  predictedLabel: number | null;
}

interface NodeInfo {
  x: number;
  y: number;
  layerIdx: number;
  neuronIdx: number;
  activation: number;
}

const STATUS_CYCLE: NeuronStatus[] = ['active', 'frozen', 'killed'];

/**
 * Neuron Surgery — click individual neurons to freeze or kill them
 * and watch how the network compensates or breaks.
 */
export function NeuronSurgery({
  layers,
  onSetNeuronStatus,
  onGetNeuronStatus,
  onClearAll,
  onSurgeryChange,
  currentPrediction,
  predictedLabel,
}: NeuronSurgeryProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<NodeInfo[]>([]);
  const [surgeryCount, setSurgeryCount] = useState({ frozen: 0, killed: 0 });
  const [, setTick] = useState(0); // force re-renders

  // Memoize hidden layers to avoid recreating array every render (stabilizes draw deps)
  const hiddenLayers = useMemo(() => layers ? layers.slice(0, -1) : [], [layers]);
  const numHiddenLayers = hiddenLayers.length;
  const layerSpacing = 90;
  const canvasWidth = Math.max(280, (numHiddenLayers + 1) * layerSpacing + 40);
  const maxNeurons = hiddenLayers.reduce((max, l) =>
    Math.max(max, Math.min(l.activations.length, SURGERY_MAX_DISPLAY_NEURONS)), 0);
  const canvasHeight = Math.max(160, maxNeurons * SURGERY_NODE_SPACING + 60);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layers || layers.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const nodes: NodeInfo[] = [];
    const padding = 30;

    // Only show hidden layers (not output layer)
    let frozenCount = 0;
    let killedCount = 0;

    // Draw connections first (behind nodes)
    for (let l = 0; l < hiddenLayers.length; l++) {
      if (l === 0) continue;
      const prevLayer = hiddenLayers[l - 1];
      const currLayer = hiddenLayers[l];
      const prevCount = Math.min(prevLayer.activations.length, SURGERY_MAX_DISPLAY_NEURONS);
      const currCount = Math.min(currLayer.activations.length, SURGERY_MAX_DISPLAY_NEURONS);
      const prevStartY = canvasHeight / 2 - (prevCount - 1) * SURGERY_NODE_SPACING / 2;
      const currStartY = canvasHeight / 2 - (currCount - 1) * SURGERY_NODE_SPACING / 2;
      const prevX = padding + l * layerSpacing;
      const currX = padding + (l + 1) * layerSpacing;

      // Sample connections for visual clarity (seeded RNG for stable rendering)
      const connRng = mulberry32(l * 1000 + prevCount * 100 + currCount);
      const sampleRate = prevCount * currCount > 60 ? 0.2 : 0.5;
      for (let j = 0; j < currCount; j++) {
        for (let i = 0; i < prevCount; i++) {
          if (connRng() > sampleRate && prevCount * currCount > 20) continue;
          ctx.strokeStyle = 'rgba(75, 85, 99, 0.15)';
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(prevX, prevStartY + i * SURGERY_NODE_SPACING);
          ctx.lineTo(currX, currStartY + j * SURGERY_NODE_SPACING);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    for (let l = 0; l < hiddenLayers.length; l++) {
      const layer = hiddenLayers[l];
      const neuronCount = layer.activations.length;
      const displayCount = Math.min(neuronCount, SURGERY_MAX_DISPLAY_NEURONS);
      const startY = canvasHeight / 2 - (displayCount - 1) * SURGERY_NODE_SPACING / 2;
      const x = padding + (l + 0.5) * layerSpacing;

      // Layer label
      ctx.fillStyle = '#9ca3af';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Layer ${l + 1}`, x, 14);
      ctx.fillText(`(${neuronCount})`, x, 24);

      for (let n = 0; n < displayCount; n++) {
        const y = startY + n * SURGERY_NODE_SPACING;
        const activation = layer.activations[n] || 0;
        const status = onGetNeuronStatus(l, n);

        nodes.push({ x, y, layerIdx: l, neuronIdx: n, activation });

        if (status === 'killed') killedCount++;
        if (status === 'frozen') frozenCount++;

        const r = SURGERY_NODE_RADIUS;

        // Draw glow for active neurons
        if (status === 'active' && Math.abs(activation) > 0.2) {
          const glowAlpha = Math.min(0.4, Math.abs(activation) * 0.3);
          const gradient = ctx.createRadialGradient(x, y, 0, x, y, r * 2.5);
          gradient.addColorStop(0, `rgba(99, 222, 255, ${glowAlpha})`);
          gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        // Node circle
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);

        if (status === 'killed') {
          ctx.fillStyle = 'rgba(239, 68, 68, 0.3)';
          ctx.fill();
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2;
          ctx.stroke();
          // X mark
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(x - 4, y - 4);
          ctx.lineTo(x + 4, y + 4);
          ctx.moveTo(x + 4, y - 4);
          ctx.lineTo(x - 4, y + 4);
          ctx.stroke();
        } else if (status === 'frozen') {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
          ctx.fill();
          ctx.strokeStyle = '#3b82f6';
          ctx.lineWidth = 2;
          ctx.stroke();
          // Snowflake icon
          ctx.fillStyle = '#93c5fd';
          ctx.font = 'bold 9px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('❄', x, y + 3);
        } else {
          ctx.fillStyle = getActivationColor(activation, 0.7);
          ctx.fill();
          ctx.strokeStyle = '#4b5563';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }

      // Overflow indicator
      if (neuronCount > SURGERY_MAX_DISPLAY_NEURONS) {
        const overY = startY + displayCount * SURGERY_NODE_SPACING;
        ctx.fillStyle = '#6b7280';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`+${neuronCount - SURGERY_MAX_DISPLAY_NEURONS}`, x, overY);
      }
    }

    nodesRef.current = nodes;
    setSurgeryCount({ frozen: frozenCount, killed: killedCount });
  }, [layers, hiddenLayers, canvasWidth, canvasHeight, layerSpacing, onGetNeuronStatus]);

  useEffect(() => {
    draw();
  }, [draw]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvasWidth / rect.width;
    const scaleY = canvasHeight / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top) * scaleY;

    // Find clicked node
    for (const node of nodesRef.current) {
      const dx = mx - node.x;
      const dy = my - node.y;
      if (dx * dx + dy * dy <= (SURGERY_NODE_RADIUS + 4) ** 2) {
        const currentStatus = onGetNeuronStatus(node.layerIdx, node.neuronIdx);
        const nextIdx = (STATUS_CYCLE.indexOf(currentStatus) + 1) % STATUS_CYCLE.length;
        onSetNeuronStatus(node.layerIdx, node.neuronIdx, STATUS_CYCLE[nextIdx]);
        onSurgeryChange();
        setTick(t => t + 1); // force re-draw
        return;
      }
    }
  }, [canvasWidth, canvasHeight, onGetNeuronStatus, onSetNeuronStatus, onSurgeryChange]);

  const handleRestoreAll = useCallback(() => {
    onClearAll();
    onSurgeryChange();
    setTick(t => t + 1);
  }, [onClearAll, onSurgeryChange]);

  const totalSurgeries = surgeryCount.frozen + surgeryCount.killed;
  const hasNetwork = layers && layers.length > 0;

  return (
    <div className="neuron-surgery" role="group" aria-label="Neuron surgery lab">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔬</span>
        <span>Neuron Surgery</span>
      </div>

      {!hasNetwork ? (
        <div className="surgery-empty">
          <p>Train a network first, then click neurons to freeze or kill them.</p>
        </div>
      ) : (
        <div className="surgery-content">
          <div className="surgery-instructions">
            Click neurons to cycle: <span className="surgery-active-tag">active</span> →{' '}
            <span className="surgery-frozen-tag">frozen ❄</span> →{' '}
            <span className="surgery-killed-tag">killed ✕</span>
          </div>

          <canvas
            ref={canvasRef}
            style={{ width: canvasWidth, height: canvasHeight, cursor: 'pointer' }}
            className="surgery-canvas"
            onClick={handleClick}
            role="img"
            aria-label={`Network surgery view — ${totalSurgeries} neurons modified`}
          />

          <div className="surgery-stats">
            {surgeryCount.frozen > 0 && (
              <span className="surgery-stat frozen">❄ {surgeryCount.frozen} frozen</span>
            )}
            {surgeryCount.killed > 0 && (
              <span className="surgery-stat killed">✕ {surgeryCount.killed} killed</span>
            )}
            {totalSurgeries === 0 && (
              <span className="surgery-stat none">All neurons active</span>
            )}
          </div>

          {currentPrediction && predictedLabel !== null && totalSurgeries > 0 && (
            <div className="surgery-impact">
              <div className="surgery-impact-header">
                <span>Impact on prediction</span>
              </div>
              <div className="surgery-confidence-bar">
                <div className="surgery-confidence-label">
                  Digit {predictedLabel}: {(currentPrediction[predictedLabel] * 100).toFixed(1)}%
                </div>
                <div className="surgery-bar-track">
                  <div
                    className="surgery-bar-fill"
                    style={{
                      width: `${currentPrediction[predictedLabel] * 100}%`,
                      backgroundColor: currentPrediction[predictedLabel] > 0.5
                        ? 'var(--accent-green, #10b981)'
                        : 'var(--accent-red, #ff6384)',
                    }}
                  />
                </div>
              </div>
            </div>
          )}

          {totalSurgeries > 0 && (
            <button
              className="btn btn-secondary surgery-restore-btn"
              onClick={handleRestoreAll}
            >
              🔄 Restore All Neurons
            </button>
          )}
        </div>
      )}
    </div>
  );
}

export default NeuronSurgery;
