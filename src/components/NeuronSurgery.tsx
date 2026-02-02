import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import type { LayerState, NeuronStatus } from '../types';
import {
  computeSurgeryLayout,
  drawSurgeryCanvas,
  hitTestSurgeryNode,
} from '../renderers/surgeryRenderer';
import type { SurgeryNode } from '../renderers/surgeryRenderer';

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
  const nodesRef = useRef<SurgeryNode[]>([]);
  const [surgeryCount, setSurgeryCount] = useState({ frozen: 0, killed: 0 });
  const [, setTick] = useState(0); // force re-renders

  // Memoize hidden layers to avoid recreating array every render (stabilizes draw deps)
  const hiddenLayers = useMemo(() => layers ? layers.slice(0, -1) : [], [layers]);
  const layout = useMemo(() => computeSurgeryLayout(hiddenLayers), [hiddenLayers]);
  const { canvasWidth, canvasHeight } = layout;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !layers || layers.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvasWidth * dpr;
    canvas.height = canvasHeight * dpr;
    ctx.scale(dpr, dpr);

    const { nodes, counts } = drawSurgeryCanvas(ctx, hiddenLayers, layout, onGetNeuronStatus);
    nodesRef.current = nodes;
    setSurgeryCount(counts);
  }, [layers, hiddenLayers, canvasWidth, canvasHeight, layout, onGetNeuronStatus]);

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

    const hit = hitTestSurgeryNode(mx, my, nodesRef.current);
    if (hit) {
      const currentStatus = onGetNeuronStatus(hit.layerIdx, hit.neuronIdx);
      const nextIdx = (STATUS_CYCLE.indexOf(currentStatus) + 1) % STATUS_CYCLE.length;
      onSetNeuronStatus(hit.layerIdx, hit.neuronIdx, STATUS_CYCLE[nextIdx]);
      onSurgeryChange();
      setTick(t => t + 1);
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
