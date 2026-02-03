/**
 * Gradient Flow Monitor — Real-time visualization of gradient magnitudes
 * per layer during training.
 *
 * Shows whether gradients are healthy, vanishing (dying neurons),
 * or exploding (unstable training). Fundamental for understanding
 * why networks succeed or fail — the "blood pressure monitor" of
 * neural network training.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import type { NeuralNetwork } from '../nn';
import {
  measureGradientFlow,
  GradientFlowHistory,
  type GradientFlowSnapshot,
} from '../nn/gradientFlow';
import {
  drawGradientFlow,
  getGradientFlowHeight,
  GRADIENT_FLOW_WIDTH,
} from '../renderers/gradientFlowRenderer';
import { generateTrainingData } from '../nn/sampleData';
import { GRADIENT_FLOW_SAMPLE_COUNT } from '../constants';

interface Props {
  networkRef: React.RefObject<NeuralNetwork | null>;
  epoch: number;
  layers: { weights: number[][] }[] | null;
  isTraining?: boolean;
}

export function GradientFlowMonitor({ networkRef, epoch, layers, isTraining }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const historyRef = useRef(new GradientFlowHistory(100));
  const [snapshot, setSnapshot] = useState<GradientFlowSnapshot | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Sample data for gradient measurement (stable across renders)
  const sampleRef = useRef<{ inputs: number[][]; labels: number[] } | null>(null);
  if (!sampleRef.current) {
    sampleRef.current = generateTrainingData(GRADIENT_FLOW_SAMPLE_COUNT);
  }

  // Measure gradient flow when epoch changes.
  // Throttled to every 5 epochs during training to reduce CPU load
  // (each measurement runs a full forward+backward pass).
  useEffect(() => {
    if (!networkRef.current || epoch === 0) {
      setSnapshot(null);
      return;
    }
    // During active training, sample every 5 epochs instead of every epoch
    if (isTraining && epoch % 5 !== 0) return;

    // Use first sample for gradient measurement
    const sample = sampleRef.current!;
    const input = sample.inputs[0];
    const label = sample.labels[0];

    const snap = measureGradientFlow(networkRef.current, input, label);
    historyRef.current.push(snap);
    setSnapshot(snap);
  }, [epoch, isTraining, networkRef]);

  // Clear history on reset
  useEffect(() => {
    if (epoch === 0) {
      historyRef.current.clear();
    }
  }, [epoch]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const numLayers = snapshot?.layers.length || (layers ? layers.length : 3);
    const h = getGradientFlowHeight(numLayers);
    const w = GRADIENT_FLOW_WIDTH;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.scale(dpr, dpr);

    drawGradientFlow(ctx, w, h, snapshot, historyRef.current.getAll());
  }, [snapshot, layers]);

  const toggleDetails = useCallback(() => {
    setShowDetails(prev => !prev);
  }, []);

  const numLayers = snapshot?.layers.length || (layers ? layers.length : 3);
  const canvasHeight = getGradientFlowHeight(numLayers);

  return (
    <div className="gradient-flow-monitor" role="group" aria-label="Gradient flow monitor">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">📊</span>
        <span>Gradient Flow</span>
        {snapshot && (
          <span className={`gradient-health-badge gradient-health-${snapshot.health}`}>
            {snapshot.health === 'healthy' ? '✅' : snapshot.health === 'vanishing' ? '❄️' : '🔥'}
          </span>
        )}
      </div>

      <div className="gradient-flow-content">
        <canvas
          ref={canvasRef}
          style={{ width: GRADIENT_FLOW_WIDTH, height: canvasHeight }}
          className="gradient-flow-canvas"
          role="img"
          aria-label={snapshot
            ? `Gradient flow: ${snapshot.health} — ${snapshot.layers.length} layers`
            : 'Gradient flow — not started'}
        />

        {/* Layer detail toggle */}
        {snapshot && (
          <button
            className="gradient-detail-toggle"
            onClick={toggleDetails}
            aria-expanded={showDetails}
          >
            {showDetails ? '▾ Hide details' : '▸ Layer details'}
          </button>
        )}

        {/* Detailed per-layer stats */}
        {showDetails && snapshot && (
          <div className="gradient-detail-table">
            {snapshot.layers.map((l, i) => {
              const isOutput = i === snapshot.layers.length - 1;
              const label = isOutput ? 'Output' : `Hidden ${i + 1}`;
              return (
                <div key={i} className="gradient-detail-row">
                  <span className="gradient-detail-label">{label}</span>
                  <span className="gradient-detail-stat">
                    μ={l.meanAbsGrad < 0.01 ? l.meanAbsGrad.toExponential(1) : l.meanAbsGrad.toFixed(4)}
                  </span>
                  <span className="gradient-detail-stat">
                    max={l.maxAbsGrad < 0.01 ? l.maxAbsGrad.toExponential(1) : l.maxAbsGrad.toFixed(4)}
                  </span>
                  <span className={`gradient-detail-dead ${l.deadFraction > 0.5 ? 'warning' : ''}`}>
                    dead={Math.round(l.deadFraction * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default GradientFlowMonitor;
