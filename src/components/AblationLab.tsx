import { useRef, useEffect, useState, useCallback } from 'react';
import type { NeuralNetwork } from '../nn';
import type { AblationStudy } from '../nn/ablation';
import { runAblationStudy, importanceToColor } from '../nn/ablation';
import {
  ABLATION_CELL_SIZE,
  ABLATION_CELL_GAP,
  ABLATION_MAX_NEURONS_PER_LAYER,
  ABLATION_SAMPLES_PER_DIGIT,
} from '../constants';

interface AblationLabProps {
  networkRef: React.RefObject<NeuralNetwork | null>;
  epoch: number;
  isTraining: boolean;
}

/**
 * Ablation Lab — systematic neuron knockout study.
 *
 * Kills each neuron one at a time, measures accuracy impact,
 * and displays results as a visual importance heatmap.
 * Hot cells = critical neurons. Cold cells = redundant.
 */
export default function AblationLab({ networkRef, epoch, isTraining }: AblationLabProps) {
  const [study, setStudy] = useState<AblationStudy | null>(null);
  const [running, setRunning] = useState(false);
  const [hoveredNeuron, setHoveredNeuron] = useState<{ layer: number; neuron: number } | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const runStudy = useCallback(() => {
    if (!networkRef.current || epoch === 0 || isTraining) return;
    setRunning(true);

    // Use requestAnimationFrame to let UI update before blocking computation
    requestAnimationFrame(() => {
      const result = runAblationStudy(networkRef.current!, ABLATION_SAMPLES_PER_DIGIT);
      setStudy(result);
      setRunning(false);
    });
  }, [networkRef, epoch, isTraining]);

  // Render the heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !study) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const cell = ABLATION_CELL_SIZE;
    const gap = ABLATION_CELL_GAP;
    const layerCount = study.layers.length;

    // Calculate canvas dimensions
    let maxNeurons = 0;
    for (const layer of study.layers) {
      const n = Math.min(layer.length, ABLATION_MAX_NEURONS_PER_LAYER);
      if (n > maxNeurons) maxNeurons = n;
    }

    const w = maxNeurons * (cell + gap) + gap + 60; // 60 for layer label
    const h = layerCount * (cell + gap) + gap + 24; // 24 for column labels

    canvas.width = w;
    canvas.height = h;
    ctx.fillStyle = '#0a0a1a';
    ctx.fillRect(0, 0, w, h);

    // Layer labels
    ctx.font = '10px monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let l = 0; l < layerCount; l++) {
      const y = l * (cell + gap) + gap + cell / 2;
      ctx.fillStyle = '#8899aa';
      ctx.fillText(`L${l + 1}`, 54, y);

      const neurons = study.layers[l];
      const displayCount = Math.min(neurons.length, ABLATION_MAX_NEURONS_PER_LAYER);

      for (let n = 0; n < displayCount; n++) {
        const x = 60 + n * (cell + gap) + gap;
        const result = neurons[n];
        const color = importanceToColor(result.importance);

        ctx.fillStyle = color;
        ctx.fillRect(x, l * (cell + gap) + gap, cell, cell);

        // Highlight on hover
        if (hoveredNeuron && hoveredNeuron.layer === l && hoveredNeuron.neuron === n) {
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.strokeRect(x - 1, l * (cell + gap) + gap - 1, cell + 2, cell + 2);
        }
      }
    }

    // Color scale legend at bottom
    const legendY = layerCount * (cell + gap) + gap + 6;
    const legendW = 120;
    const legendX = 60;
    const grad = ctx.createLinearGradient(legendX, 0, legendX + legendW, 0);
    grad.addColorStop(0, 'rgba(40, 80, 160, 0.8)');
    grad.addColorStop(0.33, 'rgba(60, 180, 130, 0.8)');
    grad.addColorStop(0.66, 'rgba(255, 170, 30, 0.9)');
    grad.addColorStop(1, 'rgba(255, 70, 130, 1)');
    ctx.fillStyle = grad;
    ctx.fillRect(legendX, legendY, legendW, 8);
    ctx.fillStyle = '#667788';
    ctx.font = '8px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('redundant', legendX, legendY + 16);
    ctx.textAlign = 'right';
    ctx.fillText('critical', legendX + legendW, legendY + 16);
  }, [study, hoveredNeuron]);

  // Handle hover on canvas
  const handleCanvasMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!study) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const cell = ABLATION_CELL_SIZE;
    const gap = ABLATION_CELL_GAP;

    const layerIdx = Math.floor((y - gap) / (cell + gap));
    const neuronIdx = Math.floor((x - 60 - gap) / (cell + gap));

    if (
      layerIdx >= 0 && layerIdx < study.layers.length &&
      neuronIdx >= 0 && neuronIdx < study.layers[layerIdx].length
    ) {
      setHoveredNeuron({ layer: layerIdx, neuron: neuronIdx });
    } else {
      setHoveredNeuron(null);
    }
  }, [study]);

  const handleCanvasLeave = useCallback(() => setHoveredNeuron(null), []);

  // Get tooltip data for hovered neuron
  const tooltipData = hoveredNeuron && study
    ? study.layers[hoveredNeuron.layer]?.[hoveredNeuron.neuron]
    : null;

  return (
    <div className="ablation-lab panel">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔬</span>
        <span>Ablation Lab</span>
      </div>

      {!study && !running && (
        <div className="ablation-intro">
          <p className="ablation-desc">
            Systematically kill each neuron and measure accuracy impact.
            Reveals which neurons are critical vs redundant.
          </p>
          <button
            className="ablation-run-btn"
            onClick={runStudy}
            disabled={epoch === 0 || isTraining}
            aria-label="Run ablation study"
          >
            {epoch === 0 ? 'Train first' : isTraining ? 'Pause training first' : '🧪 Run Study'}
          </button>
        </div>
      )}

      {running && (
        <div className="ablation-running">
          <span className="ablation-spinner" aria-hidden="true">⏳</span>
          <span>Testing neurons…</span>
        </div>
      )}

      {study && !running && (
        <>
          {/* Summary stats */}
          <div className="ablation-stats">
            <span className="ablation-stat">
              Baseline: <strong>{(study.baselineAccuracy * 100).toFixed(1)}%</strong>
            </span>
            <span className="ablation-stat">
              Neurons: <strong>{study.totalNeurons}</strong>
            </span>
          </div>

          {/* Critical/redundant callouts */}
          <div className="ablation-callouts">
            {study.mostCritical && (
              <div className="ablation-callout critical">
                <span className="callout-icon" aria-hidden="true">🔥</span>
                <span>
                  Most critical: L{study.mostCritical.layerIdx + 1}:N{study.mostCritical.neuronIdx}
                  <small> (−{(study.mostCritical.accuracyDrop * 100).toFixed(1)}%)</small>
                </span>
              </div>
            )}
            {study.mostRedundant && (
              <div className="ablation-callout redundant">
                <span className="callout-icon" aria-hidden="true">❄️</span>
                <span>
                  Most redundant: L{study.mostRedundant.layerIdx + 1}:N{study.mostRedundant.neuronIdx}
                  <small> ({study.mostRedundant.accuracyDrop >= 0 ? '−' : '+'}{Math.abs(study.mostRedundant.accuracyDrop * 100).toFixed(1)}%)</small>
                </span>
              </div>
            )}
          </div>

          {/* Heatmap canvas */}
          <div className="ablation-heatmap-container">
            <canvas
              ref={canvasRef}
              className="ablation-heatmap"
              onMouseMove={handleCanvasMove}
              onMouseLeave={handleCanvasLeave}
              role="img"
              aria-label="Neuron importance heatmap"
            />

            {/* Tooltip */}
            {tooltipData && (
              <div className="ablation-tooltip" aria-hidden="true">
                <strong>L{tooltipData.layerIdx + 1} : N{tooltipData.neuronIdx}</strong>
                <span>Accuracy without: {(tooltipData.accuracyWithout * 100).toFixed(1)}%</span>
                <span>Drop: {tooltipData.accuracyDrop >= 0 ? '−' : '+'}{Math.abs(tooltipData.accuracyDrop * 100).toFixed(1)}%</span>
                <span>Importance: {(tooltipData.importance * 100).toFixed(0)}%</span>
              </div>
            )}
          </div>

          {/* Re-run button */}
          <button
            className="ablation-rerun-btn"
            onClick={runStudy}
            disabled={isTraining}
            aria-label="Re-run ablation study"
          >
            ↻ Re-run
          </button>
        </>
      )}
    </div>
  );
}
