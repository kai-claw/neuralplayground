/**
 * Confusion Matrix — Interactive 10×10 heatmap of actual vs predicted digits.
 *
 * Shows classification accuracy per digit pair with hover details,
 * precision/recall per class, and F1 scores. A fundamental ML
 * education tool — instantly reveals which digits the network confuses.
 */

import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { computeConfusionMatrix } from '../nn/confusion';
import type { ConfusionData } from '../nn/confusion';
import type { NeuralNetwork } from '../nn';
import {
  drawConfusionMatrix,
  hitTestConfusion,
  CONFUSION_CANVAS_SIZE,
} from '../renderers/confusionRenderer';
import { CONFUSION_SAMPLES_PER_DIGIT } from '../constants';

interface Props {
  networkRef: React.RefObject<NeuralNetwork | null>;
  epoch: number;
  isTraining: boolean;
}

export function ConfusionMatrix({ networkRef, epoch, isTraining }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoverCell, setHoverCell] = useState<{ row: number; col: number } | null>(null);
  const [confusionData, setConfusionData] = useState<ConfusionData | null>(null);

  // Recompute confusion matrix when epoch changes (throttled: every 5 epochs or on stop)
  useEffect(() => {
    if (!networkRef.current || epoch === 0) {
      setConfusionData(null);
      return;
    }
    // During training, update every 5 epochs. When stopped, always update.
    if (isTraining && epoch % 5 !== 0) return;

    const data = computeConfusionMatrix(networkRef.current, CONFUSION_SAMPLES_PER_DIGIT);
    setConfusionData(data);
  }, [epoch, isTraining, networkRef]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !confusionData) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = CONFUSION_CANVAS_SIZE * dpr;
    canvas.height = CONFUSION_CANVAS_SIZE * dpr;
    ctx.scale(dpr, dpr);

    drawConfusionMatrix(ctx, confusionData, hoverCell);
  }, [confusionData, hoverCell]);

  // Mouse interaction
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = CONFUSION_CANVAS_SIZE / rect.width;
    const scaleY = CONFUSION_CANVAS_SIZE / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setHoverCell(hitTestConfusion(x, y));
  }, []);

  const handleMouseLeave = useCallback(() => {
    setHoverCell(null);
  }, []);

  // Tooltip info
  const tooltipInfo = useMemo(() => {
    if (!hoverCell || !confusionData) return null;
    const { row, col } = hoverCell;
    const count = confusionData.matrix[row][col];
    const total = confusionData.classCounts[row];
    const pct = total > 0 ? ((count / total) * 100).toFixed(1) : '0.0';
    const isDiag = row === col;
    return {
      actual: row,
      predicted: col,
      count,
      pct,
      isDiag,
      precision: (confusionData.precision[col] * 100).toFixed(1),
      recall: (confusionData.recall[row] * 100).toFixed(1),
    };
  }, [hoverCell, confusionData]);

  // Top confused pairs
  const topConfusions = useMemo(() => {
    if (!confusionData) return [];
    const pairs: { actual: number; predicted: number; count: number }[] = [];
    for (let r = 0; r < 10; r++) {
      for (let c = 0; c < 10; c++) {
        if (r !== c && confusionData.matrix[r][c] > 0) {
          pairs.push({ actual: r, predicted: c, count: confusionData.matrix[r][c] });
        }
      }
    }
    pairs.sort((a, b) => b.count - a.count);
    return pairs.slice(0, 3);
  }, [confusionData]);

  return (
    <div className="confusion-matrix" role="group" aria-label="Confusion matrix">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🎯</span>
        <span>Confusion Matrix</span>
        {confusionData && (
          <span className="confusion-accuracy">
            {(confusionData.accuracy * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="confusion-content">
        {confusionData ? (
          <>
            <canvas
              ref={canvasRef}
              style={{ width: CONFUSION_CANVAS_SIZE, height: CONFUSION_CANVAS_SIZE }}
              className="confusion-canvas"
              onMouseMove={handleMouseMove}
              onMouseLeave={handleMouseLeave}
              role="img"
              aria-label={`Confusion matrix: ${(confusionData.accuracy * 100).toFixed(1)}% overall accuracy`}
            />

            {/* Hover tooltip */}
            {tooltipInfo && (
              <div className={`confusion-tooltip ${tooltipInfo.isDiag ? 'correct' : 'error'}`}>
                <div className="confusion-tooltip-title">
                  {tooltipInfo.isDiag
                    ? `✅ Digit ${tooltipInfo.actual} — correct`
                    : `❌ Digit ${tooltipInfo.actual} → predicted ${tooltipInfo.predicted}`}
                </div>
                <div className="confusion-tooltip-stats">
                  <span>{tooltipInfo.count} samples ({tooltipInfo.pct}%)</span>
                  <span className="confusion-tooltip-metrics">
                    P: {tooltipInfo.precision}% · R: {tooltipInfo.recall}%
                  </span>
                </div>
              </div>
            )}

            {/* Top confused pairs */}
            {topConfusions.length > 0 && (
              <div className="confusion-top-errors">
                <div className="confusion-top-errors-title">Top confusions:</div>
                {topConfusions.map((p, i) => (
                  <span key={i} className="confusion-error-chip">
                    {p.actual}→{p.predicted} ×{p.count}
                  </span>
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="confusion-empty">
            Train the network to see classification patterns…
          </div>
        )}
      </div>
    </div>
  );
}

export default ConfusionMatrix;
