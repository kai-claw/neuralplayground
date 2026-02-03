/**
 * MisfitGallery — the network's hardest digits.
 *
 * After training, shows which digits the network struggles with most.
 * Displays a gallery of misfits (highest loss samples) with the
 * network's prediction vs true label, confidence breakdown, and
 * visual indicators of confusion.
 *
 * Educational: even trained networks have blind spots. Understanding
 * failure is understanding the model.
 */

import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import type { NeuralNetwork } from '../nn';
import { findMisfits, computeMisfitSummary, generateTrainingData } from '../nn';
import type { Misfit, MisfitSummary } from '../nn';
import {
  MISFIT_DISPLAY_SIZE,
  MISFIT_GALLERY_COUNT,
  INPUT_DIM,
  DEFAULT_SAMPLES_PER_DIGIT,
} from '../constants';

/** Digit class colors — consistent across app */
const DIGIT_COLORS = [
  '#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff',
  '#ff9f40', '#10b981', '#f472b6', '#63deff', '#a78bfa',
] as const;

interface MisfitGalleryProps {
  networkRef: React.MutableRefObject<NeuralNetwork | null>;
  epoch: number;
  isTraining: boolean;
}

export default function MisfitGallery({
  networkRef,
  epoch,
  isTraining,
}: MisfitGalleryProps) {
  const [misfits, setMisfits] = useState<Misfit[]>([]);
  const [summary, setSummary] = useState<MisfitSummary | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [lastEpoch, setLastEpoch] = useState(0);
  const canvasRefs = useRef<Map<number, HTMLCanvasElement>>(new Map());
  const detailCanvasRef = useRef<HTMLCanvasElement>(null);

  // Training data for evaluation
  const trainingData = useMemo(
    () => generateTrainingData(DEFAULT_SAMPLES_PER_DIGIT),
    [],
  );

  // Recompute misfits when training pauses or epoch changes significantly
  useEffect(() => {
    const net = networkRef.current;
    if (!net || epoch === 0) return;
    if (isTraining && epoch - lastEpoch < 5) return; // throttle during training

    const m = findMisfits(net, trainingData.inputs, trainingData.labels, MISFIT_GALLERY_COUNT);
    const s = computeMisfitSummary(net, trainingData.inputs, trainingData.labels);
    setMisfits(m);
    setSummary(s);
    setLastEpoch(epoch);
    setSelectedIdx(null);
  }, [networkRef, epoch, isTraining, trainingData, lastEpoch]);

  // Render individual misfit thumbnails
  const renderThumbnail = useCallback(
    (canvas: HTMLCanvasElement, input: number[]) => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const size = MISFIT_DISPLAY_SIZE;
      const dpr = window.devicePixelRatio || 1;
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      ctx.scale(dpr, dpr);

      const cellSize = size / INPUT_DIM;
      for (let y = 0; y < INPUT_DIM; y++) {
        for (let x = 0; x < INPUT_DIM; x++) {
          const v = Math.round(input[y * INPUT_DIM + x] * 255);
          ctx.fillStyle = `rgb(${v}, ${v}, ${v})`;
          ctx.fillRect(x * cellSize, y * cellSize, cellSize + 0.5, cellSize + 0.5);
        }
      }
    },
    [],
  );

  // Render thumbnails when misfits change
  useEffect(() => {
    misfits.forEach((m, i) => {
      const canvas = canvasRefs.current.get(i);
      if (canvas) renderThumbnail(canvas, m.input);
    });
  }, [misfits, renderThumbnail]);

  // Render detail canvas for selected misfit
  useEffect(() => {
    if (selectedIdx === null || !misfits[selectedIdx]) return;
    const canvas = detailCanvasRef.current;
    if (!canvas) return;
    renderThumbnail(canvas, misfits[selectedIdx].input);
  }, [selectedIdx, misfits, renderThumbnail]);

  const selectedMisfit = selectedIdx !== null ? misfits[selectedIdx] : null;

  // Count misclassifications in gallery
  const wrongCount = useMemo(() => misfits.filter((m) => m.isWrong).length, [misfits]);

  if (epoch === 0) {
    return (
      <div className="misfit-gallery" role="group" aria-label="Misfit Gallery">
        <div className="panel-header">
          <span className="panel-icon" aria-hidden="true">🔍</span>
          <span>Misfit Gallery</span>
        </div>
        <p className="misfit-hint">Train the network to reveal its blind spots</p>
      </div>
    );
  }

  return (
    <div
      className="misfit-gallery"
      role="group"
      aria-label="Misfit Gallery — the network's hardest digits"
    >
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔍</span>
        <span>Misfit Gallery</span>
        {summary && (
          <span className="misfit-accuracy-badge">
            {Math.round(summary.accuracy * 100)}% acc
          </span>
        )}
      </div>

      {/* Summary stats */}
      {summary && (
        <div className="misfit-summary">
          <div className="misfit-stat">
            <span className="misfit-stat-value misfit-wrong-count">{summary.totalWrong}</span>
            <span className="misfit-stat-label">wrong</span>
          </div>
          <div className="misfit-stat">
            <span className="misfit-stat-value">{summary.totalSamples}</span>
            <span className="misfit-stat-label">total</span>
          </div>
          {summary.mostConfusedPair && (
            <div className="misfit-stat misfit-confused-pair">
              <span className="misfit-stat-value">
                <span style={{ color: DIGIT_COLORS[summary.mostConfusedPair[0]] }}>
                  {summary.mostConfusedPair[0]}
                </span>
                →
                <span style={{ color: DIGIT_COLORS[summary.mostConfusedPair[1]] }}>
                  {summary.mostConfusedPair[1]}
                </span>
              </span>
              <span className="misfit-stat-label">most confused</span>
            </div>
          )}
        </div>
      )}

      {/* Gallery grid */}
      <div className="misfit-grid" role="list" aria-label={`${misfits.length} hardest digits (${wrongCount} misclassified)`}>
        {misfits.map((m, i) => (
          <button
            key={i}
            className={`misfit-cell ${m.isWrong ? 'wrong' : 'hard'} ${selectedIdx === i ? 'selected' : ''}`}
            onClick={() => setSelectedIdx(selectedIdx === i ? null : i)}
            role="listitem"
            aria-label={`True: ${m.trueLabel}, Predicted: ${m.predictedLabel}, ${m.isWrong ? 'WRONG' : 'correct'}, confidence: ${Math.round(m.trueConfidence * 100)}%`}
          >
            <canvas
              ref={(el) => {
                if (el) canvasRefs.current.set(i, el);
                else canvasRefs.current.delete(i);
              }}
              style={{ width: MISFIT_DISPLAY_SIZE, height: MISFIT_DISPLAY_SIZE }}
              className="misfit-thumb"
            />
            <div className="misfit-labels">
              <span
                className="misfit-true-label"
                style={{ color: DIGIT_COLORS[m.trueLabel] }}
              >
                {m.trueLabel}
              </span>
              {m.isWrong && (
                <>
                  <span className="misfit-arrow" aria-hidden="true">→</span>
                  <span
                    className="misfit-pred-label"
                    style={{ color: DIGIT_COLORS[m.predictedLabel] }}
                  >
                    {m.predictedLabel}
                  </span>
                </>
              )}
            </div>
            <div
              className="misfit-loss-bar"
              style={{
                width: `${Math.min(100, Math.round(m.loss * 10))}%`,
                backgroundColor: m.isWrong ? '#ef4444' : '#f59e0b',
              }}
              aria-hidden="true"
            />
          </button>
        ))}
      </div>

      {/* Detail panel for selected misfit */}
      {selectedMisfit && (
        <div className="misfit-detail" role="region" aria-label={`Detail for misfit digit ${selectedMisfit.trueLabel}`}>
          <div className="misfit-detail-header">
            <canvas
              ref={detailCanvasRef}
              style={{ width: 80, height: 80 }}
              className="misfit-detail-canvas"
              role="img"
              aria-label={`Digit image — true label ${selectedMisfit.trueLabel}`}
            />
            <div className="misfit-detail-info">
              <div className="misfit-detail-verdict">
                {selectedMisfit.isWrong ? (
                  <span className="misfit-verdict-wrong">
                    ❌ True:{' '}
                    <span style={{ color: DIGIT_COLORS[selectedMisfit.trueLabel] }}>
                      {selectedMisfit.trueLabel}
                    </span>
                    , Predicted:{' '}
                    <span style={{ color: DIGIT_COLORS[selectedMisfit.predictedLabel] }}>
                      {selectedMisfit.predictedLabel}
                    </span>
                  </span>
                ) : (
                  <span className="misfit-verdict-hard">
                    ⚠️ Correct ({selectedMisfit.trueLabel}) but only{' '}
                    {Math.round(selectedMisfit.trueConfidence * 100)}% sure
                  </span>
                )}
              </div>
              <div className="misfit-detail-loss">
                Loss: {selectedMisfit.loss.toFixed(3)}
              </div>
            </div>
          </div>

          {/* Full confidence breakdown */}
          <div className="misfit-detail-bars" role="list" aria-label="Confidence per digit class">
            {selectedMisfit.probabilities.map((conf, i) => (
              <div
                key={i}
                className={`misfit-detail-bar-row ${
                  i === selectedMisfit.trueLabel ? 'true-class' : ''
                } ${i === selectedMisfit.predictedLabel ? 'pred-class' : ''}`}
                role="listitem"
              >
                <span className="misfit-bar-digit" style={{ color: DIGIT_COLORS[i] }}>
                  {i}
                  {i === selectedMisfit.trueLabel && (
                    <span className="misfit-bar-tag true" aria-label="true label">✓</span>
                  )}
                  {i === selectedMisfit.predictedLabel && i !== selectedMisfit.trueLabel && (
                    <span className="misfit-bar-tag pred" aria-label="predicted">✗</span>
                  )}
                </span>
                <div className="misfit-bar-bg">
                  <div
                    className="misfit-bar-fill"
                    style={{
                      width: `${Math.round(conf * 100)}%`,
                      backgroundColor:
                        i === selectedMisfit.trueLabel
                          ? '#10b981'
                          : i === selectedMisfit.predictedLabel
                          ? '#ef4444'
                          : DIGIT_COLORS[i],
                    }}
                  />
                </div>
                <span className="misfit-bar-pct">{Math.round(conf * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
