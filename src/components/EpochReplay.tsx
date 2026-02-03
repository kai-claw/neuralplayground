/**
 * EpochReplay — Training Time Machine.
 *
 * Records network snapshots during training, then lets users
 * scrub through the timeline to see how the network evolved.
 *
 * When a drawing is active, shows how prediction confidence
 * for that digit changed over training epochs — a beautiful
 * animated confidence curve overlaid on the timeline.
 */

import { useRef, useEffect, useCallback, useState, useMemo } from 'react';
import { useContainerDims } from '../hooks/useContainerDims';
import {
  EpochRecorder,
  replayForward,
} from '../nn/epochReplay';
import type { EpochSnapshot } from '../nn/epochReplay';
import type { TrainingSnapshot } from '../types';
import {
  EPOCH_REPLAY_DISPLAY,
  EPOCH_REPLAY_ASPECT,
} from '../constants';

/** Digit class colors — same palette as ActivationSpace */
const DIGIT_COLORS = [
  '#ff6384', '#36a2eb', '#ffce56', '#4bc0c0', '#9966ff',
  '#ff9f40', '#10b981', '#f472b6', '#63deff', '#a78bfa',
] as const;

interface EpochReplayProps {
  /** Current training epoch */
  epoch: number;
  /** Whether training is active */
  isTraining: boolean;
  /** Latest training snapshot (for recording) */
  snapshot: TrainingSnapshot | null;
  /** Current drawn digit as pixel array */
  currentInput: number[] | null;
  /** Active activation function */
  activationFn: 'relu' | 'sigmoid' | 'tanh';
}

export default function EpochReplay({
  epoch,
  isTraining,
  snapshot,
  currentInput,
  activationFn,
}: EpochReplayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const recorderRef = useRef(new EpochRecorder(200));
  const [scrubIndex, setScrubIndex] = useState<number | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const playTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { containerRef, dims } = useContainerDims({
    defaultWidth: EPOCH_REPLAY_DISPLAY.width,
    defaultHeight: EPOCH_REPLAY_DISPLAY.height,
    aspectRatio: EPOCH_REPLAY_ASPECT,
  });

  const { width, height } = dims;

  // Record snapshots during training
  useEffect(() => {
    if (snapshot && snapshot.epoch > 0) {
      recorderRef.current.record(snapshot);
    }
  }, [snapshot]);

  // Reset recorder when epoch goes to 0 (network reset)
  useEffect(() => {
    if (epoch === 0) {
      recorderRef.current.clear();
      setScrubIndex(null);
      setIsPlaying(false);
    }
  }, [epoch]);

  const timeline = recorderRef.current.getTimeline();
  const totalSnapshots = timeline.length;

  // Compute replay predictions for current input across all epochs
  const replayPredictions = useMemo(() => {
    if (!currentInput || totalSnapshots === 0) return null;

    return timeline.map(snap => {
      try {
        return replayForward(snap.params, currentInput, activationFn);
      } catch {
        return { probabilities: new Array(10).fill(0.1), label: 0 };
      }
    });
  }, [currentInput, totalSnapshots, timeline, activationFn]);

  // Active snapshot (scrubbed or latest)
  const activeSnapshot: EpochSnapshot | null = useMemo(() => {
    if (scrubIndex !== null && scrubIndex < totalSnapshots) {
      return timeline[scrubIndex];
    }
    return totalSnapshots > 0 ? timeline[totalSnapshots - 1] : null;
  }, [scrubIndex, totalSnapshots, timeline]);

  // Active prediction from scrub position
  const activePrediction = useMemo(() => {
    if (!replayPredictions || replayPredictions.length === 0) return null;
    const idx = scrubIndex !== null ? scrubIndex : replayPredictions.length - 1;
    return idx < replayPredictions.length ? replayPredictions[idx] : null;
  }, [replayPredictions, scrubIndex]);

  // Playback animation
  useEffect(() => {
    if (!isPlaying || totalSnapshots < 2) {
      if (playTimerRef.current) clearTimeout(playTimerRef.current);
      return;
    }

    const startIdx = scrubIndex ?? 0;
    let currentIdx = startIdx;

    const tick = () => {
      currentIdx++;
      if (currentIdx >= totalSnapshots) {
        setIsPlaying(false);
        setScrubIndex(totalSnapshots - 1);
        return;
      }
      setScrubIndex(currentIdx);
      playTimerRef.current = setTimeout(tick, 80);
    };

    playTimerRef.current = setTimeout(tick, 80);
    return () => {
      if (playTimerRef.current) clearTimeout(playTimerRef.current);
    };
  }, [isPlaying, totalSnapshots, scrubIndex]);

  // Render the timeline visualization
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    if (totalSnapshots === 0) {
      ctx.fillStyle = '#6b7280';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Train to start recording epochs', width / 2, height / 2);
      return;
    }

    const pad = { top: 8, right: 16, bottom: 32, left: 40 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    // Draw loss/accuracy curves (background context)
    const maxLoss = Math.max(3, ...timeline.map(s => s.loss));

    // Loss curve (dim red)
    ctx.strokeStyle = 'rgba(255, 99, 132, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < totalSnapshots; i++) {
      const x = pad.left + (i / (totalSnapshots - 1)) * plotW;
      const y = pad.top + (1 - timeline[i].loss / maxLoss) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Accuracy curve (dim green)
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < totalSnapshots; i++) {
      const x = pad.left + (i / (totalSnapshots - 1)) * plotW;
      const y = pad.top + (1 - timeline[i].accuracy) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw prediction confidence curves if we have input
    if (replayPredictions && replayPredictions.length > 0) {
      // Find the predicted label at the scrub position
      const activeIdx = scrubIndex !== null ? Math.min(scrubIndex, replayPredictions.length - 1) : replayPredictions.length - 1;
      const currentLabel = replayPredictions[activeIdx]?.label ?? 0;

      // Draw confidence for each digit class, highlighting the predicted one
      for (let digit = 0; digit < 10; digit++) {
        const isActive = digit === currentLabel;

        // Fill area under the active curve
        if (isActive) {
          ctx.fillStyle = DIGIT_COLORS[digit] + '18';
          ctx.beginPath();
          ctx.moveTo(pad.left, pad.top + plotH);
          for (let i = 0; i < replayPredictions.length; i++) {
            const x = pad.left + (i / (replayPredictions.length - 1)) * plotW;
            const conf = replayPredictions[i].probabilities[digit] || 0;
            const y = pad.top + (1 - conf) * plotH;
            ctx.lineTo(x, y);
          }
          ctx.lineTo(pad.left + plotW, pad.top + plotH);
          ctx.closePath();
          ctx.fill();
        }

        ctx.strokeStyle = isActive
          ? DIGIT_COLORS[digit]
          : DIGIT_COLORS[digit] + '30';
        ctx.lineWidth = isActive ? 2.5 : 0.8;
        ctx.beginPath();
        for (let i = 0; i < replayPredictions.length; i++) {
          const x = pad.left + (i / (replayPredictions.length - 1)) * plotW;
          const conf = replayPredictions[i].probabilities[digit] || 0;
          const y = pad.top + (1 - conf) * plotH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
    }

    // Scrub position indicator
    const scrubPos = scrubIndex !== null ? scrubIndex : totalSnapshots - 1;
    const scrubX = pad.left + (scrubPos / Math.max(1, totalSnapshots - 1)) * plotW;

    // Vertical scrub line
    ctx.strokeStyle = 'rgba(99, 222, 255, 0.6)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(scrubX, pad.top);
    ctx.lineTo(scrubX, pad.top + plotH);
    ctx.stroke();
    ctx.setLineDash([]);

    // Scrub handle
    ctx.fillStyle = '#63deff';
    ctx.beginPath();
    ctx.arc(scrubX, pad.top + plotH + 8, 5, 0, Math.PI * 2);
    ctx.fill();

    // Scrub label
    if (activeSnapshot) {
      ctx.fillStyle = '#63deff';
      ctx.font = 'bold 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Epoch ${activeSnapshot.epoch}`, scrubX, pad.top + plotH + 24);
    }

    // Y-axis labels
    ctx.fillStyle = '#6b7280';
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText('1.0', pad.left - 4, pad.top + 4);
    ctx.fillText('0.5', pad.left - 4, pad.top + plotH / 2 + 3);
    ctx.fillText('0.0', pad.left - 4, pad.top + plotH + 4);

    // Epoch axis
    ctx.textAlign = 'center';
    ctx.fillText('1', pad.left, pad.top + plotH + 24);
    if (totalSnapshots > 1) {
      ctx.fillText(String(timeline[totalSnapshots - 1].epoch), pad.left + plotW, pad.top + plotH + 24);
    }
  }, [width, height, totalSnapshots, timeline, replayPredictions, scrubIndex, activeSnapshot]);

  useEffect(() => {
    render();
  }, [render]);

  // Scrub handler
  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const idx = parseInt(e.target.value, 10);
    setScrubIndex(idx);
    setIsPlaying(false);
  }, []);

  const handlePlayToggle = useCallback(() => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      // Start from beginning if at end
      if (scrubIndex !== null && scrubIndex >= totalSnapshots - 1) {
        setScrubIndex(0);
      } else if (scrubIndex === null) {
        setScrubIndex(0);
      }
      setIsPlaying(true);
    }
  }, [isPlaying, scrubIndex, totalSnapshots]);

  const handleRewind = useCallback(() => {
    setScrubIndex(0);
    setIsPlaying(false);
  }, []);

  return (
    <div
      className="epoch-replay"
      ref={containerRef}
      role="group"
      aria-label="Epoch replay — training time machine"
    >
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">⏪</span>
        <span>Training Time Machine</span>
        {activeSnapshot && (
          <span className="epoch-replay-badge">
            Epoch {activeSnapshot.epoch} · Loss {activeSnapshot.loss.toFixed(2)} · Acc {(activeSnapshot.accuracy * 100).toFixed(0)}%
          </span>
        )}
      </div>

      <canvas
        ref={canvasRef}
        style={{ width, height }}
        className="epoch-replay-canvas"
        role="img"
        aria-label={
          totalSnapshots > 0
            ? `Training timeline showing ${totalSnapshots} epochs${currentInput ? ' with prediction confidence curves' : ''}`
            : 'Training timeline — train to start recording'
        }
      />

      {totalSnapshots > 1 && (
        <div className="epoch-replay-controls">
          <button
            className="replay-btn"
            onClick={handleRewind}
            aria-label="Rewind to first epoch"
            title="Rewind"
          >
            ⏮
          </button>
          <button
            className={`replay-btn ${isPlaying ? 'playing' : ''}`}
            onClick={handlePlayToggle}
            aria-label={isPlaying ? 'Pause replay' : 'Play replay'}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '⏸' : '▶'}
          </button>
          <input
            type="range"
            min={0}
            max={totalSnapshots - 1}
            value={scrubIndex ?? totalSnapshots - 1}
            onChange={handleScrub}
            className="slider replay-slider"
            aria-label="Scrub through training epochs"
            aria-valuemin={0}
            aria-valuemax={totalSnapshots - 1}
            aria-valuenow={scrubIndex ?? totalSnapshots - 1}
          />
        </div>
      )}

      {/* Active prediction readout */}
      {activePrediction && currentInput && (
        <div className="epoch-replay-prediction">
          {activePrediction.probabilities.map((prob, digit) => (
            <div
              key={digit}
              className={`replay-digit ${digit === activePrediction.label ? 'active' : ''}`}
              style={{
                '--digit-color': DIGIT_COLORS[digit],
                '--digit-conf': prob,
              } as React.CSSProperties}
            >
              <span className="replay-digit-label">{digit}</span>
              <div className="replay-digit-bar">
                <div
                  className="replay-digit-fill"
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {totalSnapshots === 0 && (
        <p className="epoch-replay-hint">
          {isTraining ? 'Recording epochs…' : 'Draw a digit, then train — scrub through epochs to see confidence evolve.'}
        </p>
      )}
    </div>
  );
}
