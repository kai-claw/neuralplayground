/**
 * useCinematic — Cinematic demo mode state machine.
 *
 * Extracted from App.tsx to reduce its complexity from ~500 to ~300 lines.
 * Manages the three cinematic phases: training → drawing → predicting,
 * with auto-cycling through all 10 digits.
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { DIGIT_STROKES, getDigitDrawDuration } from '../data/digitStrokes';
import type { TrainingConfig, CinematicPhase } from '../types';
import {
  CINEMATIC_TRAIN_EPOCHS,
  CINEMATIC_PREDICT_DWELL,
  CINEMATIC_EPOCH_INTERVAL,
} from '../constants';

interface DrawingCanvasHandle {
  clear: () => void;
  drawDot: (x: number, y: number) => void;
  drawStroke: (x1: number, y1: number, x2: number, y2: number) => void;
  getImageData: () => ImageData | null;
}

interface UseCinematicOptions {
  config: TrainingConfig;
  initNetwork: (config?: TrainingConfig) => void;
  startTraining: () => void;
  stopTraining: () => void;
  drawingCanvasRef: React.RefObject<DrawingCanvasHandle | null>;
  onSignalFlow: () => void;
}

export interface CinematicState {
  active: boolean;
  phase: CinematicPhase;
  digit: number;
  progress: number;
  epoch: number;
}

export function useCinematic({
  config,
  initNetwork,
  startTraining,
  stopTraining,
  drawingCanvasRef,
  onSignalFlow,
}: UseCinematicOptions) {
  const [state, setState] = useState<CinematicState>({
    active: false,
    phase: 'training',
    digit: 0,
    progress: 0,
    epoch: 0,
  });

  const activeRef = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    clearTimer();
    setState(prev => ({ ...prev, active: false }));
  }, [clearTimer]);

  const drawDigit = useCallback((digit: number) => {
    if (!activeRef.current) return;

    setState(prev => ({ ...prev, phase: 'drawing', digit, progress: 0 }));

    const handle = drawingCanvasRef.current;
    if (!handle) {
      timerRef.current = setTimeout(() => drawDigit((digit + 1) % 10), 1000);
      return;
    }

    handle.clear();
    const strokes = DIGIT_STROKES[digit];
    let totalPoints = 0;
    for (const s of strokes) totalPoints += s.points.length;
    const duration = getDigitDrawDuration(digit);
    const pointDelay = duration / totalPoints;

    let strokeIdx = 0;
    let pointIdx = 0;
    let pointsDrawn = 0;

    const drawStep = () => {
      if (!activeRef.current) return;
      if (strokeIdx >= strokes.length) {
        // Drawing done — show prediction
        setState(prev => ({ ...prev, phase: 'predicting', progress: 1 }));
        onSignalFlow();

        timerRef.current = setTimeout(() => {
          if (!activeRef.current) return;
          drawDigit((digit + 1) % 10);
        }, CINEMATIC_PREDICT_DWELL);
        return;
      }

      const stroke = strokes[strokeIdx];
      const pt = stroke.points[pointIdx];

      if (pointIdx === 0) {
        handle.drawDot(pt.x, pt.y);
      } else {
        const prev = stroke.points[pointIdx - 1];
        handle.drawStroke(prev.x, prev.y, pt.x, pt.y);
      }

      pointsDrawn++;
      setState(prev => ({ ...prev, progress: pointsDrawn / totalPoints }));
      pointIdx++;

      if (pointIdx >= stroke.points.length) {
        strokeIdx++;
        pointIdx = 0;
      }

      timerRef.current = setTimeout(drawStep, pointDelay);
    };

    timerRef.current = setTimeout(drawStep, 200);
  }, [drawingCanvasRef, onSignalFlow]);

  const start = useCallback(() => {
    if (activeRef.current) {
      stop();
      return;
    }

    activeRef.current = true;
    setState({
      active: true,
      phase: 'training',
      digit: 0,
      progress: 0,
      epoch: 0,
    });

    // Stop any existing training and reinitialize
    stopTraining();
    initNetwork(config);
    startTraining();

    // Track training progress — store in ref for proper cleanup
    let progressCount = 0;
    intervalRef.current = setInterval(() => {
      if (!activeRef.current) {
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
          intervalRef.current = null;
        }
        return;
      }
      progressCount++;
      setState(prev => ({
        ...prev,
        epoch: Math.min(progressCount, CINEMATIC_TRAIN_EPOCHS),
        progress: Math.min(progressCount / CINEMATIC_TRAIN_EPOCHS, 1),
      }));
    }, CINEMATIC_EPOCH_INTERVAL);

    // After training, start drawing digits
    timerRef.current = setTimeout(() => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      stopTraining();
      drawDigit(0);
    }, CINEMATIC_TRAIN_EPOCHS * CINEMATIC_EPOCH_INTERVAL);
  }, [config, initNetwork, startTraining, stopTraining, stop, drawDigit]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      activeRef.current = false;
      clearTimer();
    };
  }, [clearTimer]);

  return {
    cinematic: state,
    startCinematic: start,
    stopCinematic: stop,
  };
}
