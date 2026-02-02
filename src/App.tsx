import { useState, useCallback, useEffect, useRef } from 'react';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import { canvasToInput } from './nn/sampleData';
import DrawingCanvas from './components/DrawingCanvas';
import type { DrawingCanvasHandle } from './components/DrawingCanvas';
import NetworkVisualizer from './components/NetworkVisualizer';
import LossChart from './components/LossChart';
import ActivationVisualizer from './components/ActivationVisualizer';
import PredictionBar from './components/PredictionBar';
import ControlPanel from './components/ControlPanel';
import WeightPanel from './components/WeightPanel';
import CinematicBadge from './components/CinematicBadge';
import DigitMorph from './components/DigitMorph';
import FeatureMaps from './components/FeatureMaps';
import AdversarialLab from './components/AdversarialLab';
import { DIGIT_STROKES, getDigitDrawDuration } from './data/digitStrokes';
import './App.css';

/** Cinematic demo constants */
const CINEMATIC_TRAIN_EPOCHS = 30;
const CINEMATIC_PREDICT_DWELL = 1800; // ms to show prediction before next digit

function App() {
  const {
    state,
    initNetwork,
    startTraining,
    stopTraining,
    predict,
    updateConfig,
    updateLayers,
  } = useNeuralNetwork();

  const [livePrediction, setLivePrediction] = useState<number[] | null>(null);
  const [predictedLabel, setPredictedLabel] = useState<number | null>(null);
  const [predictionLayers, setPredictionLayers] = useState<NonNullable<typeof state.snapshot>['layers'] | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // Signal flow animation trigger
  const [signalFlowTrigger, setSignalFlowTrigger] = useState(0);

  // Cinematic demo mode
  const [cinematicActive, setCinematicActive] = useState(false);
  const [cinematicPhase, setCinematicPhase] = useState<'training' | 'drawing' | 'predicting'>('training');
  const [cinematicDigit, setCinematicDigit] = useState(0);
  const [cinematicProgress, setCinematicProgress] = useState(0);
  const [cinematicEpoch, setCinematicEpoch] = useState(0);
  const cinematicRef = useRef(false);
  const cinematicTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Drawing canvas ref for programmatic drawing
  const drawingCanvasRef = useRef<DrawingCanvasHandle>(null);

  // Digit morph state
  const [morphSlotA, setMorphSlotA] = useState<number[] | null>(null);
  const [morphSlotB, setMorphSlotB] = useState<number[] | null>(null);

  // Adversarial lab — track current drawing as pixel array
  const [currentDrawingInput, setCurrentDrawingInput] = useState<number[] | null>(null);

  const handleDraw = useCallback((imageData: ImageData) => {
    const input = canvasToInput(imageData);
    setCurrentDrawingInput(input);
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
      // Trigger signal flow animation
      setSignalFlowTrigger(prev => prev + 1);
    }
  }, [predict]);

  const handleStart = useCallback(() => {
    if (state.epoch === 0) {
      initNetwork(state.config);
    }
    startTraining();
  }, [state.epoch, state.config, initNetwork, startTraining]);

  const handleReset = useCallback(() => {
    initNetwork(state.config);
    setLivePrediction(null);
    setPredictedLabel(null);
    setPredictionLayers(null);
    setCurrentDrawingInput(null);
  }, [initNetwork, state.config]);

  // ─── Save morph slot from current drawing ───
  const handleSaveMorphSlot = useCallback((slot: 'A' | 'B') => {
    const handle = drawingCanvasRef.current;
    if (!handle) return;
    const imageData = handle.getImageData();
    if (!imageData) return;
    const input = canvasToInput(imageData);
    if (slot === 'A') setMorphSlotA(input);
    else setMorphSlotB(input);
  }, []);

  // ─── Morph predict handler ───
  const handleMorphPredict = useCallback((input: number[]) => {
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
    }
  }, [predict]);

  // ─── Adversarial lab predict handler ───
  const handleAdversarialPredict = useCallback((input: number[]) => {
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
    }
  }, [predict]);

  // ─── Cinematic Demo Mode ───
  const stopCinematic = useCallback(() => {
    cinematicRef.current = false;
    setCinematicActive(false);
    if (cinematicTimerRef.current) clearTimeout(cinematicTimerRef.current);
  }, []);

  const startCinematic = useCallback(() => {
    if (cinematicRef.current) {
      stopCinematic();
      return;
    }
    cinematicRef.current = true;
    setCinematicActive(true);
    setCinematicPhase('training');
    setCinematicEpoch(0);
    setCinematicProgress(0);
    setCinematicDigit(0);

    // Stop any existing training
    stopTraining();

    // Phase 1: Train for N epochs
    initNetwork(state.config);
    let epochCount = 0;

    const trainStep = () => {
      if (!cinematicRef.current) return;
      epochCount++;
      setCinematicEpoch(epochCount);
      setCinematicProgress(epochCount / CINEMATIC_TRAIN_EPOCHS);

      if (epochCount < CINEMATIC_TRAIN_EPOCHS) {
        startTraining();
        // Training runs on a timer internally; we just need to track epochs
        // Instead, let's do manual epoch tracking via state updates
        cinematicTimerRef.current = setTimeout(trainStep, 80);
      } else {
        // Training phase done — move to drawing phase
        stopTraining();
        cinematicTimerRef.current = setTimeout(() => drawDigit(0), 500);
      }
    };

    // Start training — the hook handles the actual training loop
    startTraining();
    cinematicTimerRef.current = setTimeout(() => {
      stopTraining();
      drawDigit(0);
    }, CINEMATIC_TRAIN_EPOCHS * 80);

    // Track training progress
    let progressInterval: ReturnType<typeof setInterval> | null = null;
    progressInterval = setInterval(() => {
      if (!cinematicRef.current) {
        if (progressInterval) clearInterval(progressInterval);
        return;
      }
      // Use actual epoch from state
      setCinematicEpoch(prev => Math.min(prev + 1, CINEMATIC_TRAIN_EPOCHS));
      setCinematicProgress(prev => Math.min(prev + 1 / CINEMATIC_TRAIN_EPOCHS, 1));
    }, 80);

    // Phase 2: Auto-draw digits
    const drawDigit = (digit: number) => {
      if (!cinematicRef.current) return;
      setCinematicPhase('drawing');
      setCinematicDigit(digit);
      setCinematicProgress(0);

      const handle = drawingCanvasRef.current;
      if (!handle) {
        cinematicTimerRef.current = setTimeout(() => drawDigit((digit + 1) % 10), 1000);
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
        if (!cinematicRef.current) return;
        if (strokeIdx >= strokes.length) {
          // Drawing done — show prediction
          setCinematicPhase('predicting');
          setCinematicProgress(1);
          setSignalFlowTrigger(prev => prev + 1);

          cinematicTimerRef.current = setTimeout(() => {
            if (!cinematicRef.current) return;
            const nextDigit = (digit + 1) % 10;
            drawDigit(nextDigit);
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
        setCinematicProgress(pointsDrawn / totalPoints);
        pointIdx++;

        if (pointIdx >= stroke.points.length) {
          strokeIdx++;
          pointIdx = 0;
        }

        cinematicTimerRef.current = setTimeout(drawStep, pointDelay);
      };

      cinematicTimerRef.current = setTimeout(drawStep, 200);
    };
  }, [state.config, initNetwork, startTraining, stopTraining, stopCinematic, predict]);

  // Cleanup cinematic on unmount
  useEffect(() => {
    return () => {
      cinematicRef.current = false;
      if (cinematicTimerRef.current) clearTimeout(cinematicTimerRef.current);
    };
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          if (cinematicActive) { stopCinematic(); break; }
          if (state.isTraining) stopTraining();
          else handleStart();
          break;
        case 'r':
          if (!state.isTraining && !cinematicActive) handleReset();
          break;
        case 'h':
          setShowHelp(prev => !prev);
          break;
        case 'd':
          startCinematic();
          break;
        case 'escape':
          if (cinematicActive) stopCinematic();
          setShowHelp(false);
          break;
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [state.isTraining, cinematicActive, stopTraining, handleStart, handleReset, startCinematic, stopCinematic]);

  const displayLayers = predictionLayers || state.snapshot?.layers || null;

  return (
    <div className="app" role="application" aria-label="NeuralPlayground — Neural network visualizer">
      <a href="#main-content" className="sr-only focus-visible-skip">Skip to content</a>

      <header className="app-header">
        <div className="logo">
          <span className="logo-icon" aria-hidden="true">🧬</span>
          <h1>NeuralPlayground</h1>
        </div>
        <p className="subtitle">Watch neural networks learn in real-time</p>
      </header>

      <main className="app-main" id="main-content">
        <div className="grid-layout">
          <div className="column column-left">
            <ControlPanel
              config={state.config}
              isTraining={state.isTraining}
              epoch={state.epoch}
              onUpdateConfig={updateConfig}
              onUpdateLayers={updateLayers}
              onStart={handleStart}
              onStop={stopTraining}
              onReset={handleReset}
            />

            {/* Experience section */}
            <div className="experience-panel" role="group" aria-label="Experience modes">
              <div className="panel-header">
                <span className="panel-icon" aria-hidden="true">✨</span>
                <span>Experience</span>
              </div>
              <div className="experience-buttons">
                <button
                  className={`btn btn-experience ${cinematicActive ? 'active' : ''}`}
                  onClick={startCinematic}
                  aria-label={cinematicActive ? 'Stop cinematic demo' : 'Start cinematic demo'}
                  aria-pressed={cinematicActive}
                >
                  <span aria-hidden="true">🎬</span> {cinematicActive ? 'Stop Demo' : 'Cinematic'}
                </button>
              </div>
            </div>

            <DrawingCanvas ref={drawingCanvasRef} onDraw={handleDraw} />
            <PredictionBar probabilities={livePrediction} />
          </div>

          <div className="column column-center">
            <NetworkVisualizer
              layers={displayLayers}
              inputSize={784}
              signalFlowTrigger={signalFlowTrigger}
            />
            <LossChart
              lossHistory={state.lossHistory}
              accuracyHistory={state.accuracyHistory}
            />
            <FeatureMaps layers={displayLayers} />
          </div>

          <div className="column column-right">
            <ActivationVisualizer
              layers={displayLayers}
            />
            <WeightPanel layers={displayLayers} />

            <DigitMorph
              slotA={morphSlotA}
              slotB={morphSlotB}
              onMorphPredict={handleMorphPredict}
              onSaveSlot={handleSaveMorphSlot}
            />

            <AdversarialLab
              currentInput={currentDrawingInput}
              onPredict={handleAdversarialPredict}
              probabilities={livePrediction}
              predictedLabel={predictedLabel}
            />
            
            <div className="stats-panel" role="region" aria-label="Training statistics">
              <div className="panel-header">
                <span className="panel-icon" aria-hidden="true">📊</span>
                <span>Statistics</span>
              </div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-number" aria-label={`${state.epoch} epochs`}>{state.epoch}</span>
                  <span className="stat-desc">Epochs</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={state.snapshot ? `Loss ${state.snapshot.loss.toFixed(4)}` : 'No loss data'}>
                    {state.snapshot ? state.snapshot.loss.toFixed(4) : '—'}
                  </span>
                  <span className="stat-desc">Loss</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={state.snapshot ? `Accuracy ${(state.snapshot.accuracy * 100).toFixed(1)} percent` : 'No accuracy data'}>
                    {state.snapshot ? `${(state.snapshot.accuracy * 100).toFixed(1)}%` : '—'}
                  </span>
                  <span className="stat-desc">Accuracy</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={predictedLabel !== null ? `Predicted digit ${predictedLabel}` : 'No prediction'}>
                    {predictedLabel !== null ? predictedLabel : '—'}
                  </span>
                  <span className="stat-desc">Prediction</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Training status announcement for screen readers */}
      <div className="sr-only" role="status" aria-live="polite">
        {state.isTraining ? `Training in progress — epoch ${state.epoch}` : state.epoch > 0 ? `Training paused at epoch ${state.epoch}` : ''}
      </div>

      {/* Cinematic badge */}
      {cinematicActive && (
        <CinematicBadge
          phase={cinematicPhase}
          epoch={cinematicEpoch}
          maxEpochs={CINEMATIC_TRAIN_EPOCHS}
          currentDigit={cinematicDigit}
          progress={cinematicProgress}
        />
      )}

      {/* Help overlay */}
      {showHelp && (
        <div className="help-overlay" role="dialog" aria-label="Keyboard shortcuts" onClick={() => setShowHelp(false)}>
          <div className="help-panel" onClick={(e) => e.stopPropagation()}>
            <div className="help-header">
              <h2>Keyboard Shortcuts</h2>
              <button className="btn-close" onClick={() => setShowHelp(false)} aria-label="Close help">✕</button>
            </div>
            <div className="help-list">
              <div className="help-row"><kbd>Space</kbd><span>Train / Pause</span></div>
              <div className="help-row"><kbd>R</kbd><span>Reset network</span></div>
              <div className="help-row"><kbd>D</kbd><span>Cinematic demo</span></div>
              <div className="help-row"><kbd>H</kbd><span>Toggle help</span></div>
              <div className="help-row"><kbd>Esc</kbd><span>Close / Stop demo</span></div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions bar */}
      <div className="instructions-bar" aria-hidden="true">
        <span><kbd>Space</kbd> Train</span>
        <span><kbd>R</kbd> Reset</span>
        <span><kbd>D</kbd> Demo</span>
        <span><kbd>H</kbd> Help</span>
      </div>

      <footer className="app-footer">
        <span>NeuralPlayground — Custom neural network with real-time visualization</span>
        <span className="footer-dot" aria-hidden="true">·</span>
        <a href="https://github.com/kai-claw/neuralplayground" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>
      </footer>
    </div>
  );
}

export default App;
