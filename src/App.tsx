import { useState, useCallback, useEffect, useRef } from 'react';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import { useCinematic } from './hooks/useCinematic';
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
import {
  AUTO_TRAIN_EPOCHS,
  AUTO_TRAIN_DELAY,
  CINEMATIC_EPOCH_INTERVAL,
  CINEMATIC_TRAIN_EPOCHS,
  SHORTCUTS,
} from './constants';
import './App.css';

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

  // Auto-start flag
  const autoStartedRef = useRef(false);

  // Signal flow animation trigger
  const [signalFlowTrigger, setSignalFlowTrigger] = useState(0);

  // Drawing canvas ref for programmatic drawing
  const drawingCanvasRef = useRef<DrawingCanvasHandle>(null);

  // Digit morph state
  const [morphSlotA, setMorphSlotA] = useState<number[] | null>(null);
  const [morphSlotB, setMorphSlotB] = useState<number[] | null>(null);

  // Adversarial lab — track current drawing as pixel array
  const [currentDrawingInput, setCurrentDrawingInput] = useState<number[] | null>(null);

  // ─── Cinematic demo (extracted hook) ───
  const { cinematic, startCinematic, stopCinematic } = useCinematic({
    config: state.config,
    initNetwork,
    startTraining,
    stopTraining,
    drawingCanvasRef,
    onSignalFlow: () => setSignalFlowTrigger(prev => prev + 1),
  });

  const handleDraw = useCallback((imageData: ImageData) => {
    const input = canvasToInput(imageData);
    setCurrentDrawingInput(input);
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
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

  // ─── Auto-start training on first load for instant wow ───
  useEffect(() => {
    if (autoStartedRef.current) return;
    autoStartedRef.current = true;
    const timer = setTimeout(() => {
      initNetwork(state.config);
      startTraining();
      setTimeout(() => stopTraining(), AUTO_TRAIN_EPOCHS * CINEMATIC_EPOCH_INTERVAL);
    }, AUTO_TRAIN_DELAY);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          if (cinematic.active) { stopCinematic(); break; }
          if (state.isTraining) stopTraining();
          else handleStart();
          break;
        case 'r':
          if (!state.isTraining && !cinematic.active) handleReset();
          break;
        case 'h':
          setShowHelp(prev => !prev);
          break;
        case 'd':
          startCinematic();
          break;
        case 'escape':
          if (cinematic.active) stopCinematic();
          setShowHelp(false);
          break;
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [state.isTraining, cinematic.active, stopTraining, handleStart, handleReset, startCinematic, stopCinematic]);

  const displayLayers = predictionLayers || state.snapshot?.layers || null;

  return (
    <div className="app" role="application" aria-label="NeuralPlayground — Neural network visualizer">
      <a href="#main-content" className="sr-only focus-visible-skip">Skip to content</a>

      <header className="app-header">
        <div className="logo">
          <span className="logo-icon" aria-hidden="true">🧬</span>
          <h1>NeuralPlayground</h1>
          <span className={`heartbeat-dot ${state.isTraining ? 'active' : state.epoch > 0 ? 'idle' : 'off'}`} aria-hidden="true" title={state.isTraining ? 'Training…' : state.epoch > 0 ? 'Paused' : 'Not started'} />
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
                  className={`btn btn-experience ${cinematic.active ? 'active' : ''}`}
                  onClick={startCinematic}
                  aria-label={cinematic.active ? 'Stop cinematic demo' : 'Start cinematic demo'}
                  aria-pressed={cinematic.active}
                >
                  <span aria-hidden="true">🎬</span> {cinematic.active ? 'Stop Demo' : 'Cinematic'}
                </button>
              </div>
            </div>

            <div style={livePrediction ? {
              borderRadius: 'var(--radius)',
              boxShadow: `0 0 ${Math.round((livePrediction[predictedLabel ?? 0] ?? 0) * 20)}px rgba(16, 185, 129, ${(livePrediction[predictedLabel ?? 0] ?? 0) * 0.25})`,
              transition: 'box-shadow 0.4s ease',
            } : undefined}>
              <DrawingCanvas ref={drawingCanvasRef} onDraw={handleDraw} />
            </div>
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
            <ActivationVisualizer layers={displayLayers} />
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
                  <span className="stat-number" key={`epoch-${state.epoch}`} style={state.isTraining ? { animation: 'statTick 0.2s cubic-bezier(0.34, 1.56, 0.64, 1)' } : undefined} aria-label={`${state.epoch} epochs`}>{state.epoch}</span>
                  <span className="stat-desc">Epochs</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={state.snapshot ? `Loss ${state.snapshot.loss.toFixed(4)}` : 'No loss data'}
                    style={state.snapshot && state.snapshot.loss < 0.5 ? { color: 'var(--accent-green)' } : undefined}>
                    {state.snapshot ? state.snapshot.loss.toFixed(4) : '—'}
                  </span>
                  <span className="stat-desc">Loss</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={state.snapshot ? `Accuracy ${(state.snapshot.accuracy * 100).toFixed(1)} percent` : 'No accuracy data'}
                    style={state.snapshot && state.snapshot.accuracy > 0.8 ? { color: 'var(--accent-green)' } : undefined}>
                    {state.snapshot ? `${(state.snapshot.accuracy * 100).toFixed(1)}%` : '—'}
                  </span>
                  <span className="stat-desc">Accuracy</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number" aria-label={predictedLabel !== null ? `Predicted digit ${predictedLabel}` : 'No prediction'}
                    style={predictedLabel !== null ? { color: 'var(--accent-green)', fontSize: '24px' } : undefined}>
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
      {cinematic.active && (
        <CinematicBadge
          phase={cinematic.phase}
          epoch={cinematic.epoch}
          maxEpochs={CINEMATIC_TRAIN_EPOCHS}
          currentDigit={cinematic.digit}
          progress={cinematic.progress}
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
              {SHORTCUTS.map(({ key, description }) => (
                <div className="help-row" key={key}><kbd>{key}</kbd><span>{description}</span></div>
              ))}
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
        <span className="footer-version">v1.0.0</span>
        <span className="footer-dot" aria-hidden="true">·</span>
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
