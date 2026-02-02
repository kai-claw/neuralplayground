import { useState, useCallback, useEffect } from 'react';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import { canvasToInput } from './nn/sampleData';
import DrawingCanvas from './components/DrawingCanvas';
import NetworkVisualizer from './components/NetworkVisualizer';
import LossChart from './components/LossChart';
import ActivationVisualizer from './components/ActivationVisualizer';
import PredictionBar from './components/PredictionBar';
import ControlPanel from './components/ControlPanel';
import WeightPanel from './components/WeightPanel';
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

  const handleDraw = useCallback((imageData: ImageData) => {
    const input = canvasToInput(imageData);
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
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
  }, [initNetwork, state.config]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      // Skip when typing in inputs/selects
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key.toLowerCase()) {
        case ' ':
          e.preventDefault();
          if (state.isTraining) stopTraining();
          else handleStart();
          break;
        case 'r':
          if (!state.isTraining) handleReset();
          break;
        case 'h':
          setShowHelp(prev => !prev);
          break;
        case 'escape':
          setShowHelp(false);
          break;
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [state.isTraining, stopTraining, handleStart, handleReset]);

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
            <DrawingCanvas onDraw={handleDraw} />
            <PredictionBar probabilities={livePrediction} />
          </div>

          <div className="column column-center">
            <NetworkVisualizer
              layers={displayLayers}
              inputSize={784}
            />
            <LossChart
              lossHistory={state.lossHistory}
              accuracyHistory={state.accuracyHistory}
            />
          </div>

          <div className="column column-right">
            <ActivationVisualizer
              layers={displayLayers}
            />
            <WeightPanel layers={displayLayers} />
            
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
              <div className="help-row"><kbd>H</kbd><span>Toggle help</span></div>
              <div className="help-row"><kbd>Esc</kbd><span>Close overlay</span></div>
            </div>
          </div>
        </div>
      )}

      {/* Instructions bar */}
      <div className="instructions-bar" aria-hidden="true">
        <span><kbd>Space</kbd> Train</span>
        <span><kbd>R</kbd> Reset</span>
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
