import { useState, useCallback } from 'react';
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

  const displayLayers = predictionLayers || state.snapshot?.layers || null;

  return (
    <div className="app">
      <header className="app-header">
        <div className="logo">
          <span className="logo-icon">🧬</span>
          <h1>NeuralPlayground</h1>
        </div>
        <p className="subtitle">Watch neural networks learn in real-time</p>
      </header>

      <main className="app-main">
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
              width={620}
              height={420}
            />
            <LossChart
              lossHistory={state.lossHistory}
              accuracyHistory={state.accuracyHistory}
              width={620}
              height={220}
            />
          </div>

          <div className="column column-right">
            <ActivationVisualizer
              layers={displayLayers}
              width={320}
              height={280}
            />
            <WeightPanel layers={displayLayers} />
            
            <div className="stats-panel">
              <div className="panel-header">
                <span className="panel-icon">📊</span>
                <span>Statistics</span>
              </div>
              <div className="stats-grid">
                <div className="stat-item">
                  <span className="stat-number">{state.epoch}</span>
                  <span className="stat-desc">Epochs</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number">
                    {state.snapshot ? state.snapshot.loss.toFixed(4) : '—'}
                  </span>
                  <span className="stat-desc">Loss</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number">
                    {state.snapshot ? `${(state.snapshot.accuracy * 100).toFixed(1)}%` : '—'}
                  </span>
                  <span className="stat-desc">Accuracy</span>
                </div>
                <div className="stat-item">
                  <span className="stat-number">
                    {predictedLabel !== null ? predictedLabel : '—'}
                  </span>
                  <span className="stat-desc">Prediction</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <span>NeuralPlayground — Custom neural network with real-time visualization</span>
        <span className="footer-dot">·</span>
        <a href="https://github.com/kai-claw/neuralplayground" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>
      </footer>
    </div>
  );
}

export default App;
