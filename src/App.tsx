import { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { useNeuralNetwork } from './hooks/useNeuralNetwork';
import { useCinematic } from './hooks/useCinematic';
import { useActivationSpace } from './hooks/useActivationSpace';
import { usePerformanceMonitor } from './hooks/usePerformanceMonitor';
import { canvasToInput } from './nn/sampleData';
import { WeightEvolutionRecorder } from './nn/weightEvolution';
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
import NeuronSurgery from './components/NeuronSurgery';
import NetworkDreams from './components/NetworkDreams';
import TrainingRace from './components/TrainingRace';
import StatsPanel from './components/StatsPanel';
import HelpOverlay from './components/HelpOverlay';
import ExperiencePanel from './components/ExperiencePanel';
import SaliencyMap from './components/SaliencyMap';
import ActivationSpace from './components/ActivationSpace';
import ConfusionMatrix from './components/ConfusionMatrix';
import GradientFlowMonitor from './components/GradientFlowMonitor';
import EpochReplay from './components/EpochReplay';
import DecisionBoundary from './components/DecisionBoundary';
import ChimeraLab from './components/ChimeraLab';
import MisfitGallery from './components/MisfitGallery';
import WeightEvolution from './components/WeightEvolution';
import AblationLab from './components/AblationLab';
import {
  AUTO_TRAIN_EPOCHS,
  AUTO_TRAIN_DELAY,
  CINEMATIC_EPOCH_INTERVAL,
  CINEMATIC_TRAIN_EPOCHS,
} from './constants';
import './App.css';

function App() {
  const {
    state,
    networkRef,
    initNetwork,
    startTraining,
    stopTraining,
    predict,
    updateConfig,
    updateLayers,
    setNeuronStatus,
    getNeuronStatus,
    clearNeuronMasks,
    dream,
    saliency: computeSaliency,
  } = useNeuralNetwork();

  const [livePrediction, setLivePrediction] = useState<number[] | null>(null);
  const [predictedLabel, setPredictedLabel] = useState<number | null>(null);
  const [predictionLayers, setPredictionLayers] = useState<NonNullable<typeof state.snapshot>['layers'] | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  // ─── Performance monitoring ───
  const perfState = usePerformanceMonitor();

  // Auto-start flag
  const autoStartedRef = useRef(false);

  // Signal flow animation trigger
  const [signalFlowTrigger, setSignalFlowTrigger] = useState(0);

  // Weight evolution recorder
  const weightRecorderRef = useRef(new WeightEvolutionRecorder());
  // Trigger re-render when new frames are recorded
  const [, setWeightFrameTick] = useState(0);

  // Drawing canvas ref for programmatic drawing
  const drawingCanvasRef = useRef<DrawingCanvasHandle>(null);

  // Digit morph state
  const [morphSlotA, setMorphSlotA] = useState<number[] | null>(null);
  const [morphSlotB, setMorphSlotB] = useState<number[] | null>(null);

  // Adversarial lab — track current drawing as pixel array
  const [currentDrawingInput, setCurrentDrawingInput] = useState<number[] | null>(null);

  // Saliency map — computed from current input + predicted label.
  // Only recompute when the drawing or prediction changes, NOT on every epoch.
  // During pure training (no drawing), this is a no-op.
  const saliencyData = useMemo(() => {
    if (!currentDrawingInput || predictedLabel === null || state.epoch === 0) return null;
    return computeSaliency(currentDrawingInput, predictedLabel);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentDrawingInput, predictedLabel, computeSaliency]);

  // ─── Record weight evolution on each epoch ───
  // When performance is degraded, record every 3rd epoch to reduce allocation pressure
  // (each frame allocates a Float32Array of ~50KB for 16-neuron networks)
  useEffect(() => {
    if (state.snapshot && state.epoch > 0) {
      if (perfState.degraded && state.epoch % 3 !== 0) return;
      weightRecorderRef.current.record(state.snapshot);
      setWeightFrameTick(prev => prev + 1);
    }
  }, [state.epoch, state.snapshot, perfState.degraded]);

  // ─── Activation space projection ───
  const activationProjection = useActivationSpace(networkRef, state.epoch, currentDrawingInput);

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
    weightRecorderRef.current.clear();
    setWeightFrameTick(0);
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

  // ─── Shared predict handler (morph + adversarial) ───
  const handleExternalPredict = useCallback((input: number[]) => {
    const result = predict(input);
    if (result) {
      setLivePrediction(result.probabilities);
      setPredictedLabel(result.label);
      setPredictionLayers(result.layers);
    }
  }, [predict]);

  // ─── Neuron surgery change handler — re-predict with modified network ───
  const handleSurgeryChange = useCallback(() => {
    if (currentDrawingInput) {
      const result = predict(currentDrawingInput);
      if (result) {
        setLivePrediction(result.probabilities);
        setPredictedLabel(result.label);
        setPredictionLayers(result.layers);
      }
    }
  }, [currentDrawingInput, predict]);

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
          {state.isTraining && (
            <span
              className={`fps-badge ${perfState.fps >= 45 ? 'fps-good' : perfState.fps >= 30 ? 'fps-warn' : 'fps-bad'}`}
              title={`${perfState.fps} FPS${perfState.degraded ? ' — quality reduced' : ''}`}
              aria-hidden="true"
            >
              {perfState.fps} fps{perfState.degraded ? ' ⚠' : ''}
            </span>
          )}
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

            <ExperiencePanel
              cinematicActive={cinematic.active}
              onStartCinematic={startCinematic}
            />

            <div style={livePrediction ? {
              borderRadius: 'var(--radius)',
              boxShadow: `0 0 ${Math.round((livePrediction[predictedLabel ?? 0] ?? 0) * 20)}px rgba(16, 185, 129, ${(livePrediction[predictedLabel ?? 0] ?? 0) * 0.25})`,
              transition: 'box-shadow 0.4s ease',
            } : undefined}>
              <DrawingCanvas ref={drawingCanvasRef} onDraw={handleDraw} />
            </div>
            <PredictionBar probabilities={livePrediction} />

            <SaliencyMap
              saliency={saliencyData}
              currentInput={currentDrawingInput}
              predictedLabel={predictedLabel}
              hasTrained={state.epoch > 0}
            />
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
              onMorphPredict={handleExternalPredict}
              onSaveSlot={handleSaveMorphSlot}
            />

            <AdversarialLab
              currentInput={currentDrawingInput}
              onPredict={handleExternalPredict}
              probabilities={livePrediction}
              predictedLabel={predictedLabel}
            />
            
            <StatsPanel
              epoch={state.epoch}
              isTraining={state.isTraining}
              loss={state.snapshot?.loss ?? null}
              accuracy={state.snapshot?.accuracy ?? null}
              predictedLabel={predictedLabel}
            />
          </div>
        </div>

        {/* ─── Creative Features Row ─── */}
        {/* When performance is degraded, skip heavy analysis components to reduce CPU load */}
        <div className="creative-features-row">
          <NeuronSurgery
            layers={displayLayers}
            onSetNeuronStatus={setNeuronStatus}
            onGetNeuronStatus={getNeuronStatus}
            onClearAll={clearNeuronMasks}
            onSurgeryChange={handleSurgeryChange}
            currentPrediction={livePrediction}
            predictedLabel={predictedLabel}
          />
          <NetworkDreams
            onDream={dream}
            hasTrained={state.epoch > 0}
          />
          {!perfState.degraded && <TrainingRace />}
          <ActivationSpace
            projection={activationProjection}
            epoch={state.epoch}
            predictedLabel={predictedLabel}
          />
          <ConfusionMatrix
            networkRef={networkRef}
            epoch={state.epoch}
            isTraining={state.isTraining}
          />
          <GradientFlowMonitor
            networkRef={networkRef}
            epoch={state.epoch}
            layers={displayLayers}
            isTraining={state.isTraining}
          />
          <EpochReplay
            epoch={state.epoch}
            isTraining={state.isTraining}
            snapshot={state.snapshot}
            currentInput={currentDrawingInput}
            activationFn={state.config.layers[0]?.activation || 'relu'}
          />
          {!perfState.degraded && (
            <DecisionBoundary
              networkRef={networkRef}
              epoch={state.epoch}
              isTraining={state.isTraining}
            />
          )}
          <ChimeraLab
            networkRef={networkRef}
            hasTrained={state.epoch > 0}
          />
          {!perfState.degraded && (
            <MisfitGallery
              networkRef={networkRef}
              epoch={state.epoch}
              isTraining={state.isTraining}
            />
          )}
          <WeightEvolution
            frames={weightRecorderRef.current.getFrames()}
          />
          {!perfState.degraded && (
            <AblationLab
              networkRef={networkRef}
              epoch={state.epoch}
              isTraining={state.isTraining}
            />
          )}
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
      {showHelp && <HelpOverlay onClose={() => setShowHelp(false)} />}

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
