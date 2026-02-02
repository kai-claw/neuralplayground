import { useState, useRef, useCallback, useEffect } from 'react';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig, TrainingSnapshot, LayerConfig, NeuronStatus } from '../nn/NeuralNetwork';
import { generateTrainingData } from '../nn/sampleData';
import { DEFAULT_CONFIG, DEFAULT_SAMPLES_PER_DIGIT, TRAINING_STEP_INTERVAL } from '../constants';

export interface NetworkState {
  isTraining: boolean;
  snapshot: TrainingSnapshot | null;
  lossHistory: number[];
  accuracyHistory: number[];
  epoch: number;
  config: TrainingConfig;
}

export function useNeuralNetwork() {
  const networkRef = useRef<NeuralNetwork | null>(null);
  const trainingRef = useRef<boolean>(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [state, setState] = useState<NetworkState>({
    isTraining: false,
    snapshot: null,
    lossHistory: [],
    accuracyHistory: [],
    epoch: 0,
    config: DEFAULT_CONFIG,
  });

  // BUG FIX: Clean up training timer on unmount to prevent memory leak
  useEffect(() => {
    return () => {
      trainingRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  const initNetwork = useCallback((config?: TrainingConfig) => {
    const cfg = config || state.config;
    networkRef.current = new NeuralNetwork(784, cfg);
    trainingRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setState(prev => ({
      ...prev,
      config: cfg,
      snapshot: null,
      lossHistory: [],
      accuracyHistory: [],
      epoch: 0,
      isTraining: false,
    }));
  }, [state.config]);

  const startTraining = useCallback((customData?: { inputs: number[][]; labels: number[] }) => {
    if (!networkRef.current) {
      networkRef.current = new NeuralNetwork(784, state.config);
    }
    
    trainingRef.current = true;
    setState(prev => ({ ...prev, isTraining: true }));
    
    const data = customData || generateTrainingData(DEFAULT_SAMPLES_PER_DIGIT);
    
    const step = () => {
      if (!trainingRef.current || !networkRef.current) return;
      
      const snapshot = networkRef.current.trainBatch(data.inputs, data.labels);
      
      setState(prev => ({
        ...prev,
        snapshot,
        lossHistory: networkRef.current!.getLossHistory(),
        accuracyHistory: networkRef.current!.getAccuracyHistory(),
        epoch: snapshot.epoch,
      }));
      
      if (trainingRef.current) {
        timerRef.current = setTimeout(step, TRAINING_STEP_INTERVAL);
      }
    };
    
    step();
  }, [state.config]);

  const stopTraining = useCallback(() => {
    trainingRef.current = false;
    if (timerRef.current) clearTimeout(timerRef.current);
    setState(prev => ({ ...prev, isTraining: false }));
  }, []);

  const predict = useCallback((input: number[]) => {
    if (!networkRef.current) return null;
    return networkRef.current.predict(input);
  }, []);

  const updateConfig = useCallback((updates: Partial<TrainingConfig>) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, ...updates },
    }));
  }, []);

  const updateLayers = useCallback((layers: LayerConfig[]) => {
    setState(prev => ({
      ...prev,
      config: { ...prev.config, layers },
    }));
  }, []);

  // ─── Neuron Surgery API ──────────────────────────────────────────

  const setNeuronStatus = useCallback((layerIdx: number, neuronIdx: number, status: NeuronStatus) => {
    if (!networkRef.current) return;
    networkRef.current.setNeuronStatus(layerIdx, neuronIdx, status);
  }, []);

  const getNeuronStatus = useCallback((layerIdx: number, neuronIdx: number): NeuronStatus => {
    if (!networkRef.current) return 'active';
    return networkRef.current.getNeuronStatus(layerIdx, neuronIdx);
  }, []);

  const clearNeuronMasks = useCallback(() => {
    if (!networkRef.current) return;
    networkRef.current.clearAllMasks();
  }, []);

  // ─── Network Dreams API ──────────────────────────────────────────

  const computeInputGradient = useCallback((input: number[], targetClass: number): number[] | null => {
    if (!networkRef.current) return null;
    return networkRef.current.computeInputGradient(input, targetClass);
  }, []);

  const dream = useCallback((
    targetClass: number,
    steps?: number,
    lr?: number,
    startImage?: number[],
  ) => {
    if (!networkRef.current) return null;
    return networkRef.current.dream(targetClass, steps, lr, startImage);
  }, []);

  return {
    state,
    initNetwork,
    startTraining,
    stopTraining,
    predict,
    updateConfig,
    updateLayers,
    setNeuronStatus,
    getNeuronStatus,
    clearNeuronMasks,
    computeInputGradient,
    dream,
  };
}
