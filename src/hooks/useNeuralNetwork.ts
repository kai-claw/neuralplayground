import { useState, useRef, useCallback, useEffect } from 'react';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig, TrainingSnapshot, LayerConfig } from '../nn/NeuralNetwork';
import { generateTrainingData } from '../nn/sampleData';

export interface NetworkState {
  isTraining: boolean;
  snapshot: TrainingSnapshot | null;
  lossHistory: number[];
  accuracyHistory: number[];
  epoch: number;
  config: TrainingConfig;
}

const defaultConfig: TrainingConfig = {
  learningRate: 0.01,
  layers: [
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
  ],
};

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
    config: defaultConfig,
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
    
    const data = customData || generateTrainingData(20);
    
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
        timerRef.current = setTimeout(step, 60);
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

  return {
    state,
    initNetwork,
    startTraining,
    stopTraining,
    predict,
    updateConfig,
    updateLayers,
  };
}
