import { useRef, useCallback, useState, useEffect } from 'react';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig, ActivationFn } from '../nn/NeuralNetwork';
import { generateTrainingData } from '../nn/sampleData';
import {
  RACE_EPOCHS,
  RACE_STEP_INTERVAL,
  DEFAULT_SAMPLES_PER_DIGIT,
} from '../constants';

export interface RacerConfig {
  name: string;
  color: string;
  config: TrainingConfig;
}

export interface RaceState {
  isRacing: boolean;
  epoch: number;
  lossA: number[];
  lossB: number[];
  accA: number[];
  accB: number[];
  winner: 'A' | 'B' | 'tie' | null;
}

const DEFAULT_RACER_A: RacerConfig = {
  name: 'Network A',
  color: '#63deff',
  config: {
    learningRate: 0.01,
    layers: [
      { neurons: 64, activation: 'relu' },
      { neurons: 32, activation: 'relu' },
    ],
  },
};

const DEFAULT_RACER_B: RacerConfig = {
  name: 'Network B',
  color: '#f59e0b',
  config: {
    learningRate: 0.01,
    layers: [
      { neurons: 32, activation: 'sigmoid' },
    ],
  },
};

export const RACE_PRESETS: { label: string; a: TrainingConfig; b: TrainingConfig }[] = [
  {
    label: 'Deep vs Shallow',
    a: {
      learningRate: 0.01,
      layers: [
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
      ],
    },
    b: {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    },
  },
  {
    label: 'ReLU vs Sigmoid',
    a: {
      learningRate: 0.01,
      layers: [{ neurons: 64, activation: 'relu' }],
    },
    b: {
      learningRate: 0.01,
      layers: [{ neurons: 64, activation: 'sigmoid' }],
    },
  },
  {
    label: 'Fast vs Slow LR',
    a: {
      learningRate: 0.05,
      layers: [{ neurons: 32, activation: 'relu' }],
    },
    b: {
      learningRate: 0.005,
      layers: [{ neurons: 32, activation: 'relu' }],
    },
  },
  {
    label: 'Wide vs Narrow',
    a: {
      learningRate: 0.01,
      layers: [{ neurons: 128, activation: 'relu' }],
    },
    b: {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    },
  },
];

export function useTrainingRace() {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const networkARef = useRef<NeuralNetwork | null>(null);
  const networkBRef = useRef<NeuralNetwork | null>(null);
  const dataRef = useRef<{ inputs: number[][]; labels: number[] } | null>(null);

  const [racerA, setRacerA] = useState<RacerConfig>(DEFAULT_RACER_A);
  const [racerB, setRacerB] = useState<RacerConfig>(DEFAULT_RACER_B);
  const [raceState, setRaceState] = useState<RaceState>({
    isRacing: false,
    epoch: 0,
    lossA: [],
    lossB: [],
    accA: [],
    accB: [],
    winner: null,
  });

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      // Release network weight matrices on unmount
      networkARef.current = null;
      networkBRef.current = null;
      dataRef.current = null;
    };
  }, []);

  const startRace = useCallback(() => {
    if (raceState.isRacing) return;

    dataRef.current = generateTrainingData(DEFAULT_SAMPLES_PER_DIGIT);
    networkARef.current = new NeuralNetwork(784, racerA.config);
    networkBRef.current = new NeuralNetwork(784, racerB.config);

    setRaceState({
      isRacing: true,
      epoch: 0,
      lossA: [],
      lossB: [],
      accA: [],
      accB: [],
      winner: null,
    });

    let epoch = 0;
    const lossA: number[] = [];
    const lossB: number[] = [];
    const accA: number[] = [];
    const accB: number[] = [];

    const step = () => {
      if (epoch >= RACE_EPOCHS || !dataRef.current) {
        const finalAccA = accA[accA.length - 1] || 0;
        const finalAccB = accB[accB.length - 1] || 0;
        const winner: 'A' | 'B' | 'tie' =
          Math.abs(finalAccA - finalAccB) < 0.02 ? 'tie' :
          finalAccA > finalAccB ? 'A' : 'B';

        setRaceState(prev => ({ ...prev, isRacing: false, winner }));
        return;
      }

      const data = dataRef.current;

      if (networkARef.current) {
        const snapA = networkARef.current.trainBatch(data.inputs, data.labels);
        lossA.push(snapA.loss);
        accA.push(snapA.accuracy);
      }

      if (networkBRef.current) {
        const snapB = networkBRef.current.trainBatch(data.inputs, data.labels);
        lossB.push(snapB.loss);
        accB.push(snapB.accuracy);
      }

      epoch++;
      setRaceState(prev => ({
        ...prev,
        epoch,
        lossA: [...lossA],
        lossB: [...lossB],
        accA: [...accA],
        accB: [...accB],
      }));

      timerRef.current = setTimeout(step, RACE_STEP_INTERVAL);
    };

    step();
  }, [raceState.isRacing, racerA.config, racerB.config]);

  const stopRace = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    // Release network weight matrices to free memory
    networkARef.current = null;
    networkBRef.current = null;
    dataRef.current = null;
    setRaceState(prev => ({ ...prev, isRacing: false }));
  }, []);

  const applyPreset = useCallback((preset: typeof RACE_PRESETS[number]) => {
    if (raceState.isRacing) return;
    setRacerA(prev => ({ ...prev, config: preset.a }));
    setRacerB(prev => ({ ...prev, config: preset.b }));
  }, [raceState.isRacing]);

  const updateRacerLayers = useCallback((
    racer: 'A' | 'B',
    neurons: number,
    activation: ActivationFn,
  ) => {
    if (raceState.isRacing) return;
    const update = (prev: RacerConfig): RacerConfig => ({
      ...prev,
      config: {
        ...prev.config,
        layers: [{ neurons, activation }],
      },
    });
    if (racer === 'A') setRacerA(update);
    else setRacerB(update);
  }, [raceState.isRacing]);

  return {
    racerA,
    racerB,
    raceState,
    startRace,
    stopRace,
    applyPreset,
    updateRacerLayers,
  };
}
