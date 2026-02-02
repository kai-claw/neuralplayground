/**
 * Training Race presets — curated network configuration matchups.
 *
 * Each preset defines two TrainingConfig objects for head-to-head comparison,
 * isolating a single variable (depth, activation, learning rate, width).
 */

import type { TrainingConfig } from '../types';

export interface RacePreset {
  label: string;
  a: TrainingConfig;
  b: TrainingConfig;
}

export const RACE_PRESETS: RacePreset[] = [
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

/** Default racer A configuration */
export const DEFAULT_RACER_A_CONFIG: TrainingConfig = {
  learningRate: 0.01,
  layers: [
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
  ],
};

/** Default racer B configuration */
export const DEFAULT_RACER_B_CONFIG: TrainingConfig = {
  learningRate: 0.01,
  layers: [
    { neurons: 32, activation: 'sigmoid' },
  ],
};
