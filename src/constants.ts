/**
 * Application-wide constants for NeuralPlayground.
 *
 * Single source of truth for all magic numbers, timing values,
 * display sizes, and configuration defaults.
 */

import type { TrainingConfig, NoiseType } from './types';

// ─── Training defaults ───────────────────────────────────────────────

export const DEFAULT_CONFIG: TrainingConfig = {
  learningRate: 0.01,
  layers: [
    { neurons: 64, activation: 'relu' },
    { neurons: 32, activation: 'relu' },
  ],
};

export const INPUT_SIZE = 784; // 28 × 28
export const INPUT_DIM = 28;
export const OUTPUT_CLASSES = 10;
export const DEFAULT_SAMPLES_PER_DIGIT = 20;
export const NEURON_OPTIONS = [8, 16, 32, 64, 128, 256] as const;
export const MAX_HIDDEN_LAYERS = 5;

// ─── Cinematic demo ──────────────────────────────────────────────────

export const CINEMATIC_TRAIN_EPOCHS = 30;
export const CINEMATIC_PREDICT_DWELL = 1800; // ms to show prediction before next digit
export const CINEMATIC_EPOCH_INTERVAL = 80; // ms per epoch tick
export const AUTO_TRAIN_EPOCHS = 15; // auto-train on first load
export const AUTO_TRAIN_DELAY = 400; // ms before auto-start

// ─── Display sizes ───────────────────────────────────────────────────

export const NETWORK_VIS_ASPECT = 0.68;
export const LOSS_CHART_ASPECT = 0.355;
export const ACTIVATION_VIS_ASPECT = 0.875;
export const NETWORK_VIS_DEFAULT = { width: 620, height: 420 };
export const LOSS_CHART_DEFAULT = { width: 620, height: 220 };
export const ACTIVATION_VIS_DEFAULT = { width: 320, height: 280 };

// ─── Network visualizer ─────────────────────────────────────────────

export const VIS_PADDING = 50;
export const VIS_MAX_DISPLAYED_NODES = 16;
export const VIS_NODE_SPACING_MAX = 25;
export const SIGNAL_LAYER_DELAY = 0.35; // seconds between each layer's particles starting
export const SIGNAL_PARTICLE_SPEED_MIN = 1.8;
export const SIGNAL_PARTICLE_SPEED_RANGE = 0.8;
export const SIGNAL_WEIGHT_THRESHOLD = 0.05; // skip near-zero connections

// ─── Feature maps ────────────────────────────────────────────────────

export const FEATURE_MAP_CELL_SIZE = 38;
export const FEATURE_MAP_CELL_GAP = 3;
export const FEATURE_MAP_MAGNIFIER_SIZE = 140;
export const FEATURE_MAP_MAX_COLS = 8;

// ─── Adversarial lab ─────────────────────────────────────────────────

export const ADVERSARIAL_DISPLAY_SIZE = 160;
export const ADVERSARIAL_DEFAULT_SEED = 42;
export const ADVERSARIAL_DEFAULT_TARGET = 3;

export const NOISE_LABELS: Record<NoiseType, string> = {
  gaussian: '🌊 Gaussian',
  'salt-pepper': '🧂 Salt & Pepper',
  adversarial: '🎯 Targeted',
};

export const NOISE_DESCRIPTIONS: Record<NoiseType, string> = {
  gaussian: 'Random bell-curve noise — like TV static',
  'salt-pepper': 'Random black & white pixel flips',
  adversarial: 'Push the prediction toward a target digit',
};

// ─── Digit morph ─────────────────────────────────────────────────────

export const MORPH_DISPLAY_SIZE = 140;

// ─── Colors ──────────────────────────────────────────────────────────

export const COLOR_CYAN = 'rgba(99, 222, 255,';
export const COLOR_RED = 'rgba(255, 99, 132,';
export const COLOR_CYAN_HEX = '#63deff';
export const COLOR_RED_HEX = '#ff6384';
export const COLOR_GREEN_HEX = '#10b981';

// ─── Keyboard shortcuts ─────────────────────────────────────────────

export const SHORTCUTS = [
  { key: 'Space', description: 'Train / Pause' },
  { key: 'R', description: 'Reset network' },
  { key: 'D', description: 'Cinematic demo' },
  { key: 'H', description: 'Toggle help' },
  { key: 'Esc', description: 'Close / Stop demo' },
] as const;

// ─── Training step interval ─────────────────────────────────────────

export const TRAINING_STEP_INTERVAL = 60; // ms between training steps
