import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig, TrainingSnapshot, LayerState, ActivationFn, LayerConfig } from '../nn/NeuralNetwork';

describe('NeuralNetwork — Construction & Initialization', () => {
  const defaultConfig: TrainingConfig = {
    learningRate: 0.01,
    layers: [
      { neurons: 16, activation: 'relu' },
      { neurons: 8, activation: 'relu' },
    ],
  };

  it('should create a network with correct layer count (hidden + output)', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    const snap = nn.predict(new Array(784).fill(0));
    // 2 hidden + 1 output = 3 layers
    expect(snap.layers.length).toBe(3);
  });

  it('should initialize with correct weight dimensions', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    const snap = nn.predict(new Array(784).fill(0));
    // Layer 0: 16 neurons × 784 inputs
    expect(snap.layers[0].weights.length).toBe(16);
    expect(snap.layers[0].weights[0].length).toBe(784);
    // Layer 1: 8 neurons × 16 inputs
    expect(snap.layers[1].weights.length).toBe(8);
    expect(snap.layers[1].weights[0].length).toBe(16);
    // Output: 10 neurons × 8 inputs
    expect(snap.layers[2].weights.length).toBe(10);
    expect(snap.layers[2].weights[0].length).toBe(8);
  });

  it('should initialize biases with correct lengths', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    const snap = nn.predict(new Array(784).fill(0));
    expect(snap.layers[0].biases.length).toBe(16);
    expect(snap.layers[1].biases.length).toBe(8);
    expect(snap.layers[2].biases.length).toBe(10);
  });

  it('should start at epoch 0', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    expect(nn.getEpoch()).toBe(0);
  });

  it('should start with empty loss history', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    expect(nn.getLossHistory().length).toBe(0);
  });

  it('should start with empty accuracy history', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    expect(nn.getAccuracyHistory().length).toBe(0);
  });

  it('should support single hidden layer', () => {
    const cfg: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, cfg);
    const snap = nn.predict(new Array(784).fill(0));
    expect(snap.layers.length).toBe(2); // 1 hidden + 1 output
  });

  it('should support many hidden layers', () => {
    const cfg: TrainingConfig = {
      learningRate: 0.01,
      layers: [
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'sigmoid' },
        { neurons: 16, activation: 'tanh' },
        { neurons: 8, activation: 'relu' },
      ],
    };
    const nn = new NeuralNetwork(784, cfg);
    const snap = nn.predict(new Array(784).fill(0));
    expect(snap.layers.length).toBe(5); // 4 hidden + 1 output
  });

  it('should use Xavier initialization (weights distributed around 0)', () => {
    const nn = new NeuralNetwork(784, defaultConfig);
    const snap = nn.predict(new Array(784).fill(0));
    const weights = snap.layers[0].weights.flat();
    const mean = weights.reduce((a, b) => a + b, 0) / weights.length;
    // Xavier init should center around 0
    expect(Math.abs(mean)).toBeLessThan(0.1);
  });
});

describe('NeuralNetwork — Forward Pass', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [{ neurons: 16, activation: 'relu' }],
  };

  it('should produce output of length 10 (digit classes)', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0));
    expect(snap.probabilities.length).toBe(10);
  });

  it('should produce valid softmax (sums to ~1)', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    const sum = snap.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
  });

  it('should produce all positive probabilities', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.3));
    for (const p of snap.probabilities) {
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it('should return predicted label in range 0-9', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    expect(snap.label).toBeGreaterThanOrEqual(0);
    expect(snap.label).toBeLessThanOrEqual(9);
  });

  it('should return layers with correct activation dimensions', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    expect(snap.layers[0].activations.length).toBe(16);
    expect(snap.layers[1].activations.length).toBe(10);
  });

  it('should not contain NaN in output', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    for (const p of snap.probabilities) {
      expect(isNaN(p)).toBe(false);
    }
  });

  it('should not contain NaN in activations', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    for (const layer of snap.layers) {
      for (const a of layer.activations) {
        expect(isNaN(a)).toBe(false);
      }
    }
  });

  it('should handle zero input', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0));
    expect(snap.probabilities.length).toBe(10);
    const sum = snap.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
  });

  it('should handle all-ones input', () => {
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(1));
    expect(snap.probabilities.length).toBe(10);
    const sum = snap.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
  });
});

describe('NeuralNetwork — Activation Functions', () => {
  it('ReLU: hidden layer activations should be non-negative', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    for (const a of snap.layers[0].activations) {
      expect(a).toBeGreaterThanOrEqual(0);
    }
  });

  it('sigmoid: hidden layer activations should be in (0, 1)', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'sigmoid' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    for (const a of snap.layers[0].activations) {
      expect(a).toBeGreaterThanOrEqual(0);
      expect(a).toBeLessThanOrEqual(1);
    }
  });

  it('tanh: hidden layer activations should be in [-1, 1]', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'tanh' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(0.5));
    for (const a of snap.layers[0].activations) {
      expect(a).toBeGreaterThanOrEqual(-1);
      expect(a).toBeLessThanOrEqual(1);
    }
  });
});

describe('NeuralNetwork — Training', () => {
  function makeTrainingData() {
    const inputs: number[][] = [];
    const labels: number[] = [];
    for (let d = 0; d < 10; d++) {
      for (let s = 0; s < 3; s++) {
        const input = new Array(784).fill(0);
        // Simple pattern: set a block based on digit
        for (let i = d * 78; i < d * 78 + 78; i++) {
          input[i] = 0.8 + Math.random() * 0.2;
        }
        inputs.push(input);
        labels.push(d);
      }
    }
    return { inputs, labels };
  }

  it('should increment epoch after training batch', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    nn.trainBatch(data.inputs, data.labels);
    expect(nn.getEpoch()).toBe(1);
  });

  it('should append to loss history each epoch', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    nn.trainBatch(data.inputs, data.labels);
    nn.trainBatch(data.inputs, data.labels);
    nn.trainBatch(data.inputs, data.labels);
    expect(nn.getLossHistory().length).toBe(3);
    expect(nn.getAccuracyHistory().length).toBe(3);
  });

  it('should return valid snapshot with all fields', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    const snap = nn.trainBatch(data.inputs, data.labels);
    expect(snap.epoch).toBe(1);
    expect(typeof snap.loss).toBe('number');
    expect(typeof snap.accuracy).toBe('number');
    expect(snap.layers.length).toBe(2);
    expect(snap.predictions.length).toBe(10);
    expect(snap.outputProbabilities.length).toBe(10);
  });

  it('should produce finite loss', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    const snap = nn.trainBatch(data.inputs, data.labels);
    expect(isFinite(snap.loss)).toBe(true);
    expect(snap.loss).toBeGreaterThanOrEqual(0);
  });

  it('should produce accuracy in [0, 1]', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    const snap = nn.trainBatch(data.inputs, data.labels);
    expect(snap.accuracy).toBeGreaterThanOrEqual(0);
    expect(snap.accuracy).toBeLessThanOrEqual(1);
  });

  it('loss should decrease over many epochs', () => {
    const config: TrainingConfig = {
      learningRate: 0.05,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    for (let i = 0; i < 50; i++) {
      nn.trainBatch(data.inputs, data.labels);
    }
    const losses = nn.getLossHistory();
    // Average of first 5 should be higher than average of last 5
    const firstAvg = losses.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
    const lastAvg = losses.slice(-5).reduce((a, b) => a + b, 0) / 5;
    expect(lastAvg).toBeLessThan(firstAvg);
  });

  it('should remain stable with 100 training epochs', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = makeTrainingData();
    for (let i = 0; i < 100; i++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
      expect(isNaN(snap.loss)).toBe(false);
    }
  });
});

describe('NeuralNetwork — Reset', () => {
  it('should reset epoch to 0', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = { inputs: [new Array(784).fill(0.5)], labels: [3] };
    nn.trainBatch(data.inputs, data.labels);
    expect(nn.getEpoch()).toBe(1);
    nn.reset(784);
    expect(nn.getEpoch()).toBe(0);
  });

  it('should clear loss/accuracy history on reset', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = { inputs: [new Array(784).fill(0.5)], labels: [3] };
    nn.trainBatch(data.inputs, data.labels);
    nn.reset(784);
    expect(nn.getLossHistory().length).toBe(0);
    expect(nn.getAccuracyHistory().length).toBe(0);
  });

  it('should accept new config on reset', () => {
    const config1: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const config2: TrainingConfig = {
      learningRate: 0.05,
      layers: [{ neurons: 64, activation: 'sigmoid' }],
    };
    const nn = new NeuralNetwork(784, config1);
    nn.reset(784, config2);
    const snap = nn.predict(new Array(784).fill(0.5));
    // New architecture: 64 hidden + 10 output = 2 layers
    expect(snap.layers.length).toBe(2);
    expect(snap.layers[0].weights.length).toBe(64);
  });
});

describe('NeuralNetwork — Stability & Edge Cases', () => {
  it('should handle extreme input values', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(100));
    for (const p of snap.probabilities) {
      expect(isFinite(p)).toBe(true);
      expect(isNaN(p)).toBe(false);
    }
  });

  it('should handle negative input values', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(-1));
    for (const p of snap.probabilities) {
      expect(isFinite(p)).toBe(true);
    }
  });

  it('should handle very small input values', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap = nn.predict(new Array(784).fill(1e-10));
    const sum = snap.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(1e-6);
  });

  it('should produce different outputs for different inputs', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const input1 = new Array(784).fill(0);
    input1[0] = 1;
    const input2 = new Array(784).fill(0);
    input2[783] = 1;
    const snap1 = nn.predict(input1);
    const snap2 = nn.predict(input2);
    // At least one probability should differ
    const differs = snap1.probabilities.some((p, i) => Math.abs(p - snap2.probabilities[i]) > 1e-10);
    expect(differs).toBe(true);
  });

  it('should be stable with high learning rate training', () => {
    const config: TrainingConfig = {
      learningRate: 0.1,
      layers: [{ neurons: 32, activation: 'sigmoid' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = { inputs: [new Array(784).fill(0.5)], labels: [5] };
    for (let i = 0; i < 20; i++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
    }
  });

  it('should return cached snapshot that refreshes in-place after weight changes', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const snap1 = nn.predict(new Array(784).fill(0.5));
    const snap2 = nn.predict(new Array(784).fill(0.5));
    // Perf optimization: cached snapshot reused between calls
    expect(snap1.layers).toBe(snap2.layers);
    // Training changes weights — snapshot updates in-place on next predict
    nn.trainBatch([new Array(784).fill(0.5)], [3]);
    const snap3 = nn.predict(new Array(784).fill(0.5));
    // Same cached object reference, updated in-place
    expect(snap3.layers).toBe(snap1.layers);
    // Probabilities are still independent per-call copies
    expect(snap1.probabilities).not.toBe(snap3.probabilities);
  });

  it('getLossHistory should return readonly live array (perf: no copy overhead)', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = { inputs: [new Array(784).fill(0.5)], labels: [5] };
    nn.trainBatch(data.inputs, data.labels);
    const history = nn.getLossHistory();
    // Performance optimization: returns readonly live array instead of copying
    expect(history.length).toBe(1);
    // Training another batch grows the same array
    nn.trainBatch(data.inputs, data.labels);
    expect(nn.getLossHistory().length).toBe(2);
    // Reference is the same (no copy)
    expect(nn.getLossHistory()).toBe(nn.getLossHistory());
  });
});

describe('NeuralNetwork — Type System', () => {
  it('ActivationFn should accept relu/sigmoid/tanh', () => {
    const fns: ActivationFn[] = ['relu', 'sigmoid', 'tanh'];
    expect(fns.length).toBe(3);
  });

  it('TrainingConfig should have learningRate and layers', () => {
    const cfg: TrainingConfig = { learningRate: 0.01, layers: [] };
    expect(cfg.learningRate).toBe(0.01);
    expect(Array.isArray(cfg.layers)).toBe(true);
  });

  it('LayerConfig should have neurons and activation', () => {
    const lc: LayerConfig = { neurons: 32, activation: 'relu' };
    expect(lc.neurons).toBe(32);
    expect(lc.activation).toBe('relu');
  });

  it('LayerState should have weights, biases, preActivations, activations', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 8, activation: 'relu' }] });
    const snap = nn.predict(new Array(784).fill(0));
    const layer: LayerState = snap.layers[0];
    expect(Array.isArray(layer.weights)).toBe(true);
    expect(Array.isArray(layer.biases)).toBe(true);
    expect(Array.isArray(layer.preActivations)).toBe(true);
    expect(Array.isArray(layer.activations)).toBe(true);
  });

  it('TrainingSnapshot should have all required fields', () => {
    const nn = new NeuralNetwork(784, { learningRate: 0.01, layers: [{ neurons: 8, activation: 'relu' }] });
    const data = { inputs: [new Array(784).fill(0.5)], labels: [5] };
    const snap: TrainingSnapshot = nn.trainBatch(data.inputs, data.labels);
    expect(typeof snap.epoch).toBe('number');
    expect(typeof snap.loss).toBe('number');
    expect(typeof snap.accuracy).toBe('number');
    expect(Array.isArray(snap.layers)).toBe(true);
    expect(Array.isArray(snap.predictions)).toBe(true);
    expect(Array.isArray(snap.outputProbabilities)).toBe(true);
  });
});
