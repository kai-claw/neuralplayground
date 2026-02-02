import { describe, it, expect } from 'vitest';
import { NeuralNetwork, type TrainingConfig, type ActivationFn } from '../nn/NeuralNetwork';
import { generateTrainingData, canvasToInput } from '../nn/sampleData';

// ─── NaN / Infinity Stability Guards ──────────────────────────────
describe('NaN / Infinity stability', () => {
  it('should survive forward pass with NaN inputs', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 4, activation: 'relu' }],
    });
    const result = nn.forward([NaN, 0.5, Infinity, -Infinity]);
    expect(result.every(v => isFinite(v))).toBe(true);
    expect(result.every(v => !isNaN(v))).toBe(true);
  });

  it('should produce valid softmax even with extreme pre-activations', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'sigmoid' }],
    });
    // Force extreme inputs that could cause overflow
    const result = nn.forward([1e6, -1e6, 1e6, -1e6]);
    expect(result.every(v => isFinite(v))).toBe(true);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.001);
  });

  it('should not corrupt weights after training with extreme inputs', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.001,
      layers: [{ neurons: 4, activation: 'relu' }],
    });
    // Train with some extreme values
    const snap = nn.trainBatch(
      [[1e10, -1e10, 0, 0], [0, 0, 1e10, -1e10]],
      [3, 7]
    );
    // Weights should still be finite
    for (const layer of snap.layers) {
      for (const row of layer.weights) {
        for (const w of row) {
          expect(isFinite(w)).toBe(true);
        }
      }
      for (const b of layer.biases) {
        expect(isFinite(b)).toBe(true);
      }
    }
  });

  it('should handle all-zero inputs gracefully', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const result = nn.forward(new Array(784).fill(0));
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.001);
  });

  it('should handle all-ones inputs gracefully', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const result = nn.forward(new Array(784).fill(1));
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
  });
});

// ─── Training Edge Cases ──────────────────────────────────────────
describe('Training edge cases', () => {
  it('should handle single-sample training batch', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const snap = nn.trainBatch([[0.5, 0.3, 0.1, 0.9]], [5]);
    expect(snap.epoch).toBe(1);
    expect(isFinite(snap.loss)).toBe(true);
    expect(snap.accuracy).toBeGreaterThanOrEqual(0);
    expect(snap.accuracy).toBeLessThanOrEqual(1);
  });

  it('should handle label at boundary (0 and 9)', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const snap0 = nn.trainBatch([[0.1, 0.2, 0.3, 0.4]], [0]);
    expect(isFinite(snap0.loss)).toBe(true);
    const snap9 = nn.trainBatch([[0.4, 0.3, 0.2, 0.1]], [9]);
    expect(isFinite(snap9.loss)).toBe(true);
  });

  it('should handle very high learning rate without crashing', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 1.0, // Insanely high
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    // Run 10 epochs — weights may explode but should not produce NaN
    for (let i = 0; i < 10; i++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss) || snap.loss === 10).toBe(true); // capped degenerate loss
      expect(snap.epoch).toBe(i + 1);
    }
  });

  it('should handle very low learning rate (near zero)', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 1e-12,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const snap = nn.trainBatch([[0.5, 0.5, 0.5, 0.5]], [3]);
    expect(isFinite(snap.loss)).toBe(true);
  });

  it('should handle repeated identical samples', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const inputs = Array.from({ length: 20 }, () => [0.5, 0.5, 0.5, 0.5]);
    const labels = Array.from({ length: 20 }, () => 3);
    const snap = nn.trainBatch(inputs, labels);
    expect(isFinite(snap.loss)).toBe(true);
  });
});

// ─── Activation Function Edge Cases ──────────────────────────────
describe('Activation function edge cases', () => {
  const activations: ActivationFn[] = ['relu', 'sigmoid', 'tanh'];

  for (const act of activations) {
    it(`should handle ${act} with extreme positive input`, () => {
      const nn = new NeuralNetwork(4, {
        learningRate: 0.01,
        layers: [{ neurons: 4, activation: act }],
      });
      const result = nn.forward([1e10, 1e10, 1e10, 1e10]);
      expect(result.every(v => isFinite(v))).toBe(true);
    });

    it(`should handle ${act} with extreme negative input`, () => {
      const nn = new NeuralNetwork(4, {
        learningRate: 0.01,
        layers: [{ neurons: 4, activation: act }],
      });
      const result = nn.forward([-1e10, -1e10, -1e10, -1e10]);
      expect(result.every(v => isFinite(v))).toBe(true);
    });
  }

  it('should work with mixed activation functions across layers', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 32, activation: 'relu' },
        { neurons: 16, activation: 'sigmoid' },
        { neurons: 8, activation: 'tanh' },
      ],
    });
    const result = nn.forward(new Array(784).fill(0.5));
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v))).toBe(true);
  });
});

// ─── Architecture Edge Cases ─────────────────────────────────────
describe('Architecture edge cases', () => {
  it('should work with single neuron per hidden layer', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 1, activation: 'relu' }],
    });
    const result = nn.forward(new Array(784).fill(0.5));
    expect(result.length).toBe(10);
  });

  it('should work with 5 hidden layers (max UI allows)', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 128, activation: 'relu' },
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
        { neurons: 16, activation: 'relu' },
        { neurons: 8, activation: 'relu' },
      ],
    });
    const result = nn.forward(new Array(784).fill(0.3));
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v))).toBe(true);
  });

  it('should work with 256 neurons (max UI option)', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 256, activation: 'relu' }],
    });
    const result = nn.forward(new Array(784).fill(0.5));
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v))).toBe(true);
  });

  it('should reset cleanly and produce new outputs', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);
    expect(nn.getEpoch()).toBe(1);
    
    nn.reset(784);
    expect(nn.getEpoch()).toBe(0);
    expect(nn.getLossHistory()).toEqual([]);
    expect(nn.getAccuracyHistory()).toEqual([]);
  });

  it('should reset with new config', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    nn.reset(784, {
      learningRate: 0.05,
      layers: [{ neurons: 64, activation: 'tanh' }],
    });
    const result = nn.forward(new Array(784).fill(0.5));
    expect(result.length).toBe(10);
  });
});

// ─── Predict Consistency ─────────────────────────────────────────
describe('Predict consistency', () => {
  it('should return label matching highest probability', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const result = nn.predict(new Array(784).fill(0.5));
    const maxProb = Math.max(...result.probabilities);
    expect(result.label).toBe(result.probabilities.indexOf(maxProb));
  });

  it('should return valid probability distribution (sums to ~1)', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    });
    const result = nn.predict(new Array(784).fill(0.5));
    const sum = result.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.01);
  });

  it('should return deep-copied layers (mutation safe)', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const result1 = nn.predict(new Array(784).fill(0.5));
    const result2 = nn.predict(new Array(784).fill(0.5));
    // Mutating result1 should not affect result2
    result1.layers[0].weights[0][0] = 999;
    expect(result2.layers[0].weights[0][0]).not.toBe(999);
  });
});

// ─── canvasToInput Edge Cases ────────────────────────────────────
describe('canvasToInput edge cases', () => {
  it('should handle 1x1 image', () => {
    const data = new Uint8ClampedArray([255, 255, 255, 255]);
    const imageData = { width: 1, height: 1, data, colorSpace: 'srgb' } as unknown as ImageData;
    const result = canvasToInput(imageData, 28);
    expect(result.length).toBe(784);
    // All cells should be 0 except possibly (0,0) area
  });

  it('should handle very large image (1000x1000)', () => {
    const data = new Uint8ClampedArray(1000 * 1000 * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 128; data[i + 1] = 128; data[i + 2] = 128; data[i + 3] = 255;
    }
    const imageData = { width: 1000, height: 1000, data, colorSpace: 'srgb' } as unknown as ImageData;
    const result = canvasToInput(imageData, 28);
    expect(result.length).toBe(784);
    // Should be roughly 0.502 (128/255) everywhere
    for (const v of result) {
      expect(v).toBeGreaterThan(0.4);
      expect(v).toBeLessThan(0.6);
    }
  });

  it('should handle non-square image', () => {
    const w = 400, h = 200;
    const data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 200; data[i + 1] = 200; data[i + 2] = 200; data[i + 3] = 255;
    }
    const imageData = { width: w, height: h, data, colorSpace: 'srgb' } as unknown as ImageData;
    const result = canvasToInput(imageData);
    expect(result.length).toBe(784);
    expect(result.every(v => v >= 0 && v <= 1)).toBe(true);
  });
});

// ─── Training Data Generation ────────────────────────────────────
describe('Training data stress', () => {
  it('should generate 1 sample per digit with samplesPerDigit=1', () => {
    const data = generateTrainingData(1);
    expect(data.inputs.length).toBe(10);
    expect(data.labels.length).toBe(10);
    for (let d = 0; d < 10; d++) {
      expect(data.labels).toContain(d);
    }
  });

  it('should generate 100 samples per digit without error', () => {
    const data = generateTrainingData(100);
    expect(data.inputs.length).toBe(1000);
    expect(data.labels.length).toBe(1000);
  });

  it('should produce unique patterns (no two identical)', () => {
    const data = generateTrainingData(5);
    for (let i = 0; i < data.inputs.length; i++) {
      for (let j = i + 1; j < data.inputs.length; j++) {
        // At least some pixels should differ
        const diff = data.inputs[i].some((v, k) => Math.abs(v - data.inputs[j][k]) > 0.001);
        expect(diff).toBe(true);
      }
    }
  });
});

// ─── Softmax Degenerate Cases ────────────────────────────────────
describe('Softmax degenerate cases', () => {
  it('should handle uniform pre-activations', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 4, activation: 'relu' }],
    });
    // This tests internal softmax with similar values
    const result = nn.forward([0, 0, 0, 0]);
    expect(result.length).toBe(10);
    expect(result.every(v => isFinite(v) && v >= 0)).toBe(true);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.01);
  });
});

// ─── Loss History Immutability ───────────────────────────────────
describe('Loss history immutability', () => {
  it('should return defensive copies of loss and accuracy history', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    nn.trainBatch([[0.1, 0.2, 0.3, 0.4]], [3]);
    
    const history1 = nn.getLossHistory();
    history1[0] = 999;
    const history2 = nn.getLossHistory();
    expect(history2[0]).not.toBe(999);
  });

  it('should accumulate loss history across epochs', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    for (let i = 0; i < 5; i++) {
      nn.trainBatch([[0.1, 0.2, 0.3, 0.4]], [3]);
    }
    expect(nn.getLossHistory().length).toBe(5);
    expect(nn.getAccuracyHistory().length).toBe(5);
  });
});

// ─── Large-Scale Stability ───────────────────────────────────────
describe('Large-scale stability', () => {
  it('should survive 200 training epochs without NaN', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
      ],
    });
    const data = generateTrainingData(10);
    for (let i = 0; i < 200; i++) {
      const snap = nn.trainBatch(data.inputs, data.labels);
      expect(isFinite(snap.loss)).toBe(true);
      expect(isFinite(snap.accuracy)).toBe(true);
      expect(snap.accuracy).toBeGreaterThanOrEqual(0);
      expect(snap.accuracy).toBeLessThanOrEqual(1);
    }
    // After 200 epochs should have some learning
    const finalAcc = nn.getAccuracyHistory()[199];
    expect(finalAcc).toBeGreaterThan(0);
  });

  it('should handle rapid reset/train cycles', () => {
    const nn = new NeuralNetwork(784, {
      learningRate: 0.01,
      layers: [{ neurons: 16, activation: 'relu' }],
    });
    const data = generateTrainingData(5);
    for (let cycle = 0; cycle < 10; cycle++) {
      nn.reset(784);
      for (let ep = 0; ep < 5; ep++) {
        const snap = nn.trainBatch(data.inputs, data.labels);
        expect(isFinite(snap.loss)).toBe(true);
      }
    }
    expect(nn.getEpoch()).toBe(5);
  });
});

// ─── Type System Completeness ────────────────────────────────────
describe('Type system completeness', () => {
  it('TrainingSnapshot should have all required fields', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 4, activation: 'relu' }],
    });
    const snap = nn.trainBatch([[0.1, 0.2, 0.3, 0.4]], [5]);
    expect(snap).toHaveProperty('epoch');
    expect(snap).toHaveProperty('loss');
    expect(snap).toHaveProperty('accuracy');
    expect(snap).toHaveProperty('layers');
    expect(snap).toHaveProperty('predictions');
    expect(snap).toHaveProperty('outputProbabilities');
    expect(typeof snap.epoch).toBe('number');
    expect(typeof snap.loss).toBe('number');
    expect(typeof snap.accuracy).toBe('number');
    expect(Array.isArray(snap.layers)).toBe(true);
    expect(Array.isArray(snap.predictions)).toBe(true);
    expect(Array.isArray(snap.outputProbabilities)).toBe(true);
  });

  it('LayerState should have weights, biases, preActivations, activations', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const result = nn.predict([0.1, 0.2, 0.3, 0.4]);
    for (const layer of result.layers) {
      expect(layer).toHaveProperty('weights');
      expect(layer).toHaveProperty('biases');
      expect(layer).toHaveProperty('preActivations');
      expect(layer).toHaveProperty('activations');
      expect(Array.isArray(layer.weights)).toBe(true);
      expect(Array.isArray(layer.biases)).toBe(true);
    }
  });

  it('predict result should have label, probabilities, layers', () => {
    const nn = new NeuralNetwork(4, {
      learningRate: 0.01,
      layers: [{ neurons: 8, activation: 'relu' }],
    });
    const result = nn.predict([0.1, 0.2, 0.3, 0.4]);
    expect(result).toHaveProperty('label');
    expect(result).toHaveProperty('probabilities');
    expect(result).toHaveProperty('layers');
    expect(typeof result.label).toBe('number');
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThanOrEqual(9);
    expect(result.probabilities.length).toBe(10);
  });
});
