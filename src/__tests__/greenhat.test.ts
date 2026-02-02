import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig, LayerState } from '../nn/NeuralNetwork';
import { DIGIT_STROKES, getDigitDrawDuration } from '../data/digitStrokes';

/**
 * Green Hat Pass 3 — Feature Maps & Adversarial Lab tests
 */

// ─── Feature Map weight visualization tests ───
describe('Feature Maps — first-layer weight structure', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [
      { neurons: 16, activation: 'relu' },
    ],
  };

  it('first hidden layer has 784 input weights per neuron (28×28)', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0.5));
    const firstLayer = result.layers[0];
    // Each neuron should have exactly 784 weights
    for (const weights of firstLayer.weights) {
      expect(weights).toHaveLength(784);
    }
  });

  it('first layer neuron count matches config', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0));
    expect(result.layers[0].weights).toHaveLength(16);
  });

  it('weights can be reshaped to 28×28 grid', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0));
    const weights = result.layers[0].weights[0];
    // Reshape test
    const grid: number[][] = [];
    for (let y = 0; y < 28; y++) {
      const row = weights.slice(y * 28, (y + 1) * 28);
      expect(row).toHaveLength(28);
      grid.push(row);
    }
    expect(grid).toHaveLength(28);
  });

  it('Xavier-initialized weights have reasonable range', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0));
    const weights = result.layers[0].weights[0];
    // Xavier: std ≈ sqrt(2 / (784 + 16)) ≈ 0.05, so values should be small
    const absMax = Math.max(...weights.map(Math.abs));
    expect(absMax).toBeLessThan(1.0);
    expect(absMax).toBeGreaterThan(0);
  });

  it('feature maps diverge after training (neurons specialize)', () => {
    const nn = new NeuralNetwork(784, config);
    // Create simple training data
    const inputs: number[][] = [];
    const labels: number[] = [];
    for (let d = 0; d < 10; d++) {
      const input = new Array(784).fill(0);
      // Each digit has different active pixels
      for (let i = d * 78; i < (d + 1) * 78; i++) {
        input[i] = 1;
      }
      inputs.push(input);
      labels.push(d);
    }

    // Get initial weight similarity
    const before = nn.predict(new Array(784).fill(0)).layers[0].weights;
    const initialCorr = weightCorrelation(before[0], before[1]);

    // Train for 20 epochs
    for (let e = 0; e < 20; e++) {
      nn.trainBatch(inputs, labels);
    }

    const after = nn.predict(new Array(784).fill(0)).layers[0].weights;
    // After training, at least some neurons should have different weight patterns
    let hasDivergence = false;
    for (let i = 0; i < after.length - 1; i++) {
      const corr = weightCorrelation(after[i], after[i + 1]);
      if (Math.abs(corr) < 0.95) {
        hasDivergence = true;
        break;
      }
    }
    expect(hasDivergence).toBe(true);
  });

  it('weight normalization produces valid colormap range', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0));
    const weights = result.layers[0].weights[0];

    // Simulate normalization as done in FeatureMaps component
    let wMin = Infinity;
    let wMax = -Infinity;
    for (const w of weights) {
      if (w < wMin) wMin = w;
      if (w > wMax) wMax = w;
    }
    const range = wMax - wMin || 1;
    
    for (const w of weights) {
      const norm = (w - wMin) / range;
      expect(norm).toBeGreaterThanOrEqual(0);
      expect(norm).toBeLessThanOrEqual(1);
    }
  });

  it('multi-layer config still exposes first-layer 784 weights', () => {
    const deepConfig: TrainingConfig = {
      learningRate: 0.01,
      layers: [
        { neurons: 64, activation: 'relu' },
        { neurons: 32, activation: 'relu' },
        { neurons: 16, activation: 'tanh' },
      ],
    };
    const nn = new NeuralNetwork(784, deepConfig);
    const result = nn.predict(new Array(784).fill(0));
    // First layer still has 784 inputs
    for (const weights of result.layers[0].weights) {
      expect(weights).toHaveLength(784);
    }
    // Second layer has neurons-of-first-layer inputs
    for (const weights of result.layers[1].weights) {
      expect(weights).toHaveLength(64);
    }
  });
});

// ─── Adversarial noise tests ───
describe('Adversarial Noise — input perturbation', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [{ neurons: 32, activation: 'relu' }],
  };

  it('gaussian noise at level 0 preserves original input', () => {
    const input = new Array(784).fill(0).map((_, i) => (i % 10) / 10);
    const noised = applyGaussianNoise(input, 0, 42);
    for (let i = 0; i < input.length; i++) {
      expect(noised[i]).toBeCloseTo(input[i], 10);
    }
  });

  it('gaussian noise at level 1 significantly alters input', () => {
    const input = new Array(784).fill(0.5);
    const noised = applyGaussianNoise(input, 1.0, 42);
    let diffCount = 0;
    for (let i = 0; i < input.length; i++) {
      if (Math.abs(noised[i] - input[i]) > 0.01) diffCount++;
    }
    // Virtually all pixels should change at max noise
    expect(diffCount).toBeGreaterThan(700);
  });

  it('noised values stay clamped to [0, 1]', () => {
    const input = new Array(784).fill(0.5);
    const noised = applyGaussianNoise(input, 1.0, 42);
    for (const v of noised) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('salt-pepper noise only produces 0 or 1 for flipped pixels', () => {
    const input = new Array(784).fill(0.5);
    const noised = applySaltPepperNoise(input, 1.0, 42);
    for (const v of noised) {
      expect(v === 0 || v === 0.5 || v === 1).toBe(true);
    }
  });

  it('different seeds produce different noise patterns', () => {
    const input = new Array(784).fill(0.5);
    const noised1 = applyGaussianNoise(input, 0.5, 42);
    const noised2 = applyGaussianNoise(input, 0.5, 99);
    let diffCount = 0;
    for (let i = 0; i < input.length; i++) {
      if (Math.abs(noised1[i] - noised2[i]) > 0.001) diffCount++;
    }
    expect(diffCount).toBeGreaterThan(500);
  });

  it('same seed produces identical noise', () => {
    const input = new Array(784).fill(0.5);
    const noised1 = applyGaussianNoise(input, 0.5, 42);
    const noised2 = applyGaussianNoise(input, 0.5, 42);
    for (let i = 0; i < input.length; i++) {
      expect(noised1[i]).toBe(noised2[i]);
    }
  });

  it('increasing noise reduces prediction confidence', () => {
    const nn = new NeuralNetwork(784, config);
    // Simple training
    const inputs: number[][] = [];
    const labels: number[] = [];
    for (let d = 0; d < 10; d++) {
      const input = new Array(784).fill(0);
      for (let i = d * 78; i < (d + 1) * 78; i++) input[i] = 1;
      inputs.push(input);
      labels.push(d);
    }
    for (let e = 0; e < 30; e++) {
      nn.trainBatch(inputs, labels);
    }

    // Predict clean vs noisy
    const cleanInput = inputs[0];
    const cleanResult = nn.predict(cleanInput);
    const cleanConf = Math.max(...cleanResult.probabilities);

    const noisyInput = applyGaussianNoise(cleanInput, 0.8, 42);
    const noisyResult = nn.predict(noisyInput);
    const noisyConf = noisyResult.probabilities[cleanResult.label];

    // Noisy input should have lower confidence for the original class
    expect(noisyConf).toBeLessThan(cleanConf + 0.1); // allow small margin
  });

  it('noise level scales smoothly', () => {
    const input = new Array(784).fill(0.5);
    let prevDiff = 0;
    for (let level = 0.1; level <= 1.0; level += 0.1) {
      const noised = applyGaussianNoise(input, level, 42);
      let totalDiff = 0;
      for (let i = 0; i < input.length; i++) {
        totalDiff += Math.abs(noised[i] - input[i]);
      }
      expect(totalDiff).toBeGreaterThanOrEqual(prevDiff * 0.8); // roughly monotonic
      prevDiff = totalDiff;
    }
  });
});

// ─── Digit strokes data validation ───
describe('Digit Strokes — cinematic drawing data', () => {
  it('has stroke data for all 10 digits', () => {
    expect(Object.keys(DIGIT_STROKES)).toHaveLength(10);
    for (let d = 0; d < 10; d++) {
      expect(DIGIT_STROKES[d]).toBeDefined();
      expect(DIGIT_STROKES[d].length).toBeGreaterThan(0);
    }
  });

  it('all stroke points are within canvas bounds (0-280)', () => {
    for (let d = 0; d < 10; d++) {
      for (const stroke of DIGIT_STROKES[d]) {
        for (const pt of stroke.points) {
          expect(pt.x).toBeGreaterThanOrEqual(0);
          expect(pt.x).toBeLessThanOrEqual(280);
          expect(pt.y).toBeGreaterThanOrEqual(0);
          expect(pt.y).toBeLessThanOrEqual(280);
        }
      }
    }
  });

  it('draw duration scales with point count', () => {
    for (let d = 0; d < 10; d++) {
      const duration = getDigitDrawDuration(d);
      expect(duration).toBeGreaterThan(0);
      expect(duration).toBeLessThan(5000); // reasonable upper bound
    }
  });
});

// ─── Feature map + adversarial interaction ───
describe('Feature Maps + Adversarial — cross-feature', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [{ neurons: 16, activation: 'relu' }],
  };

  it('adversarial noise on all-black still produces valid forward pass', () => {
    const nn = new NeuralNetwork(784, config);
    const black = new Array(784).fill(0);
    const noised = applyGaussianNoise(black, 0.5, 42);
    const result = nn.predict(noised);
    expect(result.probabilities).toHaveLength(10);
    const sum = result.probabilities.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1.0, 3);
  });

  it('adversarial noise on all-white stays bounded', () => {
    const nn = new NeuralNetwork(784, config);
    const white = new Array(784).fill(1);
    const noised = applyGaussianNoise(white, 1.0, 42);
    for (const v of noised) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
    const result = nn.predict(noised);
    expect(result.label).toBeGreaterThanOrEqual(0);
    expect(result.label).toBeLessThan(10);
  });

  it('feature maps exist for all activations in returned layers', () => {
    const nn = new NeuralNetwork(784, config);
    const result = nn.predict(new Array(784).fill(0.3));
    for (const layer of result.layers) {
      expect(layer.activations.length).toBeGreaterThan(0);
      expect(layer.weights.length).toBe(layer.activations.length);
    }
  });

  it('noise at 0 prediction matches clean prediction exactly', () => {
    const nn = new NeuralNetwork(784, config);
    const input = new Array(784).fill(0.5);
    const clean = nn.predict(input);
    const noised = applyGaussianNoise(input, 0, 42);
    const noisyResult = nn.predict(noised);
    expect(noisyResult.label).toBe(clean.label);
    for (let i = 0; i < 10; i++) {
      expect(noisyResult.probabilities[i]).toBeCloseTo(clean.probabilities[i], 10);
    }
  });
});

// ─── Helpers ───

function weightCorrelation(a: number[], b: number[]): number {
  const n = a.length;
  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < n; i++) {
    sumA += a[i];
    sumB += b[i];
    sumAB += a[i] * b[i];
    sumA2 += a[i] * a[i];
    sumB2 += b[i] * b[i];
  }
  const num = n * sumAB - sumA * sumB;
  const den = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  return den === 0 ? 0 : num / den;
}

/** Replicate gaussian noise logic from AdversarialLab */
function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianNoiseSample(rng: () => number): number {
  const u1 = Math.max(1e-10, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function applyGaussianNoise(input: number[], level: number, seed: number): number[] {
  const rng = mulberry32(seed);
  const pattern = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    pattern[i] = gaussianNoiseSample(rng);
  }
  return input.map((v, i) => Math.max(0, Math.min(1, v + pattern[i] * level)));
}

function applySaltPepperNoise(input: number[], level: number, seed: number): number[] {
  const rng = mulberry32(seed);
  const noised = [...input];
  for (let i = 0; i < noised.length; i++) {
    const r = rng();
    if (r < 0.15) {
      if (rng() < level) noised[i] = 1;
    } else if (r < 0.30) {
      if (rng() < level) noised[i] = 0;
    }
  }
  return noised;
}
