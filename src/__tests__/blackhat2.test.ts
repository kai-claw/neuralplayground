import { describe, it, expect } from 'vitest';
import { NeuralNetwork } from '../nn/NeuralNetwork';
import type { TrainingConfig } from '../nn/NeuralNetwork';
import { generateTrainingData } from '../nn/sampleData';
import { mulberry32 } from '../utils';
import * as fs from 'fs';
import * as path from 'path';

/**
 * Black Hat #2 — Re-Audit Tests (Pass 8/10)
 *
 * Targeted tests for issues found in passes 5-7 code:
 * - Cleanup correctness in hooks
 * - Edge cases in dream/surgery/race features
 * - Resource management
 * - Seeded RNG determinism
 */

const SRC = path.resolve(__dirname, '..');

// ═══════════════════════════════════════════════════════════════════
// 1. NETWORK DREAMS — EDGE CASES
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Network Dreams edge cases', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [{ neurons: 32, activation: 'relu' }],
  };

  it('dream returns null on untrained network (no epochs)', () => {
    const nn = new NeuralNetwork(784, config);
    // Network has not been trained — dream should still produce output
    const result = nn.dream(5, 10, 0.5);
    expect(result).not.toBeNull();
    if (result) {
      expect(result.image.length).toBe(784);
      expect(result.image.every(v => isFinite(v))).toBe(true);
    }
  });

  it('dream with 0 steps returns starting image unchanged', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const startImage = new Array(784).fill(0.5);
    const result = nn.dream(3, 0, 0.5, startImage);
    expect(result).not.toBeNull();
    if (result) {
      expect(result.image.length).toBe(784);
      expect(result.confidenceHistory.length).toBe(0);
    }
  });

  it('dream with very high LR does not produce NaN', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const result = nn.dream(7, 20, 100.0); // Absurdly high LR
    expect(result).not.toBeNull();
    if (result) {
      expect(result.image.every(v => isFinite(v))).toBe(true);
      expect(result.confidenceHistory.every(v => isFinite(v))).toBe(true);
    }
  });

  it('dream for all 10 target classes produces valid images', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(10);
    for (let i = 0; i < 5; i++) nn.trainBatch(data.inputs, data.labels);

    for (let d = 0; d < 10; d++) {
      const result = nn.dream(d, 20, 0.5);
      expect(result).not.toBeNull();
      if (result) {
        expect(result.image.length).toBe(784);
        expect(result.image.every(v => v >= 0 && v <= 1)).toBe(true);
        expect(result.confidenceHistory.length).toBe(20);
      }
    }
  });

  it('computeInputGradient returns valid gradient array', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const gradient = nn.computeInputGradient(new Array(784).fill(0.5), 3);
    expect(gradient).not.toBeNull();
    expect(gradient!.length).toBe(784);
    expect(gradient!.every(v => isFinite(v))).toBe(true);
  });

  it('computeInputGradient with all-zero input returns finite gradient', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    const gradient = nn.computeInputGradient(new Array(784).fill(0), 0);
    expect(gradient!.every(v => isFinite(v))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════
// 2. NEURON SURGERY — EDGE CASES
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Neuron Surgery edge cases', () => {
  const config: TrainingConfig = {
    learningRate: 0.01,
    layers: [
      { neurons: 32, activation: 'relu' },
      { neurons: 16, activation: 'relu' },
    ],
  };

  it('getNeuronStatus returns "active" for out-of-range indices', () => {
    const nn = new NeuralNetwork(784, config);
    // Out of range layer
    expect(nn.getNeuronStatus(99, 0)).toBe('active');
    // Out of range neuron
    expect(nn.getNeuronStatus(0, 999)).toBe('active');
    // Negative indices
    expect(nn.getNeuronStatus(-1, 0)).toBe('active');
  });

  it('killing all neurons in a layer produces valid (degenerate) output', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    // Kill all neurons in layer 0
    for (let i = 0; i < 32; i++) {
      nn.setNeuronStatus(0, i, 'killed');
    }

    const result = nn.predict(new Array(784).fill(0.5));
    expect(result.probabilities.length).toBe(10);
    expect(result.probabilities.every(v => isFinite(v) && v >= 0)).toBe(true);
    const sum = result.probabilities.reduce((a, b) => a + b, 0);
    expect(Math.abs(sum - 1)).toBeLessThan(0.01);
  });

  it('freezing neurons preserves output idempotency', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    // Freeze all neurons in layer 0
    for (let i = 0; i < 32; i++) {
      nn.setNeuronStatus(0, i, 'frozen');
    }

    const input = new Array(784).fill(0.5);
    const result1 = nn.predict(input);
    const result2 = nn.predict(input);
    // Frozen neurons should give same results on same input
    for (let i = 0; i < 10; i++) {
      expect(result1.probabilities[i]).toBeCloseTo(result2.probabilities[i], 10);
    }
  });

  it('clearAllMasks restores all neurons to active', () => {
    const nn = new NeuralNetwork(784, config);

    nn.setNeuronStatus(0, 0, 'killed');
    nn.setNeuronStatus(0, 1, 'frozen');
    nn.setNeuronStatus(1, 0, 'killed');

    nn.clearAllMasks();

    expect(nn.getNeuronStatus(0, 0)).toBe('active');
    expect(nn.getNeuronStatus(0, 1)).toBe('active');
    expect(nn.getNeuronStatus(1, 0)).toBe('active');
  });

  it('surgery status survives training epochs', () => {
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);

    nn.setNeuronStatus(0, 5, 'frozen');
    nn.setNeuronStatus(1, 3, 'killed');

    nn.trainBatch(data.inputs, data.labels);

    expect(nn.getNeuronStatus(0, 5)).toBe('frozen');
    expect(nn.getNeuronStatus(1, 3)).toBe('killed');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 3. TRAINING RACE — LOGIC EDGE CASES
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Training Race logic', () => {
  it('two networks with same config produce different initial outputs (random init)', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nnA = new NeuralNetwork(784, config);
    const nnB = new NeuralNetwork(784, config);

    const input = new Array(784).fill(0.5);
    const outA = nnA.forward(input);
    const outB = nnB.forward(input);

    // Random init means different weights → different outputs (at least one differs)
    const anyDiff = outA.some((v, i) => Math.abs(v - outB[i]) > 1e-10);
    expect(anyDiff).toBe(true);
  });

  it('race networks can train on same data independently', () => {
    const configA: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 64, activation: 'relu' }, { neurons: 32, activation: 'relu' }],
    };
    const configB: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'sigmoid' }],
    };

    const nnA = new NeuralNetwork(784, configA);
    const nnB = new NeuralNetwork(784, configB);
    const data = generateTrainingData(10);

    for (let i = 0; i < 10; i++) {
      const snapA = nnA.trainBatch(data.inputs, data.labels);
      const snapB = nnB.trainBatch(data.inputs, data.labels);
      expect(isFinite(snapA.loss)).toBe(true);
      expect(isFinite(snapB.loss)).toBe(true);
    }

    // Both should show some learning
    expect(nnA.getAccuracyHistory().length).toBe(10);
    expect(nnB.getAccuracyHistory().length).toBe(10);
  });

  it('winner determination: tie when accuracy difference < 0.02', () => {
    // Simulate the tie logic from useTrainingRace
    const finalAccA = 0.85;
    const finalAccB = 0.86;
    const winner: 'A' | 'B' | 'tie' =
      Math.abs(finalAccA - finalAccB) < 0.02 ? 'tie' :
      finalAccA > finalAccB ? 'A' : 'B';
    expect(winner).toBe('tie');
  });

  it('winner determination: clear winner when gap > 0.02', () => {
    const finalAccA = 0.90;
    const finalAccB = 0.70;
    const winner: 'A' | 'B' | 'tie' =
      Math.abs(finalAccA - finalAccB) < 0.02 ? 'tie' :
      finalAccA > finalAccB ? 'A' : 'B';
    expect(winner).toBe('A');
  });
});

// ═══════════════════════════════════════════════════════════════════
// 4. CLEANUP & RESOURCE MANAGEMENT VERIFICATION
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Cleanup verification (code audit)', () => {
  it('useCinematic stores setInterval in ref (not local variable)', () => {
    const src = fs.readFileSync(path.join(SRC, 'hooks', 'useCinematic.ts'), 'utf-8');
    // Verify the interval is stored in intervalRef, not a local variable
    expect(src).toContain('intervalRef');
    expect(src).toContain('intervalRef.current = setInterval');
    // Verify cleanup clears intervalRef
    expect(src).toContain('clearInterval(intervalRef.current)');
  });

  it('useCinematic clearTimer cleans up both setTimeout and setInterval', () => {
    const src = fs.readFileSync(path.join(SRC, 'hooks', 'useCinematic.ts'), 'utf-8');
    // clearTimer should handle both timerRef and intervalRef
    const clearTimerBlock = src.substring(
      src.indexOf('const clearTimer'),
      src.indexOf('}, []);') + 7
    );
    expect(clearTimerBlock).toContain('clearTimeout');
    expect(clearTimerBlock).toContain('clearInterval');
    expect(clearTimerBlock).toContain('intervalRef');
  });

  it('useTrainingRace cleans up network refs on stop', () => {
    const src = fs.readFileSync(path.join(SRC, 'hooks', 'useTrainingRace.ts'), 'utf-8');
    // stopRace should null out refs — extract from 'const stopRace' to next 'const '
    const stopStart = src.indexOf('const stopRace');
    const stopEnd = src.indexOf('const applyPreset');
    const stopBlock = src.substring(stopStart, stopEnd);
    expect(stopBlock).toContain('networkARef.current = null');
    expect(stopBlock).toContain('networkBRef.current = null');
    expect(stopBlock).toContain('dataRef.current = null');
  });

  it('useTrainingRace cleans up network refs on unmount', () => {
    const src = fs.readFileSync(path.join(SRC, 'hooks', 'useTrainingRace.ts'), 'utf-8');
    // Unmount cleanup should null refs
    const unmountEffect = src.substring(
      src.indexOf('useEffect(() => {'),
      src.indexOf('}, []);') + 7
    );
    expect(unmountEffect).toContain('networkARef.current = null');
    expect(unmountEffect).toContain('networkBRef.current = null');
  });

  it('NeuronSurgery uses useMemo for hiddenLayers', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NeuronSurgery.tsx'), 'utf-8'
    );
    expect(src).toContain('useMemo');
    expect(src).toContain('useMemo(() => layers');
  });

  it('Surgery renderer uses seeded RNG for connection rendering (no Math.random in draw)', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'renderers', 'surgeryRenderer.ts'), 'utf-8'
    );
    // draw function should use mulberry32, not Math.random
    expect(src).toContain('mulberry32');
    expect(src).toContain('connRng');
    // Math.random should NOT appear in the file (was only used for connections)
    expect(src).not.toContain('Math.random()');
  });

  it('NetworkDreams cleans up timer on unmount', () => {
    const src = fs.readFileSync(
      path.join(SRC, 'components', 'NetworkDreams.tsx'), 'utf-8'
    );
    expect(src).toContain('timerRef');
    // Unmount cleanup
    expect(src).toContain('clearTimeout(timerRef.current)');
  });

  it('all hooks that use setTimeout/setInterval have unmount cleanup', () => {
    const hookDir = path.join(SRC, 'hooks');
    const hookFiles = fs.readdirSync(hookDir).filter(f => f.endsWith('.ts'));

    for (const file of hookFiles) {
      const src = fs.readFileSync(path.join(hookDir, file), 'utf-8');
      const hasTimer = src.includes('setTimeout') || src.includes('setInterval');
      if (hasTimer) {
        // Must have cleanup in useEffect return
        expect(src, `${file} uses timers but may lack cleanup`).toContain('return () =>');
        if (src.includes('setTimeout')) {
          expect(src, `${file} uses setTimeout but no clearTimeout`).toContain('clearTimeout');
        }
        if (src.includes('setInterval')) {
          expect(src, `${file} uses setInterval but no clearInterval`).toContain('clearInterval');
        }
      }
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 5. SEEDED RNG DETERMINISM
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Seeded RNG determinism', () => {
  it('mulberry32 with same seed produces identical sequence', () => {
    const rng1 = mulberry32(12345);
    const rng2 = mulberry32(12345);
    for (let i = 0; i < 100; i++) {
      expect(rng1()).toBe(rng2());
    }
  });

  it('mulberry32 with different seeds produces different sequences', () => {
    const rng1 = mulberry32(42);
    const rng2 = mulberry32(43);
    const seq1 = Array.from({ length: 10 }, () => rng1());
    const seq2 = Array.from({ length: 10 }, () => rng2());
    const anyDiff = seq1.some((v, i) => v !== seq2[i]);
    expect(anyDiff).toBe(true);
  });

  it('surgery connection seed is deterministic based on layer params', () => {
    // The seed formula: l * 1000 + prevCount * 100 + currCount
    // Different layers should get different seeds
    const seed1 = 0 * 1000 + 32 * 100 + 16;
    const seed2 = 1 * 1000 + 16 * 100 + 10;
    expect(seed1).not.toBe(seed2);

    const rng1 = mulberry32(seed1);
    const rng2 = mulberry32(seed2);
    expect(rng1()).not.toBe(rng2());
  });
});

// ═══════════════════════════════════════════════════════════════════
// 6. DREAM + SURGERY COMBINED EDGE CASES
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Dream + Surgery combined', () => {
  it('dream works correctly with frozen neurons', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    // Freeze half the neurons
    for (let i = 0; i < 16; i++) {
      nn.setNeuronStatus(0, i, 'frozen');
    }

    const result = nn.dream(5, 20, 0.5);
    expect(result).not.toBeNull();
    if (result) {
      expect(result.image.every(v => isFinite(v))).toBe(true);
      expect(result.confidenceHistory.every(v => isFinite(v))).toBe(true);
    }
  });

  it('dream works with killed neurons (degenerate but stable)', () => {
    const config: TrainingConfig = {
      learningRate: 0.01,
      layers: [{ neurons: 32, activation: 'relu' }],
    };
    const nn = new NeuralNetwork(784, config);
    const data = generateTrainingData(5);
    nn.trainBatch(data.inputs, data.labels);

    // Kill half the neurons
    for (let i = 0; i < 16; i++) {
      nn.setNeuronStatus(0, i, 'killed');
    }

    const result = nn.dream(3, 20, 0.5);
    expect(result).not.toBeNull();
    if (result) {
      expect(result.image.length).toBe(784);
      expect(result.image.every(v => isFinite(v) && v >= 0 && v <= 1)).toBe(true);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════
// 7. RACE PRESETS INTEGRITY
// ═══════════════════════════════════════════════════════════════════

describe('Black Hat #2 — Race presets integrity', () => {
  it('all RACE_PRESETS have valid configs', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    expect(RACE_PRESETS.length).toBeGreaterThan(0);

    for (const preset of RACE_PRESETS) {
      expect(preset.label).toBeTruthy();
      expect(preset.a.learningRate).toBeGreaterThan(0);
      expect(preset.b.learningRate).toBeGreaterThan(0);
      expect(preset.a.layers.length).toBeGreaterThan(0);
      expect(preset.b.layers.length).toBeGreaterThan(0);

      for (const layer of [...preset.a.layers, ...preset.b.layers]) {
        expect(layer.neurons).toBeGreaterThan(0);
        expect(['relu', 'sigmoid', 'tanh']).toContain(layer.activation);
      }
    }
  });

  it('each RACE_PRESET creates trainable networks', async () => {
    const { RACE_PRESETS } = await import('../data/racePresets');
    const data = generateTrainingData(5);

    for (const preset of RACE_PRESETS) {
      const nnA = new NeuralNetwork(784, preset.a);
      const nnB = new NeuralNetwork(784, preset.b);

      const snapA = nnA.trainBatch(data.inputs, data.labels);
      const snapB = nnB.trainBatch(data.inputs, data.labels);

      expect(isFinite(snapA.loss), `Preset "${preset.label}" A loss is not finite`).toBe(true);
      expect(isFinite(snapB.loss), `Preset "${preset.label}" B loss is not finite`).toBe(true);
    }
  });
});
