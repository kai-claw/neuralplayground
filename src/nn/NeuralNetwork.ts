/**
 * NeuralNetwork — core feedforward neural network with backpropagation.
 *
 * Responsibilities:
 *   - Weight initialization (Xavier/Glorot)
 *   - Forward pass with configurable activations
 *   - Backward pass (SGD)
 *   - Neuron surgery (freeze/kill individual neurons)
 *   - Batch training with shuffled mini-batches
 *   - Prediction with full layer snapshot
 *
 * Dream/gradient-ascent functionality is in nn/dreams.ts.
 */

import type {
  ActivationFn,
  NeuronStatus,
  TrainingConfig,
  LayerState,
  TrainingSnapshot,
  DreamResult,
} from '../types';
import {
  activate,
  activateDerivative,
  xavierInit,
  argmax,
} from '../utils';
import {
  computeInputGradient as _computeInputGradient,
  dream as _dream,
} from './dreams';

export class NeuralNetwork {
  private layers: LayerState[] = [];
  private config: TrainingConfig;
  private epoch = 0;
  private lossHistory: number[] = [];
  private accuracyHistory: number[] = [];
  private neuronMasks: Map<string, NeuronStatus> = new Map();

  // ─── Pre-allocated scratch arrays (avoid per-call GC) ───────────
  private _scratchTarget: number[] = new Array(10).fill(0);
  private _scratchDeltas: number[] = [];
  private _scratchNewDeltas: number[] = [];
  private _scratchIndices: number[] = [];

  constructor(inputSize: number, config: TrainingConfig) {
    this.config = config;
    this.initializeWeights(inputSize);
  }

  // ─── Neuron Surgery API ──────────────────────────────────────────

  setNeuronStatus(layerIdx: number, neuronIdx: number, status: NeuronStatus): void {
    const key = `${layerIdx}-${neuronIdx}`;
    if (status === 'active') {
      this.neuronMasks.delete(key);
    } else {
      this.neuronMasks.set(key, status);
    }
  }

  getNeuronStatus(layerIdx: number, neuronIdx: number): NeuronStatus {
    return this.neuronMasks.get(`${layerIdx}-${neuronIdx}`) || 'active';
  }

  getAllNeuronStatuses(): Map<string, NeuronStatus> {
    return new Map(this.neuronMasks);
  }

  clearAllMasks(): void {
    this.neuronMasks.clear();
  }

  getConfig(): TrainingConfig {
    return this.config;
  }

  // ─── Layer access (for dreams module) ────────────────────────────

  getLayers(): LayerState[] {
    return this.layers;
  }

  getNeuronMasks(): Map<string, NeuronStatus> {
    return this.neuronMasks;
  }

  // ─── Weight initialization ───────────────────────────────────────

  private initializeWeights(inputSize: number) {
    this.layers = [];
    let prevSize = inputSize;
    
    for (const layerConfig of this.config.layers) {
      const weights: number[][] = [];
      const biases: number[] = new Array(layerConfig.neurons).fill(0);
      
      for (let j = 0; j < layerConfig.neurons; j++) {
        const neuronWeights: number[] = [];
        for (let i = 0; i < prevSize; i++) {
          neuronWeights.push(xavierInit(prevSize, layerConfig.neurons));
        }
        weights.push(neuronWeights);
      }
      
      this.layers.push({
        weights,
        biases,
        preActivations: new Array(layerConfig.neurons).fill(0),
        activations: new Array(layerConfig.neurons).fill(0),
      });
      
      prevSize = layerConfig.neurons;
    }

    const outputWeights: number[][] = [];
    const outputBiases: number[] = new Array(10).fill(0);
    for (let j = 0; j < 10; j++) {
      const neuronWeights: number[] = [];
      for (let i = 0; i < prevSize; i++) {
        neuronWeights.push(xavierInit(prevSize, 10));
      }
      outputWeights.push(neuronWeights);
    }
    this.layers.push({
      weights: outputWeights,
      biases: outputBiases,
      preActivations: new Array(10).fill(0),
      activations: new Array(10).fill(0),
    });
  }

  // ─── Forward pass ────────────────────────────────────────────────

  forward(input: number[]): number[] {
    let current = input;
    
    for (let l = 0; l < this.layers.length; l++) {
      const layer = this.layers[l];
      const isOutput = l === this.layers.length - 1;
      const activation = isOutput ? 'sigmoid' : this.config.layers[l]?.activation || 'relu';
      const numNeurons = layer.weights.length;

      // Reuse preActivations/activations arrays in-place (avoid allocation)
      const preAct = layer.preActivations;
      const act = layer.activations;
      
      for (let j = 0; j < numNeurons; j++) {
        const wj = layer.weights[j];
        let sum = layer.biases[j];
        for (let i = 0; i < current.length; i++) {
          sum += wj[i] * current[i];
        }
        // NaN/Infinity guard
        if (!isFinite(sum)) sum = 0;
        preAct[j] = sum;
        act[j] = isOutput ? sum : activate(sum, activation as ActivationFn);
      }
      
      if (isOutput) {
        // In-place softmax to avoid allocation
        let maxVal = act[0];
        for (let i = 1; i < numNeurons; i++) {
          if (act[i] > maxVal) maxVal = act[i];
        }
        let sumExp = 0;
        for (let i = 0; i < numNeurons; i++) {
          act[i] = Math.exp(act[i] - maxVal);
          sumExp += act[i];
        }
        if (sumExp > 0 && isFinite(sumExp)) {
          for (let i = 0; i < numNeurons; i++) act[i] /= sumExp;
        } else {
          const uniform = 1 / numNeurons;
          for (let i = 0; i < numNeurons; i++) act[i] = uniform;
        }
      } else {
        // Apply neuron surgery masks
        if (this.neuronMasks.size > 0) {
          for (let j = 0; j < numNeurons; j++) {
            if (this.neuronMasks.get(`${l}-${j}`) === 'killed') {
              act[j] = 0;
            }
          }
        }
      }
      
      current = act;
    }
    
    return current;
  }

  // ─── Backward pass ───────────────────────────────────────────────

  private backward(input: number[], target: number[]): void {
    const lr = this.config.learningRate;
    const numLayers = this.layers.length;
    const hasMasks = this.neuronMasks.size > 0;
    
    const outputLayer = this.layers[numLayers - 1];
    const outputSize = outputLayer.activations.length;

    // Reuse scratch deltas array
    if (this._scratchDeltas.length < outputSize) {
      this._scratchDeltas = new Array(outputSize);
    }
    for (let i = 0; i < outputSize; i++) {
      this._scratchDeltas[i] = outputLayer.activations[i] - target[i];
    }
    let deltas = this._scratchDeltas;
    
    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevActivations = l > 0 ? this.layers[l - 1].activations : input;
      const numNeurons = layer.weights.length;
      
      for (let j = 0; j < numNeurons; j++) {
        if (hasMasks) {
          const status = this.neuronMasks.get(`${l}-${j}`);
          if (status === 'frozen' || status === 'killed') continue;
        }

        const wj = layer.weights[j];
        const delta_j = deltas[j];
        const scaledDelta = lr * delta_j;
        for (let i = 0; i < wj.length; i++) {
          const grad = scaledDelta * prevActivations[i];
          if (isFinite(grad)) {
            wj[i] -= grad;
          }
        }
        if (isFinite(scaledDelta)) {
          layer.biases[j] -= scaledDelta;
        }
      }
      
      if (l > 0) {
        const prevLayer = this.layers[l - 1];
        const activation = this.config.layers[l - 1]?.activation || 'relu';
        const prevSize = prevLayer.weights.length;

        // Reuse scratch newDeltas array
        if (this._scratchNewDeltas.length < prevSize) {
          this._scratchNewDeltas = new Array(prevSize);
        }
        for (let i = 0; i < prevSize; i++) {
          let sum = 0;
          for (let j = 0; j < numNeurons; j++) {
            sum += layer.weights[j][i] * deltas[j];
          }
          const d = sum * activateDerivative(prevLayer.preActivations[i], activation as ActivationFn);
          this._scratchNewDeltas[i] = isFinite(d) ? d : 0;
        }
        deltas = this._scratchNewDeltas;
      }
    }
  }

  // ─── Training ────────────────────────────────────────────────────

  trainBatch(inputs: number[][], labels: number[]): TrainingSnapshot {
    let totalLoss = 0;
    let correct = 0;
    const batchSize = inputs.length;
    
    // Reuse indices array (avoid allocation per batch)
    if (this._scratchIndices.length !== batchSize) {
      this._scratchIndices = Array.from({ length: batchSize }, (_, i) => i);
    } else {
      // Reset values (they get shuffled)
      for (let i = 0; i < batchSize; i++) this._scratchIndices[i] = i;
    }
    const indices = this._scratchIndices;
    for (let i = batchSize - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp;
    }

    // Reuse target array (avoid 10-element allocation per sample)
    const target = this._scratchTarget;
    
    // Reference to output layer activations (set after final forward call)
    let lastProbs: number[] = [];
    
    for (let s = 0; s < batchSize; s++) {
      const idx = indices[s];
      const input = inputs[idx];
      const label = labels[idx];
      
      // Zero-fill and set one-hot in-place
      for (let t = 0; t < 10; t++) target[t] = 0;
      target[label] = 1;
      
      const output = this.forward(input);
      lastProbs = output;
      
      const loss = -Math.log(Math.max(output[label], 1e-10));
      totalLoss += isFinite(loss) ? loss : 10;
      
      const predicted = argmax(output);
      if (predicted === label) correct++;
      
      this.backward(input, target);
    }
    
    this.epoch++;
    const avgLoss = totalLoss / batchSize;
    const accuracy = correct / batchSize;
    this.lossHistory.push(avgLoss);
    this.accuracyHistory.push(accuracy);

    const bestIdx = argmax(lastProbs);
    const predictions = new Array(lastProbs.length);
    for (let i = 0; i < lastProbs.length; i++) predictions[i] = i === bestIdx ? 1 : 0;
    
    return {
      epoch: this.epoch,
      loss: avgLoss,
      accuracy,
      layers: this.snapshotLayers(),
      predictions,
      outputProbabilities: [...lastProbs],
    };
  }

  // ─── Prediction ──────────────────────────────────────────────────

  predict(input: number[]): { label: number; probabilities: number[]; layers: LayerState[] } {
    const output = this.forward(input);
    const label = argmax(output);
    return {
      label,
      probabilities: output.slice(), // copy — forward() returns a live array reference
      layers: this.snapshotLayers(),
    };
  }

  // ─── Snapshot helpers ────────────────────────────────────────────

  /** Deep-copy all layers for safe external consumption */
  snapshotLayers(): LayerState[] {
    return this.layers.map(l => ({
      weights: l.weights.map(w => [...w]),
      biases: [...l.biases],
      preActivations: [...l.preActivations],
      activations: [...l.activations],
    }));
  }

  // ─── Dream delegation (implementation in nn/dreams.ts) ──────────
  //     Thin wrappers for backward compatibility with existing API.

  computeInputGradient(input: number[], targetClass: number): number[] {
    return _computeInputGradient(this, input, targetClass);
  }

  dream(
    targetClass: number,
    steps: number = 100,
    lr: number = 0.5,
    startImage?: number[],
  ): DreamResult {
    return _dream(this, targetClass, steps, lr, startImage);
  }

  // ─── Accessors ───────────────────────────────────────────────────

  getLossHistory(): number[] { return [...this.lossHistory]; }
  getAccuracyHistory(): number[] { return [...this.accuracyHistory]; }
  getEpoch(): number { return this.epoch; }

  reset(inputSize: number, config?: TrainingConfig) {
    if (config) this.config = config;
    this.epoch = 0;
    this.lossHistory = [];
    this.accuracyHistory = [];
    this.neuronMasks.clear();
    this.initializeWeights(inputSize);
  }
}
