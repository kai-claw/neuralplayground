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
  softmax,
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
      
      const preAct: number[] = [];
      const act: number[] = [];
      
      for (let j = 0; j < layer.weights.length; j++) {
        let sum = layer.biases[j];
        for (let i = 0; i < current.length; i++) {
          sum += layer.weights[j][i] * current[i];
        }
        // NaN/Infinity guard
        if (!isFinite(sum)) sum = 0;
        preAct.push(sum);
        act.push(isOutput ? sum : activate(sum, activation as ActivationFn));
      }
      
      layer.preActivations = preAct;
      
      if (isOutput) {
        layer.activations = softmax(preAct);
      } else {
        // Apply neuron surgery masks
        for (let j = 0; j < act.length; j++) {
          if (this.neuronMasks.get(`${l}-${j}`) === 'killed') {
            act[j] = 0;
          }
        }
        layer.activations = act;
      }
      
      current = layer.activations;
    }
    
    return current;
  }

  // ─── Backward pass ───────────────────────────────────────────────

  private backward(input: number[], target: number[]): void {
    const lr = this.config.learningRate;
    const numLayers = this.layers.length;
    
    const outputLayer = this.layers[numLayers - 1];
    let deltas: number[] = outputLayer.activations.map((a, i) => a - target[i]);
    
    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevActivations = l > 0 ? this.layers[l - 1].activations : input;
      
      for (let j = 0; j < layer.weights.length; j++) {
        const status = this.neuronMasks.get(`${l}-${j}`);
        if (status === 'frozen' || status === 'killed') continue;

        for (let i = 0; i < layer.weights[j].length; i++) {
          const grad = lr * deltas[j] * prevActivations[i];
          if (isFinite(grad)) {
            layer.weights[j][i] -= grad;
          }
        }
        const biasGrad = lr * deltas[j];
        if (isFinite(biasGrad)) {
          layer.biases[j] -= biasGrad;
        }
      }
      
      if (l > 0) {
        const prevLayer = this.layers[l - 1];
        const activation = this.config.layers[l - 1]?.activation || 'relu';
        const newDeltas: number[] = new Array(prevLayer.weights.length).fill(0);
        
        for (let i = 0; i < prevLayer.weights.length; i++) {
          let sum = 0;
          for (let j = 0; j < layer.weights.length; j++) {
            sum += layer.weights[j][i] * deltas[j];
          }
          const d = sum * activateDerivative(prevLayer.preActivations[i], activation as ActivationFn);
          newDeltas[i] = isFinite(d) ? d : 0;
        }
        deltas = newDeltas;
      }
    }
  }

  // ─── Training ────────────────────────────────────────────────────

  trainBatch(inputs: number[][], labels: number[]): TrainingSnapshot {
    let totalLoss = 0;
    let correct = 0;
    
    const indices = Array.from({ length: inputs.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    
    let lastProbs: number[] = [];
    
    for (const idx of indices) {
      const input = inputs[idx];
      const label = labels[idx];
      
      const target = new Array(10).fill(0);
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
    const avgLoss = totalLoss / inputs.length;
    const accuracy = correct / inputs.length;
    this.lossHistory.push(avgLoss);
    this.accuracyHistory.push(accuracy);
    
    return {
      epoch: this.epoch,
      loss: avgLoss,
      accuracy,
      layers: this.snapshotLayers(),
      predictions: lastProbs.map((_, i) => i === argmax(lastProbs) ? 1 : 0),
      outputProbabilities: lastProbs,
    };
  }

  // ─── Prediction ──────────────────────────────────────────────────

  predict(input: number[]): { label: number; probabilities: number[]; layers: LayerState[] } {
    const output = this.forward(input);
    const label = argmax(output);
    return {
      label,
      probabilities: output,
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
