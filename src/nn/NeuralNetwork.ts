export type ActivationFn = 'relu' | 'sigmoid' | 'tanh';

export interface LayerConfig {
  neurons: number;
  activation: ActivationFn;
}

export interface TrainingConfig {
  learningRate: number;
  layers: LayerConfig[];
}

export interface LayerState {
  weights: number[][];
  biases: number[];
  preActivations: number[];
  activations: number[];
}

export interface TrainingSnapshot {
  epoch: number;
  loss: number;
  accuracy: number;
  layers: LayerState[];
  predictions: number[];
  outputProbabilities: number[];
}

function activate(x: number, fn: ActivationFn): number {
  switch (fn) {
    case 'relu': return Math.max(0, x);
    case 'sigmoid': return 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500)));
    case 'tanh': return Math.tanh(x);
  }
}

function activateDerivative(x: number, fn: ActivationFn): number {
  switch (fn) {
    case 'relu': return x > 0 ? 1 : 0;
    case 'sigmoid': {
      const s = activate(x, 'sigmoid');
      return s * (1 - s);
    }
    case 'tanh': {
      const t = Math.tanh(x);
      return 1 - t * t;
    }
  }
}

/** Stack-safe max — avoids Math.max(...arr) RangeError on large arrays */
function safeMax(arr: number[]): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > m) m = arr[i];
  }
  return m;
}

/** Stack-safe argmax */
function argmax(arr: number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
  }
  return maxIdx;
}

function softmax(arr: number[]): number[] {
  const max = safeMax(arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0 || !isFinite(sum)) {
    // Uniform fallback on degenerate input
    return arr.map(() => 1 / arr.length);
  }
  return exps.map(x => x / sum);
}

function xavierInit(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  const u1 = Math.random();
  const u2 = Math.random();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export class NeuralNetwork {
  private layers: LayerState[] = [];
  private config: TrainingConfig;
  private epoch = 0;
  private lossHistory: number[] = [];
  private accuracyHistory: number[] = [];

  constructor(inputSize: number, config: TrainingConfig) {
    this.config = config;
    this.initializeWeights(inputSize);
  }

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
        // NaN/Infinity guard — clamp to safe range
        if (!isFinite(sum)) sum = 0;
        preAct.push(sum);
        act.push(isOutput ? sum : activate(sum, activation as ActivationFn));
      }
      
      layer.preActivations = preAct;
      
      if (isOutput) {
        layer.activations = softmax(preAct);
      } else {
        layer.activations = act;
      }
      
      current = layer.activations;
    }
    
    return current;
  }

  private backward(input: number[], target: number[]): void {
    const lr = this.config.learningRate;
    const numLayers = this.layers.length;
    
    const outputLayer = this.layers[numLayers - 1];
    let deltas: number[] = outputLayer.activations.map((a, i) => a - target[i]);
    
    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevActivations = l > 0 ? this.layers[l - 1].activations : input;
      
      for (let j = 0; j < layer.weights.length; j++) {
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
      totalLoss += isFinite(loss) ? loss : 10; // cap degenerate loss
      
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
      layers: this.layers.map(l => ({
        weights: l.weights.map(w => [...w]),
        biases: [...l.biases],
        preActivations: [...l.preActivations],
        activations: [...l.activations],
      })),
      predictions: lastProbs.map((_, i) => i === argmax(lastProbs) ? 1 : 0),
      outputProbabilities: lastProbs,
    };
  }

  predict(input: number[]): { label: number; probabilities: number[]; layers: LayerState[] } {
    const output = this.forward(input);
    const label = argmax(output);
    return {
      label,
      probabilities: output,
      layers: this.layers.map(l => ({
        weights: l.weights.map(w => [...w]),
        biases: [...l.biases],
        preActivations: [...l.preActivations],
        activations: [...l.activations],
      })),
    };
  }

  getLossHistory(): number[] { return [...this.lossHistory]; }
  getAccuracyHistory(): number[] { return [...this.accuracyHistory]; }
  getEpoch(): number { return this.epoch; }

  reset(inputSize: number, config?: TrainingConfig) {
    if (config) this.config = config;
    this.epoch = 0;
    this.lossHistory = [];
    this.accuracyHistory = [];
    this.initializeWeights(inputSize);
  }
}
