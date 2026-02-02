import {
  activate,
  activateDerivative,
  softmax,
  xavierInit,
  argmax,
} from '../utils';

export type ActivationFn = 'relu' | 'sigmoid' | 'tanh';

/** Neuron surgery status */
export type NeuronStatus = 'active' | 'frozen' | 'killed';

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
        // Apply neuron surgery masks — killed neurons output zero
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

  private backward(input: number[], target: number[]): void {
    const lr = this.config.learningRate;
    const numLayers = this.layers.length;
    
    const outputLayer = this.layers[numLayers - 1];
    let deltas: number[] = outputLayer.activations.map((a, i) => a - target[i]);
    
    for (let l = numLayers - 1; l >= 0; l--) {
      const layer = this.layers[l];
      const prevActivations = l > 0 ? this.layers[l - 1].activations : input;
      
      for (let j = 0; j < layer.weights.length; j++) {
        // Neuron surgery: skip weight updates for frozen/killed neurons
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

  // ─── Network Dreams — gradient ascent on input space ──────────────

  /**
   * Compute the gradient of output[targetClass] with respect to the input.
   * Used for "Network Dreams" — gradient ascent to visualize what the
   * network imagines for each digit.
   */
  computeInputGradient(input: number[], targetClass: number): number[] {
    // Forward pass to populate layer states
    this.forward(input);

    const numLayers = this.layers.length;

    // Output layer delta: gradient of cross-entropy w.r.t. logits
    // We want to MAXIMIZE output[targetClass], so delta = target - output
    const outputLayer = this.layers[numLayers - 1];
    let deltas: number[] = outputLayer.activations.map((a, i) =>
      (i === targetClass ? 1 : 0) - a
    );

    // Backpropagate through hidden layers to get input gradient
    for (let l = numLayers - 1; l >= 1; l--) {
      const layer = this.layers[l];
      const prevLayer = this.layers[l - 1];
      const activation = this.config.layers[l - 1]?.activation || 'relu';
      const newDeltas = new Array(prevLayer.weights.length).fill(0);

      for (let i = 0; i < prevLayer.weights.length; i++) {
        let sum = 0;
        for (let j = 0; j < layer.weights.length; j++) {
          sum += layer.weights[j][i] * deltas[j];
        }
        const d = sum * activateDerivative(
          prevLayer.preActivations[i],
          activation as ActivationFn,
        );
        newDeltas[i] = isFinite(d) ? d : 0;
      }
      deltas = newDeltas;
    }

    // Final step: gradient w.r.t. input
    const firstLayer = this.layers[0];
    const inputGradient = new Array(input.length).fill(0);
    for (let i = 0; i < input.length; i++) {
      let sum = 0;
      for (let j = 0; j < firstLayer.weights.length; j++) {
        sum += firstLayer.weights[j][i] * deltas[j];
      }
      inputGradient[i] = isFinite(sum) ? sum : 0;
    }

    return inputGradient;
  }

  /**
   * Run gradient ascent to "dream" what input produces a target digit.
   * Returns the optimized input image and confidence history.
   */
  dream(
    targetClass: number,
    steps: number = 100,
    lr: number = 0.5,
    startImage?: number[],
  ): { image: number[]; confidenceHistory: number[] } {
    const size = this.layers[0].weights[0]?.length || 784;
    let image = startImage
      ? [...startImage]
      : Array.from({ length: size }, () => Math.random() * 0.3 + 0.1);

    const confidenceHistory: number[] = [];

    for (let step = 0; step < steps; step++) {
      const output = this.forward(image);
      confidenceHistory.push(output[targetClass]);

      const gradient = this.computeInputGradient(image, targetClass);

      // Gradient ascent with L2 regularization for cleaner images
      for (let i = 0; i < image.length; i++) {
        image[i] += lr * gradient[i] - 0.001 * image[i];
        image[i] = Math.max(0, Math.min(1, image[i]));
      }

      // Decay learning rate slightly
      lr *= 0.998;
    }

    return { image, confidenceHistory };
  }

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
