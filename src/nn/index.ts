/**
 * Neural network barrel export.
 *
 * Single import point for the core neural network module:
 * class, dream functions, sample data generation, and noise.
 *
 * Types are imported from the canonical src/types.ts.
 */

export { NeuralNetwork } from './NeuralNetwork';
export {
  computeInputGradient,
  dream,
} from './dreams';
export {
  generateTrainingData,
  canvasToInput,
} from './sampleData';
export {
  generateNoisePattern,
  applyNoise,
} from './noise';
export {
  computeSaliency,
  saliencyToColor,
  renderSaliencyOverlay,
} from './saliency';
export {
  projectTo2D,
} from './pca';
export type { PCAProjection } from './pca';
export {
  computeConfusionMatrix,
} from './confusion';
export type { ConfusionData } from './confusion';
export {
  measureGradientFlow,
  GradientFlowHistory,
} from './gradientFlow';
export type { LayerGradientStats, GradientFlowSnapshot } from './gradientFlow';
export {
  EpochRecorder,
  paramsToLayers,
  replayForward,
} from './epochReplay';
export type { EpochSnapshot, LayerParams } from './epochReplay';
export {
  computeDecisionBoundary,
  renderDecisionBoundary,
  generateExemplar,
} from './decisionBoundary';
export type { BoundaryCell, DecisionBoundaryResult } from './decisionBoundary';
export {
  dreamChimera,
  CHIMERA_PRESETS,
} from './chimera';
export type { ChimeraResult, ChimeraPreset } from './chimera';
export {
  findMisfits,
  computeMisfitSummary,
} from './misfits';
export type { Misfit, MisfitSummary } from './misfits';
export {
  WeightEvolutionRecorder,
  renderNeuronWeights,
  computeWeightDelta,
} from './weightEvolution';
export type { WeightFrame } from './weightEvolution';
export {
  runAblationStudy,
  importanceToColor,
} from './ablation';
export type { AblationResult, AblationStudy } from './ablation';
