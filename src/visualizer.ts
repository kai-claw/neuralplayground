/**
 * Backward-compatibility re-export.
 *
 * Network layout computation has moved to visualizers/networkLayout.ts.
 * This file re-exports everything for existing import paths.
 */

export {
  computeNodePositions,
  generateParticles,
  getLayerSizes,
  type NodePos,
  type Particle,
} from './visualizers/networkLayout';
