/**
 * Backward-compatibility re-export.
 *
 * Pixel rendering has moved to renderers/pixelRendering.ts.
 * This file re-exports everything for existing import paths.
 */

export {
  weightsToImageData,
  pixelsToImageData,
  lerpPixels,
} from './renderers/pixelRendering';
