/**
 * Renderers barrel export.
 *
 * Pure canvas rendering functions extracted from React components.
 * All functions take a CanvasRenderingContext2D and data — no React dependency.
 */

export {
  drawRaceChart,
  CHART_WIDTH,
  type RaceChartData,
  type RacerVisuals,
} from './raceChart';

export {
  renderDreamImage,
  renderDreamGallery,
  dreamPixelToRGB,
  GALLERY_DIMS,
} from './dreamRenderer';

export {
  drawSurgeryCanvas,
  computeSurgeryLayout,
  hitTestSurgeryNode,
  type SurgeryNode,
  type SurgeryLayout,
  type SurgeryCounts,
} from './surgeryRenderer';

export {
  weightsToImageData,
  pixelsToImageData,
  lerpPixels,
} from './pixelRendering';
