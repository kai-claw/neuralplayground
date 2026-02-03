/**
 * Decision Boundary Map — visualize the network's classification landscape.
 *
 * Generates a 2D grid of interpolated inputs between two digit exemplars,
 * runs each through the network, and produces a confidence heatmap showing
 * where the network's decision boundary lies.
 *
 * Creates beautiful gradient visualizations of how the network separates
 * different digit classes in pixel space.
 */

/** Interface for anything that can classify inputs */
interface Classifier {
  predict(input: number[]): { label: number; probabilities: number[] };
}

/** A single cell in the decision boundary grid */
export interface BoundaryCell {
  /** Network confidence for digit A */
  confA: number;
  /** Network confidence for digit B */
  confB: number;
  /** Predicted label */
  label: number;
  /** Max confidence across all classes */
  maxConf: number;
}

/** Full decision boundary result */
export interface DecisionBoundaryResult {
  /** Grid of boundary cells [y][x] */
  grid: BoundaryCell[][];
  /** Grid resolution */
  resolution: number;
  /** Digit A (rows = more of A) */
  digitA: number;
  /** Digit B (cols = more of B) */
  digitB: number;
}

/**
 * Generate exemplar images for a digit by averaging multiple random samples.
 * Uses the same pattern generator as training data.
 */
function generateExemplar(digit: number): number[] {
  // Average 5 samples for a clean prototype
  const SIZE = 784;
  const avg = new Float64Array(SIZE);
  const N = 5;

  for (let s = 0; s < N; s++) {
    const sample = generateDigitPattern(digit);
    for (let i = 0; i < SIZE; i++) {
      avg[i] += sample[i];
    }
  }

  const result: number[] = new Array(SIZE);
  for (let i = 0; i < SIZE; i++) {
    result[i] = Math.max(0, Math.min(1, avg[i] / N));
  }
  return result;
}

/**
 * Simplified digit pattern generator (mirrors sampleData.ts createDigitPattern).
 * Self-contained to avoid circular dependency.
 */
function generateDigitPattern(digit: number): number[] {
  const canvas = new Array(28 * 28).fill(0);

  const set = (x: number, y: number, v = 1) => {
    if (x >= 0 && x < 28 && y >= 0 && y < 28) {
      canvas[y * 28 + x] = Math.min(1, canvas[y * 28 + x] + v);
    }
  };

  const line = (x1: number, y1: number, x2: number, y2: number, thickness = 2) => {
    const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1)) * 2;
    for (let i = 0; i <= steps; i++) {
      const t = steps === 0 ? 0 : i / steps;
      const x = Math.round(x1 + (x2 - x1) * t);
      const y = Math.round(y1 + (y2 - y1) * t);
      for (let dx = -thickness + 1; dx < thickness; dx++) {
        for (let dy = -thickness + 1; dy < thickness; dy++) {
          set(x + dx, y + dy, 0.8);
        }
      }
    }
  };

  const arc = (cx: number, cy: number, r: number, startAngle: number, endAngle: number, thickness = 2) => {
    const steps = Math.max(20, Math.abs(endAngle - startAngle) * r);
    for (let i = 0; i <= steps; i++) {
      const angle = startAngle + (endAngle - startAngle) * (i / steps);
      const x = Math.round(cx + r * Math.cos(angle));
      const y = Math.round(cy + r * Math.sin(angle));
      for (let dx = -thickness + 1; dx < thickness; dx++) {
        for (let dy = -thickness + 1; dy < thickness; dy++) {
          set(x + dx, y + dy, 0.8);
        }
      }
    }
  };

  const jx = () => (Math.random() - 0.5) * 1.5;
  const jy = () => (Math.random() - 0.5) * 1.5;

  switch (digit) {
    case 0: arc(14 + jx(), 14 + jy(), 7 + jx(), 0, Math.PI * 2); break;
    case 1: line(14 + jx(), 4 + jy(), 14 + jx(), 24 + jy()); line(11 + jx(), 7 + jy(), 14 + jx(), 4 + jy()); line(10, 24, 18, 24); break;
    case 2: arc(14 + jx(), 10 + jy(), 6, -Math.PI, 0.3); line(19 + jx(), 12 + jy(), 8 + jx(), 24 + jy()); line(8, 24, 20 + jx(), 24 + jy()); break;
    case 3: arc(14 + jx(), 10 + jy(), 5 + jx(), -Math.PI * 0.8, Math.PI * 0.5); arc(14 + jx(), 18 + jy(), 5 + jx(), -Math.PI * 0.5, Math.PI * 0.8); break;
    case 4: line(18 + jx(), 4 + jy(), 8 + jx(), 16 + jy()); line(8 + jx(), 16 + jy(), 22 + jx(), 16 + jy()); line(18 + jx(), 4 + jy(), 18 + jx(), 24 + jy()); break;
    case 5: line(18 + jx(), 5 + jy(), 9 + jx(), 5 + jy()); line(9 + jx(), 5 + jy(), 9 + jx(), 13 + jy()); arc(14 + jx(), 17 + jy(), 6, -Math.PI * 0.6, Math.PI * 0.7); break;
    case 6: arc(14 + jx(), 18 + jy(), 6, 0, Math.PI * 2); line(8 + jx(), 18 + jy(), 12 + jx(), 5 + jy()); break;
    case 7: line(8 + jx(), 5 + jy(), 20 + jx(), 5 + jy()); line(20 + jx(), 5 + jy(), 12 + jx(), 24 + jy()); break;
    case 8: arc(14 + jx(), 10 + jy(), 5, 0, Math.PI * 2); arc(14 + jx(), 19 + jy(), 5, 0, Math.PI * 2); break;
    case 9: arc(14 + jx(), 10 + jy(), 6, 0, Math.PI * 2); line(20 + jx(), 10 + jy(), 16 + jx(), 24 + jy()); break;
  }

  return canvas.map(v => Math.min(1, Math.max(0, v + (Math.random() - 0.5) * 0.05)));
}

/**
 * Compute the decision boundary grid between two digits.
 *
 * Creates a resolution × resolution grid where:
 * - X axis: interpolation from digit A to digit B exemplar
 * - Y axis: adding a perpendicular variation (noise-based)
 *
 * Each cell is classified by the network, producing a heatmap
 * of the decision landscape.
 */
export function computeDecisionBoundary(
  network: Classifier,
  digitA: number,
  digitB: number,
  resolution = 32,
): DecisionBoundaryResult {
  const exemplarA = generateExemplar(digitA);
  const exemplarB = generateExemplar(digitB);

  // Also generate a "perpendicular" variation for the Y axis:
  // average of A and B, plus another pair for orthogonal direction
  const exemplarA2 = generateExemplar(digitA);
  const exemplarB2 = generateExemplar(digitB);

  // Perpendicular direction: difference between second pair
  const perpendicular = new Float64Array(784);
  for (let i = 0; i < 784; i++) {
    perpendicular[i] = (exemplarA2[i] - exemplarB2[i]) * 0.5;
  }

  const grid: BoundaryCell[][] = [];

  for (let y = 0; y < resolution; y++) {
    const row: BoundaryCell[] = [];
    const ty = y / (resolution - 1); // 0 to 1 (perpendicular axis)
    const perpStrength = (ty - 0.5) * 2; // -1 to 1

    for (let x = 0; x < resolution; x++) {
      const tx = x / (resolution - 1); // 0 to 1 (A→B axis)

      // Interpolate between A and B, then add perpendicular variation
      const input: number[] = new Array(784);
      for (let i = 0; i < 784; i++) {
        const base = exemplarA[i] * (1 - tx) + exemplarB[i] * tx;
        const perp = perpendicular[i] * perpStrength * 0.3;
        input[i] = Math.max(0, Math.min(1, base + perp));
      }

      // Classify
      const result = network.predict(input);
      const probs = result.probabilities;

      row.push({
        confA: probs[digitA] || 0,
        confB: probs[digitB] || 0,
        label: result.label,
        maxConf: Math.max(...probs),
      });
    }

    grid.push(row);
  }

  return { grid, resolution, digitA, digitB };
}

/**
 * Render decision boundary as ImageData.
 *
 * Colors: digit A = blue, digit B = red, other = gray.
 * Intensity based on confidence. Boundary appears as the
 * transition zone between colors.
 */
// Cached ImageData for decision boundary rendering
const _boundaryCache = new Map<number, ImageData>();

export function renderDecisionBoundary(
  result: DecisionBoundaryResult,
  size: number,
): ImageData {
  const { grid, resolution, digitA, digitB } = result;
  let imageData = _boundaryCache.get(size);
  if (!imageData) {
    imageData = new ImageData(size, size);
    _boundaryCache.set(size, imageData);
  }
  const data = imageData.data;
  const cellSize = size / resolution;

  // Color palette for digit A (cool blue) and digit B (warm red)
  const colorA = { r: 54, g: 162, b: 235 }; // blue
  const colorB = { r: 255, g: 99, b: 132 }; // red
  const colorOther = { r: 80, g: 80, b: 100 }; // gray-purple

  for (let py = 0; py < size; py++) {
    const gy = Math.min(resolution - 1, Math.floor(py / cellSize));
    for (let px = 0; px < size; px++) {
      const gx = Math.min(resolution - 1, Math.floor(px / cellSize));
      const cell = grid[gy][gx];

      // Determine dominant color
      let r: number, g: number, b: number;

      if (cell.label === digitA) {
        const t = cell.confA; // 0-1 confidence
        r = Math.round(colorA.r * t + 15 * (1 - t));
        g = Math.round(colorA.g * t + 15 * (1 - t));
        b = Math.round(colorA.b * t + 40 * (1 - t));
      } else if (cell.label === digitB) {
        const t = cell.confB;
        r = Math.round(colorB.r * t + 15 * (1 - t));
        g = Math.round(colorB.g * t + 15 * (1 - t));
        b = Math.round(colorB.b * t + 40 * (1 - t));
      } else {
        const t = cell.maxConf * 0.5;
        r = Math.round(colorOther.r * t + 15 * (1 - t));
        g = Math.round(colorOther.g * t + 15 * (1 - t));
        b = Math.round(colorOther.b * t + 40 * (1 - t));
      }

      // Add boundary highlight: cells near the decision boundary glow
      const boundaryProximity = 1 - Math.abs(cell.confA - cell.confB);
      if (boundaryProximity > 0.7) {
        const glow = (boundaryProximity - 0.7) / 0.3;
        r = Math.min(255, r + Math.round(glow * 60));
        g = Math.min(255, g + Math.round(glow * 60));
        b = Math.min(255, b + Math.round(glow * 60));
      }

      const offset = (py * size + px) * 4;
      data[offset] = r;
      data[offset + 1] = g;
      data[offset + 2] = b;
      data[offset + 3] = 255;
    }
  }

  return imageData;
}

export { generateExemplar };
