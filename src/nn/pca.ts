/**
 * PCA — Principal Component Analysis for 2D projection.
 *
 * Projects high-dimensional hidden-layer activations to 2D for
 * the Activation Space visualizer. Uses power iteration to find
 * the top 2 eigenvectors of the covariance matrix.
 */

export interface PCAProjection {
  /** 2D coordinates for each input sample */
  points: [number, number][];
  /** Explained variance for each component */
  variance: [number, number];
}

/**
 * Project N-dimensional data to 2D via PCA.
 * Uses power iteration (50 iters) for top 2 eigenvectors.
 */
export function projectTo2D(data: number[][]): PCAProjection {
  const n = data.length;
  if (n < 2) {
    return {
      points: data.map(() => [0, 0] as [number, number]),
      variance: [0, 0],
    };
  }

  const d = data[0].length;
  if (d === 0) {
    return {
      points: data.map(() => [0, 0] as [number, number]),
      variance: [0, 0],
    };
  }

  // Compute mean
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) {
      mean[j] += data[i][j];
    }
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  // Center data
  const centered: Float64Array[] = new Array(n);
  for (let i = 0; i < n; i++) {
    centered[i] = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      centered[i][j] = data[i][j] - mean[j];
    }
  }

  // Covariance matrix (d × d)
  const cov = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += centered[k][i] * centered[k][j];
      }
      sum /= n - 1;
      cov[i * d + j] = sum;
      cov[j * d + i] = sum;
    }
  }

  // Power iteration for top eigenvector
  const powerIter = (
    matrix: Float64Array,
    dim: number,
    iters: number = 50,
  ): { vector: Float64Array; value: number } => {
    const v = new Float64Array(dim);
    // Deterministic init (seeded direction)
    for (let i = 0; i < dim; i++) v[i] = Math.sin(i * 1.618 + 0.5);
    let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (norm > 0) for (let i = 0; i < dim; i++) v[i] /= norm;

    let eigenvalue = 0;
    const newV = new Float64Array(dim);

    for (let iter = 0; iter < iters; iter++) {
      // Matrix-vector multiply
      for (let i = 0; i < dim; i++) {
        let sum = 0;
        for (let j = 0; j < dim; j++) {
          sum += matrix[i * dim + j] * v[j];
        }
        newV[i] = sum;
      }

      norm = 0;
      for (let i = 0; i < dim; i++) norm += newV[i] * newV[i];
      norm = Math.sqrt(norm);
      eigenvalue = norm;

      if (norm > 1e-12) {
        for (let i = 0; i < dim; i++) v[i] = newV[i] / norm;
      } else {
        break;
      }
    }

    return { vector: v, value: eigenvalue };
  };

  // First principal component
  const pc1 = powerIter(cov, d);

  // Deflate: remove first component from covariance
  const deflated = new Float64Array(d * d);
  for (let i = 0; i < d * d; i++) deflated[i] = cov[i];
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      deflated[i * d + j] -= pc1.value * pc1.vector[i] * pc1.vector[j];
    }
  }

  // Second principal component
  const pc2 = powerIter(deflated, d);

  // Project all points
  const points: [number, number][] = new Array(n);
  for (let k = 0; k < n; k++) {
    let x = 0;
    let y = 0;
    for (let j = 0; j < d; j++) {
      x += centered[k][j] * pc1.vector[j];
      y += centered[k][j] * pc2.vector[j];
    }
    points[k] = [x, y];
  }

  return {
    points,
    variance: [pc1.value, pc2.value],
  };
}
