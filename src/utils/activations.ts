/**
 * Activation functions and their derivatives.
 *
 * Used by the neural network forward/backward pass and visualization.
 */

import type { ActivationFn } from '../types';

export function activate(x: number, fn: ActivationFn): number {
  switch (fn) {
    case 'relu': return Math.max(0, x);
    case 'sigmoid': return 1 / (1 + Math.exp(-Math.min(Math.max(x, -500), 500)));
    case 'tanh': return Math.tanh(x);
  }
}

export function activateDerivative(x: number, fn: ActivationFn): number {
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
