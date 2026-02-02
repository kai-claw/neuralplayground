/**
 * Canvas color helpers for network visualization.
 *
 * Maps activation values and weights to CSS color strings.
 */

/** Get node color based on activation value (cyan positive, red negative) */
export function getActivationColor(value: number, alpha = 1): string {
  if (value > 0) {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(99, 222, 255, ${intensity * alpha})`;
  } else {
    const intensity = Math.min(1, Math.abs(value));
    return `rgba(255, 99, 132, ${intensity * alpha})`;
  }
}

/** Get connection color based on weight value (cyan positive, red negative) */
export function getWeightColor(value: number): string {
  const clamped = Math.max(-1, Math.min(1, value));
  if (clamped > 0) {
    return `rgba(99, 222, 255, ${Math.abs(clamped) * 0.6})`;
  } else {
    return `rgba(255, 99, 132, ${Math.abs(clamped) * 0.6})`;
  }
}
