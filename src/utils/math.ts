/**
 * Array and math helpers — softmax, argmax, Xavier init.
 *
 * Pure functions with no side effects (except xavierInit uses Math.random).
 */

/** Stack-safe max — avoids Math.max(...arr) RangeError on large arrays */
export function safeMax(arr: number[]): number {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > m) m = arr[i];
  }
  return m;
}

/** Stack-safe argmax */
export function argmax(arr: number[]): number {
  let maxIdx = 0;
  let maxVal = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > maxVal) { maxVal = arr[i]; maxIdx = i; }
  }
  return maxIdx;
}

/** Numerically stable softmax with degenerate fallback */
export function softmax(arr: number[]): number[] {
  const max = safeMax(arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  if (sum === 0 || !isFinite(sum)) {
    return arr.map(() => 1 / arr.length);
  }
  return exps.map(x => x / sum);
}

/** Xavier/Glorot weight initialization (Box-Muller normal) */
export function xavierInit(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  const u1 = Math.random();
  const u2 = Math.random();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
