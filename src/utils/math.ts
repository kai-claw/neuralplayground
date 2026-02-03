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

// Pre-allocated softmax buffer (avoids 3× array allocation per call)
let _softmaxBuf: number[] = [];

/** Numerically stable softmax with degenerate fallback.
 *  NOTE: Returns a shared buffer. Callers must consume or copy before the next call. */
export function softmax(arr: number[]): number[] {
  const len = arr.length;
  if (_softmaxBuf.length < len) _softmaxBuf = new Array(len);
  const max = safeMax(arr);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    _softmaxBuf[i] = Math.exp(arr[i] - max);
    sum += _softmaxBuf[i];
  }
  if (sum === 0 || !isFinite(sum)) {
    const uniform = 1 / len;
    for (let i = 0; i < len; i++) _softmaxBuf[i] = uniform;
  } else {
    for (let i = 0; i < len; i++) _softmaxBuf[i] /= sum;
  }
  return _softmaxBuf;
}

/** Xavier/Glorot weight initialization (Box-Muller normal) */
export function xavierInit(fanIn: number, fanOut: number): number {
  const std = Math.sqrt(2 / (fanIn + fanOut));
  const u1 = Math.random();
  const u2 = Math.random();
  return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
