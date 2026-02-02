/**
 * Digit stroke definitions for cinematic auto-draw mode.
 * Each digit is defined as an array of strokes (pen lifts between strokes).
 * Each stroke is an array of {x,y} points in 0-280 canvas space.
 */

export interface StrokePoint {
  x: number;
  y: number;
}

export interface DigitStroke {
  points: StrokePoint[];
}

function arcPts(cx: number, cy: number, r: number, start: number, end: number, n = 32): StrokePoint[] {
  const pts: StrokePoint[] = [];
  for (let i = 0; i <= n; i++) {
    const a = start + (end - start) * (i / n);
    pts.push({ x: cx + r * Math.cos(a), y: cy + r * Math.sin(a) });
  }
  return pts;
}

function linePts(x1: number, y1: number, x2: number, y2: number, n = 16): StrokePoint[] {
  const pts: StrokePoint[] = [];
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    pts.push({ x: x1 + (x2 - x1) * t, y: y1 + (y2 - y1) * t });
  }
  return pts;
}

/** Stroke definitions for digits 0-9 in 280×280 canvas space */
export const DIGIT_STROKES: DigitStroke[][] = [
  // 0: oval
  [{ points: arcPts(140, 145, 68, -Math.PI * 0.5, Math.PI * 1.5) }],
  // 1: serif + down stroke + base
  [
    { points: linePts(112, 78, 140, 48) },
    { points: linePts(140, 48, 140, 232) },
    { points: linePts(105, 232, 175, 232) },
  ],
  // 2: top curve + diagonal + base
  [
    { points: arcPts(140, 100, 52, -Math.PI, 0.3, 24) },
    { points: linePts(183, 118, 88, 232, 20) },
    { points: linePts(88, 232, 195, 232) },
  ],
  // 3: top arc + bottom arc
  [
    { points: arcPts(138, 100, 48, -Math.PI * 0.8, Math.PI * 0.5, 24) },
    { points: arcPts(138, 185, 48, -Math.PI * 0.5, Math.PI * 0.8, 24) },
  ],
  // 4: diagonal + horizontal + vertical
  [
    { points: linePts(175, 48, 88, 162) },
    { points: linePts(88, 162, 210, 162) },
    { points: linePts(175, 48, 175, 232) },
  ],
  // 5: top + down + curve
  [
    { points: linePts(178, 52, 98, 52) },
    { points: linePts(98, 52, 98, 132) },
    { points: arcPts(138, 172, 52, -Math.PI * 0.6, Math.PI * 0.7, 24) },
  ],
  // 6: stem + circle
  [
    { points: linePts(122, 52, 88, 185, 20) },
    { points: arcPts(140, 185, 52, 0, Math.PI * 2) },
  ],
  // 7: top + diagonal
  [
    { points: linePts(88, 52, 195, 52) },
    { points: linePts(195, 52, 128, 232) },
  ],
  // 8: top + bottom
  [
    { points: arcPts(140, 108, 42, 0, Math.PI * 2) },
    { points: arcPts(140, 192, 42, 0, Math.PI * 2) },
  ],
  // 9: top circle + tail
  [
    { points: arcPts(140, 108, 52, 0, Math.PI * 2) },
    { points: linePts(192, 108, 158, 232) },
  ],
];

/** Total point count for a digit (determines draw speed) */
export function getDigitPointCount(digit: number): number {
  let n = 0;
  for (const s of DIGIT_STROKES[digit]) n += s.points.length;
  return n;
}

/** Time to draw a digit in ms (longer digits take longer) */
export function getDigitDrawDuration(digit: number): number {
  return Math.max(700, getDigitPointCount(digit) * 18);
}

/** All 10 digits exist */
export const DIGIT_COUNT = 10;
