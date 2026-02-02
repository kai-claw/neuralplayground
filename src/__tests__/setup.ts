/**
 * Vitest global setup — polyfills for Node.js test environment.
 */

// ImageData is a browser API not available in Node.js.
// Provide a minimal polyfill so rendering functions can be tested.
if (typeof globalThis.ImageData === 'undefined') {
  (globalThis as Record<string, unknown>).ImageData = class ImageData {
    readonly data: Uint8ClampedArray;
    readonly width: number;
    readonly height: number;

    constructor(width: number, height: number);
    constructor(data: Uint8ClampedArray, width: number, height?: number);
    constructor(
      widthOrData: number | Uint8ClampedArray,
      heightOrWidth: number,
      maybeHeight?: number,
    ) {
      if (widthOrData instanceof Uint8ClampedArray) {
        this.data = widthOrData;
        this.width = heightOrWidth;
        this.height = maybeHeight ?? (widthOrData.length / 4 / heightOrWidth);
      } else {
        this.width = widthOrData;
        this.height = heightOrWidth;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}
