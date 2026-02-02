import { describe, it, expect } from 'vitest';
import { generateTrainingData, canvasToInput } from '../nn/sampleData';

describe('generateTrainingData', () => {
  it('should generate correct number of samples', () => {
    const data = generateTrainingData(5);
    expect(data.inputs.length).toBe(50); // 10 digits × 5 per digit
    expect(data.labels.length).toBe(50);
  });

  it('should use default 15 samples per digit', () => {
    const data = generateTrainingData();
    expect(data.inputs.length).toBe(150);
    expect(data.labels.length).toBe(150);
  });

  it('should produce 784-element inputs (28×28)', () => {
    const data = generateTrainingData(2);
    for (const input of data.inputs) {
      expect(input.length).toBe(784);
    }
  });

  it('should produce labels in range 0-9', () => {
    const data = generateTrainingData(3);
    for (const label of data.labels) {
      expect(label).toBeGreaterThanOrEqual(0);
      expect(label).toBeLessThanOrEqual(9);
    }
  });

  it('should contain all 10 digit classes', () => {
    const data = generateTrainingData(2);
    const uniqueLabels = new Set(data.labels);
    expect(uniqueLabels.size).toBe(10);
  });

  it('should produce pixel values in [0, 1]', () => {
    const data = generateTrainingData(2);
    for (const input of data.inputs) {
      for (const pixel of input) {
        expect(pixel).toBeGreaterThanOrEqual(0);
        expect(pixel).toBeLessThanOrEqual(1);
      }
    }
  });

  it('should produce non-trivial patterns (not all zeros)', () => {
    const data = generateTrainingData(2);
    for (const input of data.inputs) {
      const sum = input.reduce((a, b) => a + b, 0);
      expect(sum).toBeGreaterThan(0);
    }
  });

  it('should produce different patterns for different digits', () => {
    const data = generateTrainingData(1);
    // Compare digit 0 pattern vs digit 1 pattern
    const input0 = data.inputs[0];
    const input1 = data.inputs[1];
    const differs = input0.some((v, i) => Math.abs(v - input1[i]) > 0.1);
    expect(differs).toBe(true);
  });

  it('should produce variation within same digit (jitter)', () => {
    const data = generateTrainingData(3);
    // First 3 are digit 0
    const a = data.inputs[0];
    const b = data.inputs[1];
    // They should differ due to jitter
    const differs = a.some((v, i) => Math.abs(v - b[i]) > 0.01);
    expect(differs).toBe(true);
  });

  it('should not contain NaN or Infinity', () => {
    const data = generateTrainingData(5);
    for (const input of data.inputs) {
      for (const pixel of input) {
        expect(isFinite(pixel)).toBe(true);
        expect(isNaN(pixel)).toBe(false);
      }
    }
  });
});

describe('canvasToInput', () => {
  function createMockImageData(width: number, height: number, value: number): ImageData {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i += 4) {
      data[i] = value;     // R
      data[i + 1] = value; // G
      data[i + 2] = value; // B
      data[i + 3] = 255;   // A
    }
    return { data, width, height, colorSpace: 'srgb' } as ImageData;
  }

  it('should produce 28×28 = 784 values by default', () => {
    const imgData = createMockImageData(280, 280, 128);
    const result = canvasToInput(imgData);
    expect(result.length).toBe(784);
  });

  it('should produce values in [0, 1]', () => {
    const imgData = createMockImageData(280, 280, 200);
    const result = canvasToInput(imgData);
    for (const v of result) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('should produce ~0 for black canvas', () => {
    const imgData = createMockImageData(280, 280, 0);
    const result = canvasToInput(imgData);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(0);
  });

  it('should produce ~1 for white canvas', () => {
    const imgData = createMockImageData(280, 280, 255);
    const result = canvasToInput(imgData);
    for (const v of result) {
      expect(v).toBeCloseTo(1, 1);
    }
  });

  it('should produce ~0.5 for mid-gray canvas', () => {
    const imgData = createMockImageData(280, 280, 128);
    const result = canvasToInput(imgData);
    const avg = result.reduce((a, b) => a + b, 0) / result.length;
    expect(avg).toBeCloseTo(128 / 255, 1);
  });

  it('should handle non-square aspect ratio', () => {
    const imgData = createMockImageData(400, 200, 128);
    const result = canvasToInput(imgData);
    expect(result.length).toBe(784);
  });

  it('should handle custom target size', () => {
    const imgData = createMockImageData(100, 100, 128);
    const result = canvasToInput(imgData, 14);
    expect(result.length).toBe(196); // 14×14
  });
});
