function createDigitPattern(digit: number): number[] {
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
    case 0:
      arc(14 + jx(), 14 + jy(), 7 + jx(), 0, Math.PI * 2);
      break;
    case 1:
      line(14 + jx(), 4 + jy(), 14 + jx(), 24 + jy());
      line(11 + jx(), 7 + jy(), 14 + jx(), 4 + jy());
      line(10, 24, 18, 24);
      break;
    case 2:
      arc(14 + jx(), 10 + jy(), 6, -Math.PI, 0.3);
      line(19 + jx(), 12 + jy(), 8 + jx(), 24 + jy());
      line(8, 24, 20 + jx(), 24 + jy());
      break;
    case 3:
      arc(14 + jx(), 10 + jy(), 5 + jx(), -Math.PI * 0.8, Math.PI * 0.5);
      arc(14 + jx(), 18 + jy(), 5 + jx(), -Math.PI * 0.5, Math.PI * 0.8);
      break;
    case 4:
      line(18 + jx(), 4 + jy(), 8 + jx(), 16 + jy());
      line(8 + jx(), 16 + jy(), 22 + jx(), 16 + jy());
      line(18 + jx(), 4 + jy(), 18 + jx(), 24 + jy());
      break;
    case 5:
      line(18 + jx(), 5 + jy(), 9 + jx(), 5 + jy());
      line(9 + jx(), 5 + jy(), 9 + jx(), 13 + jy());
      arc(14 + jx(), 17 + jy(), 6, -Math.PI * 0.6, Math.PI * 0.7);
      break;
    case 6:
      arc(14 + jx(), 18 + jy(), 6, 0, Math.PI * 2);
      line(8 + jx(), 18 + jy(), 12 + jx(), 5 + jy());
      break;
    case 7:
      line(8 + jx(), 5 + jy(), 20 + jx(), 5 + jy());
      line(20 + jx(), 5 + jy(), 12 + jx(), 24 + jy());
      break;
    case 8:
      arc(14 + jx(), 10 + jy(), 5, 0, Math.PI * 2);
      arc(14 + jx(), 19 + jy(), 5, 0, Math.PI * 2);
      break;
    case 9:
      arc(14 + jx(), 10 + jy(), 6, 0, Math.PI * 2);
      line(20 + jx(), 10 + jy(), 16 + jx(), 24 + jy());
      break;
  }

  return canvas.map(v => Math.min(1, Math.max(0, v + (Math.random() - 0.5) * 0.05)));
}

export function generateTrainingData(samplesPerDigit = 15): { inputs: number[][]; labels: number[] } {
  const inputs: number[][] = [];
  const labels: number[] = [];
  
  for (let digit = 0; digit < 10; digit++) {
    for (let s = 0; s < samplesPerDigit; s++) {
      inputs.push(createDigitPattern(digit));
      labels.push(digit);
    }
  }
  
  return { inputs, labels };
}

export function canvasToInput(imageData: ImageData, targetSize = 28): number[] {
  const { width, height, data } = imageData;
  const result: number[] = new Array(targetSize * targetSize).fill(0);
  
  const scaleX = width / targetSize;
  const scaleY = height / targetSize;
  
  for (let ty = 0; ty < targetSize; ty++) {
    for (let tx = 0; tx < targetSize; tx++) {
      let sum = 0;
      let count = 0;
      
      const startX = Math.floor(tx * scaleX);
      const endX = Math.floor((tx + 1) * scaleX);
      const startY = Math.floor(ty * scaleY);
      const endY = Math.floor((ty + 1) * scaleY);
      
      for (let y = startY; y < endY; y++) {
        for (let x = startX; x < endX; x++) {
          const idx = (y * width + x) * 4;
          const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
          sum += gray / 255;
          count++;
        }
      }
      
      result[ty * targetSize + tx] = count > 0 ? sum / count : 0;
    }
  }
  
  return result;
}
