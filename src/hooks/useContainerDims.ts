/**
 * useContainerDims — responsive container dimensions via ResizeObserver.
 *
 * Replaces the duplicated pattern of useState + useEffect + ResizeObserver
 * found in NetworkVisualizer, LossChart, and ActivationVisualizer.
 *
 * If explicit width/height props are provided, those are used directly
 * (derived state, no effect needed). Otherwise, observes the container
 * element and computes dimensions from its width × aspect ratio.
 */

import { useRef, useState, useEffect } from 'react';
import type { Dimensions } from '../types';

interface UseContainerDimsOptions {
  /** Explicit width from props (overrides observer) */
  propWidth?: number;
  /** Explicit height from props (overrides observer) */
  propHeight?: number;
  /** Default width when no observer and no props */
  defaultWidth: number;
  /** Default height when no observer and no props */
  defaultHeight: number;
  /** Height = width × aspect ratio (used by observer) */
  aspectRatio: number;
}

export function useContainerDims({
  propWidth,
  propHeight,
  defaultWidth,
  defaultHeight,
  aspectRatio,
}: UseContainerDimsOptions): {
  containerRef: React.RefObject<HTMLDivElement | null>;
  dims: Dimensions;
} {
  const containerRef = useRef<HTMLDivElement>(null);
  const [observedDims, setObservedDims] = useState<Dimensions>({
    width: defaultWidth,
    height: defaultHeight,
  });

  // Only observe when no explicit props are provided
  useEffect(() => {
    if (propWidth && propHeight) return; // props take priority — no observer needed
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0].contentRect.width;
      if (w > 0) {
        setObservedDims({ width: w, height: Math.round(w * aspectRatio) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [propWidth, propHeight, aspectRatio]);

  // Derive final dims: explicit props override observed dims
  const dims: Dimensions =
    propWidth && propHeight
      ? { width: propWidth, height: propHeight }
      : observedDims;

  return { containerRef, dims };
}
