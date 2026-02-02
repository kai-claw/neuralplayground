import { useState } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';
import WeightHeatmap from './WeightHeatmap';

interface WeightPanelProps {
  layers: LayerState[] | null;
}

export function WeightPanel({ layers }: WeightPanelProps) {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const numLayers = layers?.length || 0;

  // Derived safe index — no effect needed (computed from props + state)
  const safeSelected = numLayers > 0 ? Math.min(selectedLayer, numLayers - 1) : 0;

  return (
    <div className="weight-panel" role="group" aria-label="Weight heatmap visualization">
      <div className="panel-header">
        <span className="panel-icon" aria-hidden="true">🔥</span>
        <span>Weight Heatmap</span>
      </div>
      {numLayers > 0 && (
        <div className="layer-tabs" role="tablist" aria-label="Select layer">
          {Array.from({ length: numLayers }, (_, i) => (
            <button
              key={i}
              className={`tab ${safeSelected === i ? 'active' : ''}`}
              onClick={() => setSelectedLayer(i)}
              role="tab"
              aria-selected={safeSelected === i}
              aria-label={i === numLayers - 1 ? 'Output layer' : `Layer ${i + 1}`}
            >
              {i === numLayers - 1 ? 'Out' : `L${i + 1}`}
            </button>
          ))}
        </div>
      )}
      <WeightHeatmap
        layers={layers}
        selectedLayer={safeSelected}
        width={300}
        height={180}
      />
    </div>
  );
}

export default WeightPanel;
