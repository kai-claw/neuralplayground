import { useState } from 'react';
import type { LayerState } from '../nn/NeuralNetwork';
import WeightHeatmap from './WeightHeatmap';

interface WeightPanelProps {
  layers: LayerState[] | null;
}

export function WeightPanel({ layers }: WeightPanelProps) {
  const [selectedLayer, setSelectedLayer] = useState(0);
  const numLayers = layers?.length || 0;

  return (
    <div className="weight-panel">
      <div className="panel-header">
        <span className="panel-icon">🔥</span>
        <span>Weight Heatmap</span>
      </div>
      {numLayers > 0 && (
        <div className="layer-tabs">
          {Array.from({ length: numLayers }, (_, i) => (
            <button
              key={i}
              className={`tab ${selectedLayer === i ? 'active' : ''}`}
              onClick={() => setSelectedLayer(i)}
            >
              {i === numLayers - 1 ? 'Out' : `L${i + 1}`}
            </button>
          ))}
        </div>
      )}
      <WeightHeatmap
        layers={layers}
        selectedLayer={selectedLayer}
        width={300}
        height={180}
      />
    </div>
  );
}

export default WeightPanel;
