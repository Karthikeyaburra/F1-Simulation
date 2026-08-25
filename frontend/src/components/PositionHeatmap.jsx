import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { TEAM_COLORS } from './Header';

/**
 * Position Probability Heatmap
 * Renders a driver × position grid where cell intensity represents 
 * the probability of that driver finishing in that position.
 * This is the signature visualization of the project.
 */
export default function PositionHeatmap({ simResults, drivers, selectedDriver, setSelectedDriver }) {
  if (!simResults) return null;

  const driverOrder = useMemo(() => {
    // Sort drivers by expected finish position
    return Object.keys(simResults.position_probabilities)
      .map(drv => {
        const probs = simResults.position_probabilities[drv];
        let expected = 0;
        probs.forEach((p, idx) => { expected += idx * p; });
        return { drv, expected };
      })
      .sort((a, b) => a.expected - b.expected)
      .map(d => d.drv);
  }, [simResults]);

  const nDrivers = driverOrder.length;
  const positions = Array.from({ length: nDrivers }, (_, i) => i + 1);

  // Find max probability for color scaling
  const maxProb = useMemo(() => {
    let max = 0;
    for (const drv of driverOrder) {
      const probs = simResults.position_probabilities[drv];
      for (let i = 1; i <= nDrivers; i++) {
        if (probs[i] > max) max = probs[i];
      }
    }
    return max || 1;
  }, [simResults, driverOrder, nDrivers]);

  const getHeatColor = (prob) => {
    if (prob === 0) return 'rgba(255,255,255,0.02)';
    const intensity = prob / maxProb;
    // Gradient from dark blue → cyan → yellow → red for high probabilities
    if (intensity < 0.25) {
      const t = intensity / 0.25;
      return `rgba(30, 60, 180, ${0.15 + t * 0.35})`;
    } else if (intensity < 0.5) {
      const t = (intensity - 0.25) / 0.25;
      return `rgba(${30 + t * 20}, ${60 + t * 140}, ${180 - t * 30}, ${0.5 + t * 0.2})`;
    } else if (intensity < 0.75) {
      const t = (intensity - 0.5) / 0.25;
      return `rgba(${50 + t * 205}, ${200 - t * 30}, ${150 - t * 100}, ${0.7 + t * 0.15})`;
    } else {
      const t = (intensity - 0.75) / 0.25;
      return `rgba(${255}, ${170 - t * 140}, ${50 - t * 50}, ${0.85 + t * 0.15})`;
    }
  };

  return (
    <div className="card heatmap-card">
      <h3 className="card-title">
        <span className="card-icon">🔥</span>
        Position Probability Heatmap
        <span className="card-badge">{nDrivers} drivers × {nDrivers} positions</span>
      </h3>

      <div className="heatmap-scroll">
        <div className="heatmap-grid" style={{ gridTemplateColumns: `120px repeat(${nDrivers}, 1fr)` }}>
          {/* Header row */}
          <div className="hm-corner">DRV / POS</div>
          {positions.map(pos => (
            <div key={pos} className="hm-col-header">P{pos}</div>
          ))}

          {/* Data rows */}
          {driverOrder.map((drv, rowIdx) => {
            const probs = simResults.position_probabilities[drv];
            const driverObj = drivers.find(d => d.abbreviation === drv);
            const teamColor = TEAM_COLORS[driverObj?.team] || '#777';
            const isSelected = drv === selectedDriver;

            return (
              <React.Fragment key={drv}>
                <div
                  className={`hm-row-header ${isSelected ? 'selected' : ''}`}
                  onClick={() => setSelectedDriver(drv)}
                  style={{ borderLeftColor: teamColor }}
                >
                  <span className="hm-drv-name" style={{ color: isSelected ? teamColor : '' }}>
                    {drv}
                  </span>
                </div>
                {positions.map(pos => {
                  const prob = probs[pos] || 0;
                  return (
                    <motion.div
                      key={`${drv}-${pos}`}
                      className="hm-cell"
                      style={{ backgroundColor: getHeatColor(prob) }}
                      title={`${drv} P${pos}: ${(prob * 100).toFixed(1)}%`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: rowIdx * 0.02 + pos * 0.005 }}
                    >
                      {prob >= 0.01 && (
                        <span className="hm-cell-val">
                          {prob >= 0.1 ? `${(prob * 100).toFixed(0)}` : `${(prob * 100).toFixed(1)}`}
                        </span>
                      )}
                    </motion.div>
                  );
                })}
              </React.Fragment>
            );
          })}
        </div>
      </div>
      
      <div className="heatmap-legend">
        <span className="legend-label">Low</span>
        <div className="legend-gradient" />
        <span className="legend-label">High</span>
        <span className="legend-unit">probability %</span>
      </div>
    </div>
  );
}
