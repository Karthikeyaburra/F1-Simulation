import React from 'react';
import { motion } from 'framer-motion';
import { TEAM_COLORS } from './Header';

export default function DriverGrid({ drivers, selectedDriver, setSelectedDriver }) {
  return (
    <div className="card">
      <h3 className="card-title">
        <span className="card-icon">👤</span>
        Grid Lineup
      </h3>
      <div className="driver-grid">
        {drivers.map((d, i) => {
          const isActive = selectedDriver === d.abbreviation;
          const teamColor = TEAM_COLORS[d.team] || '#777';
          return (
            <motion.button
              key={d.abbreviation}
              className={`driver-chip ${isActive ? 'active' : ''}`}
              style={{
                '--team-color': teamColor,
                borderLeftColor: teamColor,
              }}
              onClick={() => setSelectedDriver(d.abbreviation)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.02 }}
            >
              <span className="chip-pos">P{d.grid_pos}</span>
              <span className="chip-name">{d.abbreviation}</span>
              <span className="chip-team-dot" style={{ backgroundColor: teamColor }} />
            </motion.button>
          );
        })}
      </div>
    </div>
  );
}
