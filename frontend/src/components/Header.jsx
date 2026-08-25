import React from 'react';
import { motion } from 'framer-motion';

const TEAM_COLORS = {
  'Red Bull Racing': '#3671C6',
  'Mercedes': '#27F4D2',
  'Ferrari': '#E80020',
  'McLaren': '#FF8000',
  'Aston Martin': '#229971',
  'Alpine': '#0093CC',
  'Williams': '#64C4FF',
  'RB': '#6692FF',
  'Kick Sauber': '#52E252',
  'Haas F1 Team': '#B6BABD',
};

export default function Header({ config, selectedYear, setSelectedYear, selectedGP, setSelectedGP, tracks }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <motion.div 
          className="logo-mark"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: 'spring', stiffness: 260, damping: 20 }}
        >
          <span className="logo-icon">🏎️</span>
        </motion.div>
        <div className="header-titles">
          <h1>F1 STRATEGY<span className="accent">LAB</span></h1>
          <p className="header-subtitle">Monte Carlo Race Simulation · Reinforcement Learning Optimization</p>
        </div>
      </div>

      <div className="header-controls">
        <div className="control-group">
          <label>Season</label>
          <select
            value={selectedYear}
            onChange={(e) => setSelectedYear(parseInt(e.target.value))}
            className="header-select"
          >
            <option value={2022}>2022</option>
            <option value={2023}>2023</option>
            <option value={2024}>2024</option>
            <option value={2025}>2025</option>
          </select>
        </div>

        <div className="control-group">
          <label>Grand Prix</label>
          <select
            value={selectedGP}
            onChange={(e) => setSelectedGP(e.target.value)}
            className="header-select"
          >
            {tracks.map(t => (
              <option key={t.name} value={t.name}>{t.name} ({t.total_laps} laps)</option>
            ))}
          </select>
        </div>
      </div>
    </header>
  );
}

export { TEAM_COLORS };
