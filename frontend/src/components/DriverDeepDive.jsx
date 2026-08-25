import React from 'react';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { TEAM_COLORS } from './Header';

export default function DriverDeepDive({ simResults, drivers, selectedDriver, nIterations }) {
  if (!simResults || !selectedDriver) return null;

  const probs = simResults.position_probabilities[selectedDriver];
  if (!probs) return null;

  const dObj = drivers.find(d => d.abbreviation === selectedDriver);
  const teamColor = TEAM_COLORS[dObj?.team] || '#e10600';

  // Build bar chart data
  const nDrivers = drivers.length;
  const chartData = [];
  for (let pos = 1; pos <= nDrivers; pos++) {
    chartData.push({
      position: `P${pos}`,
      probability: Math.round((probs[pos] || 0) * 1000) / 10,
      raw: probs[pos] || 0,
    });
  }

  // Key stats
  let expFinish = 0;
  probs.forEach((p, idx) => { expFinish += idx * p; });
  const pWin = probs[1] || 0;
  const pPodium = (probs[1] || 0) + (probs[2] || 0) + (probs[3] || 0);
  const pPoints = Array.from({ length: 10 }, (_, i) => probs[i + 1] || 0).reduce((a, b) => a + b, 0);
  const pRetire = (simResults.retirement_counts[selectedDriver] || 0) / nIterations;

  // Expected F1 points
  const F1_POINTS = { 1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1 };
  let expPoints = 0;
  for (let i = 1; i <= 10; i++) {
    expPoints += (probs[i] || 0) * (F1_POINTS[i] || 0);
  }

  // Find most likely position
  let modePos = 1;
  let modeProb = 0;
  for (let i = 1; i <= nDrivers; i++) {
    if ((probs[i] || 0) > modeProb) {
      modeProb = probs[i];
      modePos = i;
    }
  }

  const stats = [
    { label: 'Expected Finish', value: expFinish.toFixed(1), color: '#e07000' },
    { label: 'Most Likely', value: `P${modePos}`, color: teamColor },
    { label: 'P(Win)', value: `${(pWin * 100).toFixed(1)}%`, color: '#00cc44' },
    { label: 'P(Podium)', value: `${(pPodium * 100).toFixed(1)}%`, color: '#0099ff' },
    { label: 'P(Points)', value: `${(pPoints * 100).toFixed(1)}%`, color: '#27F4D2' },
    { label: 'Exp. Points', value: expPoints.toFixed(1), color: '#FFD700' },
    { label: 'P(DNF)', value: `${(pRetire * 100).toFixed(1)}%`, color: '#ff073a' },
    { label: 'Grid', value: `P${dObj?.grid_pos}`, color: '#aaa' },
  ];

  return (
    <div className="card deep-dive-card">
      <h3 className="card-title">
        <span className="card-icon">📊</span>
        Driver Analysis — <span style={{ color: teamColor }}>{selectedDriver}</span>
        <span className="team-label">{dObj?.team}</span>
      </h3>

      {/* Stats Grid */}
      <div className="dd-stats-grid">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            className="dd-stat"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <span className="dd-stat-label">{s.label}</span>
            <span className="dd-stat-value" style={{ color: s.color }}>{s.value}</span>
          </motion.div>
        ))}
      </div>

      {/* Position Distribution Chart */}
      <div className="dd-chart">
        <h4>Position Probability Distribution</h4>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
            <XAxis dataKey="position" stroke="#666" fontSize={10} tick={{ fill: '#888' }} />
            <YAxis stroke="#666" fontSize={10} tick={{ fill: '#888' }} unit="%" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1c22',
                border: '1px solid #333',
                borderRadius: 6,
                color: '#fff',
                fontSize: 12,
              }}
              formatter={(val) => [`${val}%`, 'Probability']}
            />
            <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`}
                  fill={entry.raw > 0.05 ? teamColor : 'rgba(255,255,255,0.15)'}
                  opacity={0.3 + entry.raw * 3}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
