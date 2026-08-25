import React from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';

export default function TyreDegradation({ analytics }) {
  if (!analytics || !analytics.tyre_degradation) return null;

  const rates = analytics.tyre_degradation.degradation_rates || analytics.tyre_degradation;
  const data = [];
  for (let lap = 0; lap <= 35; lap++) {
    data.push({
      lap,
      SOFT: Math.round((rates.SOFT || 0.10) * lap * 100) / 100,
      MEDIUM: Math.round((rates.MEDIUM || 0.06) * lap * 100) / 100,
      HARD: Math.round((rates.HARD || 0.03) * lap * 100) / 100,
    });
  }

  // Noise parameters for display
  const sigmaEta = analytics.tyre_degradation.sigma_eta;
  const sigmaEps = analytics.tyre_degradation.sigma_eps;
  const skewness = analytics.tyre_degradation.skewness;

  return (
    <div className="card tyre-card">
      <h3 className="card-title">
        <span className="card-icon">🔧</span>
        Tyre Degradation Model
      </h3>
      
      <div className="tyre-chart-container">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 5 }}>
            <XAxis
              dataKey="lap"
              stroke="#666"
              fontSize={10}
              tick={{ fill: '#888' }}
              label={{ value: 'Stint Lap', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 10 }}
            />
            <YAxis
              stroke="#666"
              fontSize={10}
              tick={{ fill: '#888' }}
              label={{ value: 'Time Loss (s)', angle: -90, position: 'insideLeft', fill: '#666', fontSize: 10 }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1a1c22',
                border: '1px solid #333',
                borderRadius: 6,
                color: '#fff',
                fontSize: 12,
              }}
              formatter={(val, name) => [`${val.toFixed(2)}s`, name]}
            />
            <Legend wrapperStyle={{ fontSize: 11, color: '#aaa' }} />
            <Line type="monotone" dataKey="SOFT" stroke="#E80020" dot={false} strokeWidth={2.5} />
            <Line type="monotone" dataKey="MEDIUM" stroke="#FFD700" dot={false} strokeWidth={2.5} />
            <Line type="monotone" dataKey="HARD" stroke="#CCCCCC" dot={false} strokeWidth={2.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* State-space model parameters */}
      <div className="tyre-params">
        <div className="tyre-param">
          <span className="tp-label">σ_η (Process Noise)</span>
          <span className="tp-value">{sigmaEta ? sigmaEta.toFixed(3) : '—'}</span>
        </div>
        <div className="tyre-param">
          <span className="tp-label">σ_ε (Obs. Noise)</span>
          <span className="tp-value">{sigmaEps ? sigmaEps.toFixed(3) : '—'}</span>
        </div>
        <div className="tyre-param">
          <span className="tp-label">Skewness (α)</span>
          <span className="tp-value">{skewness ? skewness.toFixed(3) : '—'}</span>
        </div>
        <div className="tyre-param">
          <span className="tp-label" style={{color: '#E80020'}}>ν_SOFT</span>
          <span className="tp-value">{rates.SOFT ? `${rates.SOFT.toFixed(4)} s/lap` : '—'}</span>
        </div>
        <div className="tyre-param">
          <span className="tp-label" style={{color: '#FFD700'}}>ν_MEDIUM</span>
          <span className="tp-value">{rates.MEDIUM ? `${rates.MEDIUM.toFixed(4)} s/lap` : '—'}</span>
        </div>
        <div className="tyre-param">
          <span className="tp-label" style={{color: '#CCCCCC'}}>ν_HARD</span>
          <span className="tp-value">{rates.HARD ? `${rates.HARD.toFixed(4)} s/lap` : '—'}</span>
        </div>
      </div>
    </div>
  );
}
