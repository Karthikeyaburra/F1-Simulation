"""
FastAPI Backend Application
===========================
Serves APIs for the React frontend:
- GET /api/config: Returns available tracks, seasons, and baseline drivers
- GET /api/analytics: Returns fitted Fisk and starting performance parameters
- POST /api/simulate: Runs the Monte Carlo simulation and returns results
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import os

from data_pipeline import SEASONS, starting_performance, aggregate_seasons_data, driver_stints
from pit_stop_model import compute_team_pit_variability
from probability_model import compute_probabilities
from fcy_model import compute_fcy_probabilities
from race_simulator import RaceSimulator
from run_simulation import get_race_entry_list, compute_car_driver_offsets, estimate_tyre_degradation, assign_strategies, RACE_CONFIG

app = FastAPI(title="F1 Monte Carlo Simulation API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StintSelection(BaseModel):
    lap: int
    compound: str


class DriverStrategyConfig(BaseModel):
    abbreviation: str
    team: str
    grid_pos: int
    t_car: float
    t_driver: float
    strategy: list[StintSelection]


class SimulationRequest(BaseModel):
    year: int
    gp_name: str
    n_iterations: int = 500
    drivers: list[DriverStrategyConfig]


# Global caches loaded at startup, keyed by year
FITTED_PARAMS = {}
PRE_RACE_CONFIG = {}


def migrate_tyre_degradation_params(tyre_deg: dict) -> dict:
    """Helper to convert old/flat skewt_params to new state-space noise params on the fly."""
    if not isinstance(tyre_deg, dict):
        tyre_deg = {}
        
    # If it is in the simple flat format (e.g., {'SOFT': 0.12, ...})
    if 'SOFT' in tyre_deg and 'degradation_rates' not in tyre_deg:
        tyre_deg = {
            'degradation_rates': tyre_deg,
            'sigma_eta': 0.10,
            'sigma_eps': 0.30,
            'skewness': 1.5
        }
        
    if 'skewt_params' in tyre_deg and 'sigma_eta' not in tyre_deg:
        skewt = tyre_deg['skewt_params']
        scales = [v['scale'] for v in skewt.values() if isinstance(v, dict) and 'scale' in v]
        sigma_eps = float(np.mean(scales)) if scales else 0.30
        
        sigma_eta = 0.10
        
        ncs = [v['nc'] for v in skewt.values() if isinstance(v, dict) and 'nc' in v]
        skewness = float(np.mean(ncs)) if ncs else 1.5
        
        sigma_eps = np.clip(sigma_eps, 0.15, 0.60)
        sigma_eta = np.clip(sigma_eta, 0.02, 0.25)
        skewness = np.clip(skewness, 0.5, 3.0)
        
        return {
            'degradation_rates': tyre_deg.get('degradation_rates', {'SOFT': 0.10, 'MEDIUM': 0.06, 'HARD': 0.03}),
            'sigma_eta': float(sigma_eta),
            'sigma_eps': float(sigma_eps),
            'skewness': float(skewness)
        }
    return tyre_deg


# Load static assets for all seasons immediately on startup
for yr in SEASONS:
    params_file = f'fitted_parameters_{yr}.json'
    config_file = f'race_config_bahrain_{yr}.json'

    if os.path.exists(params_file):
        print(f"Startup: Loading pre-computed parameters for {yr} from {params_file}...")
        try:
            with open(params_file, 'r') as f:
                data = json.load(f)
            
            needs_save = False

            # Migrate tyre degradation parameters on-the-fly
            if 'tyre_degradation' in data:
                old_deg = data['tyre_degradation']
                new_deg = migrate_tyre_degradation_params(old_deg)
                if old_deg != new_deg:
                    data['tyre_degradation'] = new_deg
                    needs_save = True
                    print(f"Startup: Migrating tyre degradation parameters for {yr}...")

            # Migrate accident & failure probabilities to fix prior dominance on-the-fly
            if not data.get("probabilities_migrated_v2", False):
                print(f"Startup: Recomputing DNF probabilities for {yr} to fix prior-dominance...")
                try:
                    from probability_model import compute_probabilities
                    fitting_gps = [
                        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
                        'Miami', 'Spain', 'Great Britain', 'Italy'
                    ]
                    prob_result = compute_probabilities(SEASONS, fitting_gps)
                    if prob_result:
                        data['p_accident'] = prob_result.get('p_accident', {})
                        data['p_failure'] = prob_result.get('p_failure', {})
                        data['probabilities_migrated_v2'] = True
                        needs_save = True
                        print(f"Startup: Recomputed DNF probabilities successfully.")
                except Exception as e_prob:
                    print(f"Startup: Failed to recompute DNF probabilities for {yr}: {e_prob}")

            if needs_save:
                try:
                    with open(params_file, 'w') as f_out:
                        json.dump(data, f_out, indent=2)
                    print(f"Startup: Successfully saved updated parameters to {params_file}.")
                except Exception as e_write:
                    print(f"Startup: Could not write back updated parameters to {params_file}: {e_write}")
            
            FITTED_PARAMS[yr] = data
        except Exception as e:
            print(f"Error loading {params_file} on startup: {e}")
    else:
        print(f"WARNING: {params_file} not found. Run generate_assets.py to pre-compute.")

    if os.path.exists(config_file):
        print(f"Startup: Loading pre-computed race config for {yr} from {config_file}...")
        try:
            with open(config_file, 'r') as f:
                PRE_RACE_CONFIG[yr] = json.load(f)
        except Exception as e:
            print(f"Error loading {config_file} on startup: {e}")
    else:
        print(f"WARNING: {config_file} not found. Run generate_assets.py to pre-compute.")


def load_all_fitted_parameters(year: int):
    """Returns pre-loaded static parameters for the target year or falls back to dynamic fitting."""
    global FITTED_PARAMS
    if year in FITTED_PARAMS:
        return FITTED_PARAMS[year]

    print(f"Asset missing: fitting parameters dynamically for {year} (fallback)...")
    fitting_gps = [
        'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
        'Miami', 'Spain', 'Great Britain', 'Italy'
    ]

    # Fits
    a_avg_fitted, driver_start = starting_performance([year], fitting_gps)
    master_df = aggregate_seasons_data([year], fitting_gps)
    
    driver_fisk = {}
    track_fisk = {}
    if not master_df.empty:
        stint_results = driver_stints(master_df)
        driver_fisk = stint_results['driver_fisk_params']
        track_fisk = stint_results['track_fisk_params']

    team_pit_fisk = compute_team_pit_variability([year], fitting_gps)
    prob_result = compute_probabilities([year], fitting_gps)
    fcy = compute_fcy_probabilities([year], fitting_gps)
    offsets = compute_car_driver_offsets([year], fitting_gps)
    tyre_deg = estimate_tyre_degradation([year], fitting_gps)

    FITTED_PARAMS[year] = {
        'a_avg_start': a_avg_fitted or 11.2,
        'driver_start_params': driver_start or {},
        'driver_fisk_params': driver_fisk,
        'track_fisk_params': track_fisk,
        'team_pit_fisk_params': team_pit_fisk,
        'p_accident': prob_result.get('p_accident', {}),
        'p_failure': prob_result.get('p_failure', {}),
        'fcy_params': fcy,
        't_car': offsets['t_car'],
        't_driver': offsets['t_driver'],
        'tyre_degradation': tyre_deg
    }
    return FITTED_PARAMS[year]


@app.get("/api/config")
async def get_config(year: int = 2024, gp_name: str = "Bahrain"):
    """Get the initial entry list and configurations for the race."""
    try:
        # Serve immediately from memory if available for target year
        if year in PRE_RACE_CONFIG:
            stored = PRE_RACE_CONFIG[year]
            return {
                "year": stored["year"],
                "gp_name": stored["gp_name"],
                "laps": RACE_CONFIG['total_laps'],
                "drivers": stored["drivers"],
                "compounds": ["SOFT", "MEDIUM", "HARD"]
            }

        # Fallback to dynamic loading
        print(f"Asset missing: dynamically loading race config for {year}...")
        params = load_all_fitted_parameters(year)
        entries = get_race_entry_list(year, gp_name)
        entries = assign_strategies(entries)
        for driver in entries:
            driver['t_car'] = params['t_car'].get(driver['team'], 0.5)
            driver['t_driver'] = params['t_driver'].get(driver['abbreviation'], 0.0)

        return {
            "year": year,
            "gp_name": gp_name,
            "laps": RACE_CONFIG['total_laps'],
            "drivers": entries,
            "compounds": ["SOFT", "MEDIUM", "HARD"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analytics")
async def get_analytics(year: int = 2024):
    """Retrieve the pre-computed parameter distributions for analysis plots."""
    try:
        params = load_all_fitted_parameters(year)
        return {
            "a_avg_start": params['a_avg_start'],
            "tyre_degradation": params['tyre_degradation'],
            "driver_start_params": params['driver_start_params'],
            "driver_fisk_params": params['driver_fisk_params'],
            "team_pit_fisk_params": params['team_pit_fisk_params']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate")
async def simulate(req: SimulationRequest):
    """Execute the Monte Carlo simulation."""
    try:
        params = load_all_fitted_parameters(req.year)
        
        # Map frontend payload back into simulation format
        driver_configs = []
        for d in req.drivers:
            driver_configs.append({
                'abbreviation': d.abbreviation,
                'team': d.team,
                'grid_pos': d.grid_pos,
                't_car': d.t_car,
                't_driver': d.t_driver,
                'strategy': [{'lap': s.lap, 'compound': s.compound} for s in d.strategy]
            })

        # Run simulator
        config_with_track = RACE_CONFIG.copy()
        config_with_track['track_name'] = req.gp_name
        simulator = RaceSimulator(config_with_track, params)
        mc_results = simulator.monte_carlo(driver_configs, n_iterations=req.n_iterations)

        # Convert numpy arrays to standard lists for JSON output
        pos_probs = {}
        for drv, probs in mc_results['position_probabilities'].items():
            pos_probs[drv] = probs.tolist()

        return {
            "position_probabilities": pos_probs,
            "retirement_counts": mc_results['retirement_counts'],
            "n_iterations": req.n_iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
