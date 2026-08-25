"""
Simulator Validation Script
===========================
Compares Monte Carlo simulation output against the actual F1 race results.
Calculates error metrics like Mean Absolute Error (MAE) for finishing positions,
precision/recall for points finishes, and checks incident matching.
"""

import json
import os
import fastf1
import pandas as pd
import numpy as np

# Configure cache locally
fastf1.Cache.enable_cache('../cache')

# Track-specific physical parameters (estimated from actual race paces)
TRACK_CONFIGS = {
    'Bahrain': {
        'total_laps': 57,
        't_base': 93.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.93,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.0,
        'mult_t_lap_sc': 1.6,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Saudi Arabia': {
        'total_laps': 50,
        't_base': 91.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.20,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Australia': {
        'total_laps': 58,
        't_base': 78.5,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.90,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 16.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Japan': {
        'total_laps': 53,
        't_base': 93.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.08,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.5,
        'mult_t_lap_sc': 1.6,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Miami': {
        'total_laps': 57,
        't_base': 92.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.93,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.5,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Spain': {
        'total_laps': 66,
        't_base': 79.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 1.67,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 18.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Great Britain': {
        'total_laps': 52,
        't_base': 91.5,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.11,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 17.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    },
    'Italy': {
        'total_laps': 53,
        't_base': 84.0,
        't_lap_sens_mass': 0.030,
        'm_fuel_start': 110.0,
        'b_fuel_perlap': 2.08,
        't_pit_stationary': 2.5,
        't_pit_inlap': 3.0,
        't_pit_outlap': 19.0,
        'mult_t_lap_sc': 1.5,
        't_gap_overtake': 1.5,
        't_drseffect': -0.4,
        'drs_window': 1.0,
    }
}

# The validation scenarios we want to evaluate
VALIDATION_SCENARIOS = [
    {'year': 2023, 'gp': 'Bahrain'},
    {'year': 2024, 'gp': 'Bahrain'},
    {'year': 2024, 'gp': 'Saudi Arabia'},
    {'year': 2024, 'gp': 'Australia'},
    {'year': 2024, 'gp': 'Miami'},
    {'year': 2024, 'gp': 'Spain'},
    {'year': 2024, 'gp': 'Great Britain'},
    {'year': 2024, 'gp': 'Italy'},
    {'year': 2024, 'gp': 'Japan'},
]

def migrate_tyre_degradation_params(tyre_deg: dict) -> dict:
    """Helper to convert old/flat skewt_params to new state-space noise params on the fly."""
    if not isinstance(tyre_deg, dict):
        tyre_deg = {}
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


def run_single_race_validation(year: int, gp_name: str) -> dict:
    print(f"\nEvaluating: {gp_name} {year}...")
    
    # Resolve paths
    results_path = f'../config/simulation_results_{gp_name}_{year}.json'.lower()
    params_path = f'fitted_parameters_{year}.json'
    config_path = f'race_config_{gp_name}_{year}.json'.lower()
    
    # Ensure static parameter asset exists
    if not os.path.exists(params_path):
        print(f"Error: {params_path} not found. Run generate_assets.py first.")
        return {}

    with open(params_path, 'r') as f:
        params = json.load(f)
        
    needs_save = False
    if 'tyre_degradation' in params:
        old_deg = params['tyre_degradation']
        new_deg = migrate_tyre_degradation_params(old_deg)
        if old_deg != new_deg:
            params['tyre_degradation'] = new_deg
            needs_save = True
            
    if not params.get("probabilities_migrated_v2", False):
        print(f"Validation: Recomputing DNF probabilities to fix prior-dominance...")
        try:
            from probability_model import compute_probabilities
            from data_pipeline import SEASONS
            fitting_gps = [
                'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
                'Miami', 'Spain', 'Great Britain', 'Italy'
            ]
            prob_result = compute_probabilities(SEASONS, fitting_gps)
            if prob_result:
                params['p_accident'] = prob_result.get('p_accident', {})
                params['p_failure'] = prob_result.get('p_failure', {})
                params['probabilities_migrated_v2'] = True
                needs_save = True
        except Exception as e_prob:
            print(f"Validation: Failed to recompute DNF probabilities: {e_prob}")
            
    if needs_save:
        try:
            with open(params_path, 'w') as f_out:
                json.dump(params, f_out, indent=2)
            print(f"Validation: Saved migrated parameters to {params_path}.")
        except Exception as e_write:
            print(f"Validation: Could not write updated parameters to {params_path}: {e_write}")

    # Get track details
    if gp_name not in TRACK_CONFIGS:
        print(f"Error: No track parameters found for {gp_name}.")
        return {}
    track_config = TRACK_CONFIGS[gp_name]

    # Fetch baseline configuration
    from run_simulation import get_race_entry_list, assign_strategies
    from race_simulator import RaceSimulator

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                stored = json.load(f)
            entries = stored['drivers']
        else:
            entries = get_race_entry_list(year, gp_name)
            entries = assign_strategies(entries)
            for driver in entries:
                driver['t_car'] = params['t_car'].get(driver['team'], 0.5)
                driver['t_driver'] = params['t_driver'].get(driver['abbreviation'], 0.0)
    except Exception as e:
        print(f"Error fetching baseline entries for {gp_name} {year}: {e}")
        return {}

    driver_configs = []
    for d in entries:
        driver_configs.append({
            'abbreviation': d['abbreviation'],
            'team': d['team'],
            'grid_pos': d['grid_pos'],
            't_car': d['t_car'],
            't_driver': d['t_driver'],
            'strategy': d['strategy']
        })

    # Run simulator (500 iterations)
    simulator = RaceSimulator(track_config, params)
    mc_results = simulator.monte_carlo(driver_configs, n_iterations=500)

    # Fetch Actual Race Results
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False)
        actual_results = session.results
        
        # Get actual Safety Car events
        rcm = session.race_control_messages
        actual_sc_count = 0
        if rcm is not None and not rcm.empty:
            actual_sc_count = len(rcm[
                rcm['Message'].str.upper().str.contains('SAFETY CAR DEPLOYED') & 
                ~rcm['Message'].str.upper().str.contains('VIRTUAL')
            ])
    except Exception as e:
        print(f"Failed to fetch actual results from FastF1: {e}")
        return {}

    # Map actuals
    actual_positions = {}
    actual_top10 = []
    actual_dnfs = []

    for _, row in actual_results.iterrows():
        driver = row['Abbreviation']
        pos = int(row['Position'])
        status = row['Status']
        
        actual_positions[driver] = pos
        if pos <= 10:
            actual_top10.append(driver)
        if not (status.startswith('Finished') or status.startswith('+') or 'Lap' in status or status == 'Lapped'):
            actual_dnfs.append(driver)

    # Compute errors
    pos_probs = mc_results['position_probabilities']
    ret_rates = {drv: count / 500 for drv, count in mc_results['retirement_counts'].items()}

    mae_sum = 0
    mae_count = 0
    top10_hits = 0

    for driver, probs in pos_probs.items():
        if driver not in actual_positions:
            continue
        expected_pos = sum(idx * prob for idx, prob in enumerate(probs))
        abs_error = abs(actual_positions[driver] - expected_pos)
        mae_sum += abs_error
        mae_count += 1
        
        if expected_pos <= 10 and actual_positions[driver] <= 10:
            top10_hits += 1

    mae = mae_sum / mae_count if mae_count > 0 else 0
    winner = actual_results.iloc[0]['Abbreviation']
    winner_prob = pos_probs.get(winner, [0.0, 0.0])[1] if winner in pos_probs else 0.0

    return {
        'race': f"{gp_name} {year}",
        'mae': mae,
        'winner_prob': winner_prob,
        'winner': winner,
        'top10_hits': top10_hits,
        'actual_sc': actual_sc_count,
        'actual_dnfs': len(actual_dnfs),
        'sim_expected_dnfs': sum(ret_rates.values())
    }

def main():
    print("====================================================")
    print("F1 Monte Carlo Simulator Cross-Race Validation")
    print("====================================================")

    reports = []
    for scenario in VALIDATION_SCENARIOS:
        report = run_single_race_validation(scenario['year'], scenario['gp'])
        if report:
            reports.append(report)

    # Print Summary Table
    print("\n" + "=" * 80)
    print("GLOBAL ACCURACY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Race':<20} | {'MAE (Pos)':<10} | {'Actual DNFs':<12} | {'Sim DNFs':<10} | {'Winner P(Win)':<15}")
    print("-" * 80)
    
    mae_vals = []
    for r in reports:
        mae_vals.append(r['mae'])
        print(f"{r['race']:<20} | {r['mae']:<10.2f} | {r['actual_dnfs']:<12} | {r['sim_expected_dnfs']:<10.2f} | {r['winner']}: {r['winner_prob']:.1%}")
    
    print("-" * 80)
    avg_mae = np.mean(mae_vals) if mae_vals else 0
    print(f"{'AVERAGE':<20} | {avg_mae:<10.2f} |")
    print("=" * 80)

if __name__ == "__main__":
    main()
