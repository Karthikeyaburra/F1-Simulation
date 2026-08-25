"""
Static Asset Generator
======================
Runs the FastF1 data aggregation and parameter fitting locally,
then saves the outputs to static JSON files. This removes the
FastF1 API dependency from the web application during deployment.
"""

import json
import os
import fastf1
import numpy as np

# Configure cache locally
fastf1.Cache.enable_cache('../cache')

from data_pipeline import SEASONS, starting_performance, aggregate_seasons_data, driver_stints
from pit_stop_model import compute_team_pit_variability
from probability_model import compute_probabilities
from fcy_model import compute_fcy_probabilities
from run_simulation import get_race_entry_list, compute_car_driver_offsets, estimate_tyre_degradation, assign_strategies

FITTING_GPS = [
    'Bahrain', 'Saudi Arabia', 'Australia', 'Japan',
    'Miami', 'Spain', 'Great Britain', 'Italy'
]

SIM_YEAR = 2024
SIM_GP = 'Bahrain'

def main():
    print("====================================================")
    print("Generating Static Simulation Assets...")
    print("====================================================")

    # General fits over all seasons to maximize statistical sample size
    print("\nFitting General Starting Performance (All Seasons)...")
    a_avg_fitted, driver_start = starting_performance(SEASONS, FITTING_GPS)
    
    print("\nFitting General Lap Time Variability (Fisk, All Seasons)...")
    master_df = aggregate_seasons_data(SEASONS, FITTING_GPS)
    driver_fisk = {}
    track_fisk = {}
    if not master_df.empty:
        stint_results = driver_stints(master_df)
        driver_fisk = stint_results['driver_fisk_params']
        track_fisk = stint_results['track_fisk_params']

    print("\nFitting General Pit Stop Variability (Fisk, All Seasons)...")
    team_pit_fisk = compute_team_pit_variability(SEASONS, FITTING_GPS)

    print("\nComputing General Accident & Failure Probabilities (All Seasons)...")
    prob_result = compute_probabilities(SEASONS, FITTING_GPS)
    p_accident = prob_result.get('p_accident', {})
    p_failure = prob_result.get('p_failure', {})

    print("\nComputing General Safety Car Probabilities (All Seasons)...")
    fcy = compute_fcy_probabilities(SEASONS, FITTING_GPS)

    print("\nEstimating General Tyre Degradation Rates (All Seasons)...")
    tyre_deg = estimate_tyre_degradation(SEASONS, FITTING_GPS)

    # Now generate year-specific files with year-specific relative pace offsets
    for target_year in SEASONS:
        print(f"\n--- Processing Year-Specific Offsets for {target_year} ---")
        
        # Calculate t_car and t_driver exclusively for the target year
        offsets = compute_car_driver_offsets(years=[target_year], gps=FITTING_GPS)
        
        fitted_params = {
            'a_avg_start': float(a_avg_fitted) if a_avg_fitted else 11.2,
            'driver_start_params': driver_start or {},
            'driver_fisk_params': driver_fisk,
            'track_fisk_params': track_fisk,
            'team_pit_fisk_params': team_pit_fisk,
            'p_accident': p_accident,
            'p_failure': p_failure,
            'fcy_params': fcy,
            't_car': offsets['t_car'],
            't_driver': offsets['t_driver'],
            'tyre_degradation': tyre_deg
        }

        # Save year-specific fitted parameters
        params_path = f'fitted_parameters_{target_year}.json'
        with open(params_path, 'w') as f:
            json.dump(fitted_params, f, indent=2)
        print(f"Saved parameters to: {os.path.abspath(params_path)}")

        # Fetch baseline entry list and configurations for target year
        print(f"Fetching baseline entry list for {SIM_GP} {target_year}...")
        try:
            entries = get_race_entry_list(target_year, SIM_GP)
            entries = assign_strategies(entries)
            for driver in entries:
                driver['t_car'] = offsets['t_car'].get(driver['team'], 0.5)
                driver['t_driver'] = offsets['t_driver'].get(driver['abbreviation'], 0.0)

            config_path = f'race_config_{SIM_GP}_{target_year}.json'.lower()
            with open(config_path, 'w') as f:
                json.dump({
                    "year": target_year,
                    "gp_name": SIM_GP,
                    "drivers": entries
                }, f, indent=2)
            print(f"Saved baseline race config to: {os.path.abspath(config_path)}")
        except Exception as e:
            print(f"Failed to generate baseline configuration for {target_year}: {e}")

    print("\nStatic asset generation complete!")

if __name__ == "__main__":
    main()
