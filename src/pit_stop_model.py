"""
Section 3.2b - Pit Stop Duration Variability
==============================================
Heilmeier et al. (2020), Equation 7:
    t_pit_var ~ Fisk(shape, loc, scale)

The Fisk (log-logistic) distribution is fitted individually for every team
using the three parameters (c, loc, scale). A sample from the distribution
is added to the pit stop time when a driver performs a pit stop.
"""

import fastf1
import pandas as pd
import numpy as np
from scipy.stats import fisk

fastf1.Cache.enable_cache('../cache')


def extract_pit_stops(year: int, gp_name: str) -> pd.DataFrame:
    """Extract pit stop durations from a race session.
    
    Returns a DataFrame with columns: Driver, Team, PitDuration (seconds).
    PitDuration is the stationary time in the pit box.
    """
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False)
        laps = session.laps.copy()

        # Pit laps are laps where PitInTime is not NaN
        pit_laps = laps[laps['PitInTime'].notna() & laps['PitOutTime'].notna()].copy()

        if pit_laps.empty:
            return pd.DataFrame()

        # Pit duration = PitOutTime - PitInTime (this is the total pit lane time)
        pit_laps['PitDuration'] = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds()

        # Filter out unreasonable pit stops (< 10s likely data error, > 60s likely red flag/issue)
        pit_laps = pit_laps[(pit_laps['PitDuration'] > 10) & (pit_laps['PitDuration'] < 60)]

        return pit_laps[['Driver', 'Team', 'PitDuration']].copy()

    except Exception as e:
        print(f"Error extracting pit stops for {gp_name} {year}: {e}")
        return pd.DataFrame()


def compute_team_pit_variability(years: list, gps: list) -> dict:
    """Compute Fisk distribution parameters for pit stop variability per team.
    
    For each team, we:
    1. Collect all pit stop durations across the specified races
    2. Compute the team's median pit duration as the baseline
    3. Calculate the deviation: pit_var = actual_duration - team_median
    4. Fit a Fisk distribution to these deviations
    
    Returns:
        dict: {team_name: {'c': float, 'loc': float, 'scale': float, 'median_duration': float}}
    """
    all_pit_stops = []

    for year in years:
        for gp in gps:
            df = extract_pit_stops(year, gp)
            if not df.empty:
                all_pit_stops.append(df)

    if not all_pit_stops:
        print("No pit stop data found!")
        return {}

    combined = pd.concat(all_pit_stops, ignore_index=True)

    team_fisk_params = {}
    for team, team_data in combined.groupby('Team'):
        durations = team_data['PitDuration'].values
        if len(durations) < 5:
            continue

        median_duration = np.median(durations)
        # Compute deviations from the team's median pit time
        deviations = durations - median_duration

        try:
            c, loc, scale = fisk.fit(deviations)
            team_fisk_params[team] = {
                'c': c, 'loc': loc, 'scale': scale,
                'median_duration': median_duration
            }
        except Exception as e:
            print(f"Could not fit Fisk for team {team}: {e}")

    return team_fisk_params


if __name__ == "__main__":
    from data_pipeline import SEASONS

    test_gps = ['Bahrain']
    params = compute_team_pit_variability(SEASONS, test_gps)

    print("\n--- Pit Stop Fisk Parameters per Team ---")
    for team, p in sorted(params.items()):
        print(f"  {team}: median={p['median_duration']:.2f}s, "
              f"c={p['c']:.3f}, loc={p['loc']:.3f}, scale={p['scale']:.3f}")
