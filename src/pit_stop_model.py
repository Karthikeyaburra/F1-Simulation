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

        result = pit_laps[['Driver', 'Team', 'PitDuration']].copy()
        # Normalize team names across seasons
        from data_pipeline import normalize_team_name
        result['Team'] = result['Team'].apply(normalize_team_name)
        return result

    except Exception as e:
        print(f"Error extracting pit stops for {gp_name} {year}: {e}")
        return pd.DataFrame()


def compute_team_pit_variability(years: list, gps: list) -> dict:
    """Compute Fisk distribution parameters for pit stop variability per team and track.
    
    For each team, we:
    1. Collect all pit stop durations across the specified races/tracks
    2. Compute the team's median pit duration per track: median_duration[Team][Track]
    3. Calculate the deviation: deviation = PitDuration - median_duration[Team][Track]
    4. Fit a Fisk distribution to these deviations pooled across all tracks
    
    Returns:
        dict: {
            team_name: {
                'c': float, 'loc': float, 'scale': float,
                'medians': {track: float},
                'global_median': float
            }
        }
    """
    all_pit_stops = []

    for year in years:
        for gp in gps:
            df = extract_pit_stops(year, gp)
            if not df.empty:
                df['Track'] = gp
                all_pit_stops.append(df)

    if not all_pit_stops:
        print("No pit stop data found!")
        return {}

    combined = pd.concat(all_pit_stops, ignore_index=True)

    team_fisk_params = {}
    for team, team_data in combined.groupby('Team'):
        # 1. Compute track-specific medians
        track_medians = {}
        for track, track_data in team_data.groupby('Track'):
            track_medians[track] = float(np.median(track_data['PitDuration'].values))
            
        global_median = float(np.median(team_data['PitDuration'].values))
        
        # 2. Compute deviations using track-specific baselines
        deviations = []
        for _, row in team_data.iterrows():
            track = row['Track']
            duration = row['PitDuration']
            med = track_medians.get(track, global_median)
            deviations.append(duration - med)
            
        deviations = np.array(deviations)
        if len(deviations) < 5:
            continue

        try:
            c, loc, scale = fisk.fit(deviations)
            team_fisk_params[team] = {
                'c': float(c),
                'loc': float(loc),
                'scale': float(scale),
                'medians': track_medians,
                'global_median': global_median
            }
        except Exception as e:
            pass

    return team_fisk_params


if __name__ == "__main__":
    from data_pipeline import SEASONS

    test_gps = ['Bahrain']
    params = compute_team_pit_variability(SEASONS, test_gps)

    print("\n--- Pit Stop Fisk Parameters per Team ---")
    for team, p in sorted(params.items()):
        print(f"  {team}: median={p['median_duration']:.2f}s, "
              f"c={p['c']:.3f}, loc={p['loc']:.3f}, scale={p['scale']:.3f}")
