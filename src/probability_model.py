"""
Section 3.3 - Determination of Accident and Failure Probabilities
==================================================================
Heilmeier et al. (2020), Section 3.3:

Accidents are attributed to the DRIVER, technical failures to the TEAM.
Uses a Bayesian Beta-Bernoulli model:

    Prior:    Beta(alpha_hat, beta_hat)
    Posterior: Beta(alpha_hat + z, beta_hat + N - z)
    
    where z = number of events, N = total races

    alpha_hat = ((1 - mu_hat) / sigma_hat^2 - 1/mu_hat) * mu_hat^2
    beta_hat  = alpha_hat * (1/mu_hat - 1)

The mean of the posterior is used as the final probability:
    P_event = (alpha_hat + z) / (alpha_hat + beta_hat + N)
"""

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('../cache')

# Keywords to classify retirement status from FastF1
ACCIDENT_KEYWORDS = [
    'accident', 'collision', 'spun off', 'crash', 'hit',
    'contact', 'barrier', 'off track', 'damage'
]

FAILURE_KEYWORDS = [
    'engine', 'gearbox', 'hydraulics', 'power unit', 'electrical',
    'brakes', 'overheating', 'oil pressure', 'water pressure',
    'fuel system', 'exhaust', 'transmission', 'suspension',
    'mechanical', 'technical', 'pneumatic', 'water leak',
    'oil leak', 'fuel pressure', 'turbo', 'ers', 'mgu'
]

FINISHED_KEYWORDS = [
    'finished', 'lapped', '+1 lap', '+2 laps', '+3 laps', '+4 laps',
    '+5 laps', '+6 laps'
]


def classify_retirement(status: str) -> str:
    """Classify a FastF1 result status into 'accident', 'failure', or 'finished'."""
    if pd.isna(status):
        return 'finished'

    status_lower = status.lower().strip()

    # Check if the driver finished the race
    for kw in FINISHED_KEYWORDS:
        if kw in status_lower:
            return 'finished'

    # Check for accident (driver-attributed)
    for kw in ACCIDENT_KEYWORDS:
        if kw in status_lower:
            return 'accident'

    # Check for technical failure (team-attributed)
    for kw in FAILURE_KEYWORDS:
        if kw in status_lower:
            return 'failure'

    # If status doesn't match known patterns, treat as failure (conservative)
    # This catches things like "Retired", "Did not finish", etc.
    return 'failure'


def fetch_race_results(year: int, gp_name: str) -> pd.DataFrame:
    """Fetch race results with classification of retirement type.
    
    Returns DataFrame with columns: Driver, Team, Status, Classification
    """
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False)

        results = session.results[['Abbreviation', 'TeamName', 'Status']].copy()
        results = results.rename(columns={'Abbreviation': 'Driver', 'TeamName': 'Team'})
        # Normalize team names across seasons
        from data_pipeline import normalize_team_name
        results['Team'] = results['Team'].apply(normalize_team_name)
        results['Classification'] = results['Status'].apply(classify_retirement)
        results['Year'] = year
        results['GP'] = gp_name

        return results

    except Exception as e:
        print(f"Error fetching results for {gp_name} {year}: {e}")
        return pd.DataFrame()


def build_historical_database(years: list, gps: list) -> dict:
    """Build a historical database of accidents and failures grouped by Track Type.
    
    Returns:
        dict with keys:
            'driver_accidents': {driver: {track_type: {'races': int, 'accidents': int}}}
            'team_failures': {team: {track_type: {'races': int, 'failures': int}}}
            'all_results': pd.DataFrame
    """
    all_results = []
    from data_pipeline import get_track_type

    for year in years:
        for gp in gps:
            df = fetch_race_results(year, gp)
            if not df.empty:
                df['TrackType'] = df['GP'].apply(get_track_type)
                all_results.append(df)

    if not all_results:
        print("No race results found!")
        return {}

    combined = pd.concat(all_results, ignore_index=True)

    track_types = ['street', 'permanent', 'high-speed', 'All']

    # Per-driver accident stats per track type
    driver_accidents = {}
    for driver, driver_data in combined.groupby('Driver'):
        driver_accidents[driver] = {tt: {'races': 0, 'accidents': 0} for tt in track_types}
        
        # Global stats
        driver_accidents[driver]['All'] = {
            'races': len(driver_data),
            'accidents': len(driver_data[driver_data['Classification'] == 'accident'])
        }
        
        # Track-type split stats
        for tt in ['street', 'permanent', 'high-speed']:
            sub = driver_data[driver_data['TrackType'] == tt]
            driver_accidents[driver][tt] = {
                'races': len(sub),
                'accidents': len(sub[sub['Classification'] == 'accident'])
            }

    # Per-team failure stats per track type
    team_failures = {}
    for team, team_data in combined.groupby('Team'):
        team_failures[team] = {tt: {'races': 0, 'failures': 0} for tt in track_types}
        
        # Global stats
        team_failures[team]['All'] = {
            'races': len(team_data),
            'failures': len(team_data[team_data['Classification'] == 'failure'])
        }
        
        # Track-type split stats
        for tt in ['street', 'permanent', 'high-speed']:
            sub = team_data[team_data['TrackType'] == tt]
            team_failures[team][tt] = {
                'races': len(sub),
                'failures': len(sub[sub['Classification'] == 'failure'])
            }

    return {
        'driver_accidents': driver_accidents,
        'team_failures': team_failures,
        'all_results': combined
    }


def compute_beta_prior_for_type(entity_stats: dict, event_key: str, track_type: str) -> tuple:
    """Compute the Beta prior parameters (alpha_hat, beta_hat) for a specific track type."""
    fractions = []
    for entity, stats_dict in entity_stats.items():
        stats = stats_dict[track_type]
        if stats['races'] >= 3:
            fraction = stats[event_key] / stats['races']
            fractions.append(fraction)

    if len(fractions) < 2:
        return (1.0, 10.0)  # Safe default prior (approx 9% DNF probability)

    fractions = np.array(fractions)
    fractions = np.maximum(fractions, 0.001)

    mu_hat = np.mean(fractions)
    sigma_hat = np.std(fractions)

    if sigma_hat == 0 or mu_hat == 0:
        return (1.0, 10.0)

    # Beta distribution method of moments (Paper's formula)
    alpha_hat = ((1 - mu_hat) / sigma_hat**2 - 1/mu_hat) * mu_hat**2
    beta_hat = alpha_hat * (1/mu_hat - 1)

    # Prevent prior dominance: cap the prior weight (equivalent sample size) at 10.0.
    # This ensures driver/team data (typically N=10 to 24 races) can significantly update the posterior.
    prior_weight = alpha_hat + beta_hat
    max_prior_weight = 10.0
    if prior_weight > max_prior_weight:
        scale = max_prior_weight / prior_weight
        alpha_hat *= scale
        beta_hat *= scale

    return (max(alpha_hat, 0.1), max(beta_hat, 0.1))


def compute_probabilities(years: list, gps: list) -> dict:
    """Compute track-type split accident and failure probabilities using Bayesian Beta-Bernoulli update.
    
    Returns:
        dict: {
            'p_accident': {driver: {track_type: float}},
            'p_failure': {team: {track_type: float}}
        }
    """
    db = build_historical_database(years, gps)
    if not db:
        return {}

    driver_accidents = db['driver_accidents']
    team_failures = db['team_failures']
    track_types = ['street', 'permanent', 'high-speed', 'All']

    # Compute priors for all types
    priors_accident = {}
    priors_failure = {}
    for tt in track_types:
        priors_accident[tt] = compute_beta_prior_for_type(driver_accidents, 'accidents', tt)
        priors_failure[tt] = compute_beta_prior_for_type(team_failures, 'failures', tt)

    # Compute posteriors
    p_accident = {}
    for driver, stats_dict in driver_accidents.items():
        p_accident[driver] = {}
        # Pre-calculate global fallback posterior
        alpha_g, beta_g = priors_accident['All']
        z_g = stats_dict['All']['accidents']
        N_g = stats_dict['All']['races']
        global_posterior = (alpha_g + z_g) / (alpha_g + beta_g + N_g)

        for tt in ['street', 'permanent', 'high-speed']:
            stats = stats_dict[tt]
            alpha, beta = priors_accident[tt]
            z = stats['accidents']
            N = stats['races']
            
            # If driver has less than 3 races on this track type, fall back to global
            if N >= 3:
                p_accident[driver][tt] = (alpha + z) / (alpha + beta + N)
            else:
                p_accident[driver][tt] = global_posterior
        # Store global fallback
        p_accident[driver]['All'] = global_posterior

    p_failure = {}
    for team, stats_dict in team_failures.items():
        p_failure[team] = {}
        # Pre-calculate global fallback posterior
        alpha_g, beta_g = priors_failure['All']
        z_g = stats_dict['All']['failures']
        N_g = stats_dict['All']['races']
        global_posterior = (alpha_g + z_g) / (alpha_g + beta_g + N_g)

        for tt in ['street', 'permanent', 'high-speed']:
            stats = stats_dict[tt]
            alpha, beta = priors_failure[tt]
            z = stats['failures']
            N = stats['races']
            
            if N >= 3:
                p_failure[team][tt] = (alpha + z) / (alpha + beta + N)
            else:
                p_failure[team][tt] = global_posterior
        p_failure[team]['All'] = global_posterior

    return {
        'p_accident': p_accident,
        'p_failure': p_failure,
        'database': db
    }


if __name__ == "__main__":
    from data_pipeline import SEASONS

    test_gps = ['Bahrain']
    result = compute_probabilities(SEASONS, test_gps)

    if result:
        print("\n--- Accident Probabilities per Driver (per race) ---")
        alpha_a, beta_a = result['alpha_beta_accident']
        print(f"  Prior: alpha={alpha_a:.3f}, beta={beta_a:.3f}")
        for driver, p in sorted(result['p_accident'].items(), key=lambda x: -x[1]):
            print(f"  {driver}: P_accident = {p:.4f}")

        print("\n--- Failure Probabilities per Team (per race) ---")
        alpha_f, beta_f = result['alpha_beta_failure']
        print(f"  Prior: alpha={alpha_f:.3f}, beta={beta_f:.3f}")
        for team, p in sorted(result['p_failure'].items(), key=lambda x: -x[1]):
            print(f"  {team}: P_failure = {p:.4f}")

        # Show the raw classification data
        print("\n--- Retirement Classifications Sample ---")
        all_results = result['database']['all_results']
        retirements = all_results[all_results['Classification'] != 'finished']
        if not retirements.empty:
            print(retirements[['Year', 'GP', 'Driver', 'Team', 'Status', 'Classification']].to_string(index=False))
