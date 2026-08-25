"""
Section 3.4 - Determination of Full Course Yellow (FCY) Phases
===============================================================
Heilmeier et al. (2020), Section 3.4:

Safety Car (SC) and Virtual Safety Car (VSC) phases are pre-determined
BEFORE the main race loop using historical distributions:

    p_sc_quant:     Probability of 0, 1, 2, 3 SC phases per race
    p_sc_start:     Probability of SC start in each 20% race progress bin
    p_sc_duration:  Discrete duration distribution (1-10 laps)
    p_vsc_after_failure: Probability of VSC deployment after a mechanical failure
"""

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('../cache')


def extract_sc_phases(year: int, gp_name: str) -> dict:
    """Extract Safety Car and VSC phase data from a race session.
    
    Returns:
        dict with keys:
            'total_laps': int
            'sc_phases': list of {'start_lap': int, 'end_lap': int, 'duration': int}
            'vsc_phases': list of {'start_lap': int, 'end_lap': int, 'duration': int}
    """
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry=False, weather=False)

        total_laps = session.total_laps if hasattr(session, 'total_laps') else 57
        
        # Get race control messages
        rcm = session.race_control_messages
        if rcm is None or rcm.empty:
            return {'total_laps': total_laps, 'sc_phases': [], 'vsc_phases': []}

        sc_phases = []
        vsc_phases = []
        current_sc_start = None
        current_vsc_start = None

        for _, msg in rcm.iterrows():
            category = str(msg.get('Category', '')).upper()
            message = str(msg.get('Message', '')).upper()
            lap = msg.get('Lap', None)
            
            if lap is None or pd.isna(lap):
                continue
            lap = int(lap)

            # Detect SC deployment
            if 'SAFETY CAR' in message and 'VIRTUAL' not in message:
                if 'DEPLOYED' in message or 'IN THIS LAP' in message:
                    if current_sc_start is None:
                        current_sc_start = lap
                elif 'IN THIS LAP' in message and 'ENDING' in message.upper():
                    if current_sc_start is not None:
                        duration = max(1, lap - current_sc_start)
                        sc_phases.append({
                            'start_lap': current_sc_start,
                            'end_lap': lap,
                            'duration': duration
                        })
                        current_sc_start = None

            # Detect SC ending via GREEN flag
            if ('GREEN' in message or 'RESTART' in message) and current_sc_start is not None:
                duration = max(1, lap - current_sc_start)
                sc_phases.append({
                    'start_lap': current_sc_start,
                    'end_lap': lap,
                    'duration': duration
                })
                current_sc_start = None

            # Detect VSC
            if 'VIRTUAL SAFETY CAR' in message:
                if 'DEPLOYED' in message:
                    if current_vsc_start is None:
                        current_vsc_start = lap
                elif 'ENDING' in message or 'END' in message:
                    if current_vsc_start is not None:
                        duration = max(1, lap - current_vsc_start)
                        vsc_phases.append({
                            'start_lap': current_vsc_start,
                            'end_lap': lap,
                            'duration': duration
                        })
                        current_vsc_start = None

        # Close any unclosed phases
        if current_sc_start is not None:
            sc_phases.append({
                'start_lap': current_sc_start,
                'end_lap': total_laps,
                'duration': total_laps - current_sc_start
            })
        if current_vsc_start is not None:
            vsc_phases.append({
                'start_lap': current_vsc_start,
                'end_lap': total_laps,
                'duration': total_laps - current_vsc_start
            })

        return {
            'total_laps': total_laps,
            'sc_phases': sc_phases,
            'vsc_phases': vsc_phases
        }

    except Exception as e:
        print(f"Error extracting SC phases for {gp_name} {year}: {e}")
        return {'total_laps': 57, 'sc_phases': [], 'vsc_phases': []}


def compute_fcy_probabilities(years: list, gps: list) -> dict:
    """Compute FCY phase probability distributions from historical data.
    
    Returns:
        dict with keys:
            'p_sc_quant': list of 4 floats (probability of 0, 1, 2, 3+ SC phases)
            'p_sc_start': list of 6 floats (probability of SC start in each progress bin)
            'p_sc_duration': list of 10 floats (probability of SC duration 1-10 laps)
            'p_vsc_quant': list of 4 floats (probability of 0, 1, 2, 3+ VSC phases)
            'avg_total_laps': float
    """
    sc_counts = []        # Number of SC phases per race
    vsc_counts = []       # Number of VSC phases per race
    sc_start_bins = []    # Which 20% bin each SC started in
    sc_durations = []     # Duration of each SC phase
    total_laps_list = []

    for year in years:
        for gp in gps:
            data = extract_sc_phases(year, gp)
            total_laps = data['total_laps']
            total_laps_list.append(total_laps)

            sc_count = len(data['sc_phases'])
            vsc_count = len(data['vsc_phases'])
            sc_counts.append(min(sc_count, 3))  # Cap at 3+
            vsc_counts.append(min(vsc_count, 3))

            for phase in data['sc_phases']:
                # Determine which 20% bin the SC started in
                progress = phase['start_lap'] / total_laps
                if phase['start_lap'] == 1:
                    sc_start_bins.append(0)  # First lap
                elif progress <= 0.2:
                    sc_start_bins.append(1)
                elif progress <= 0.4:
                    sc_start_bins.append(2)
                elif progress <= 0.6:
                    sc_start_bins.append(3)
                elif progress <= 0.8:
                    sc_start_bins.append(4)
                else:
                    sc_start_bins.append(5)

                # Duration capped at 10 laps
                sc_durations.append(min(phase['duration'], 10))

    n_races = len(sc_counts)
    if n_races == 0:
        print("No race data found for FCY analysis!")
        return {}

    # p_sc_quant: probability of 0, 1, 2, 3+ SC phases
    p_sc_quant = [0.0] * 4
    for count in sc_counts:
        p_sc_quant[count] += 1
    p_sc_quant = [x / n_races for x in p_sc_quant]

    # p_sc_start: probability of SC start in each 20% bin
    # [first_lap, <20%, <40%, <60%, <80%, <100%]
    p_sc_start = [0.0] * 6
    if sc_start_bins:
        for b in sc_start_bins:
            p_sc_start[b] += 1
        total_sc = sum(p_sc_start)
        if total_sc > 0:
            p_sc_start = [x / total_sc for x in p_sc_start]

    # p_sc_duration: probability of each duration (1-10 laps)
    p_sc_duration = [0.0] * 10
    if sc_durations:
        for d in sc_durations:
            p_sc_duration[d - 1] += 1
        total_d = sum(p_sc_duration)
        if total_d > 0:
            p_sc_duration = [x / total_d for x in p_sc_duration]

    # p_vsc_quant
    p_vsc_quant = [0.0] * 4
    for count in vsc_counts:
        p_vsc_quant[count] += 1
    p_vsc_quant = [x / n_races for x in p_vsc_quant]

    avg_total_laps = np.mean(total_laps_list) if total_laps_list else 57

    return {
        'p_sc_quant': p_sc_quant,
        'p_sc_start': p_sc_start,
        'p_sc_duration': p_sc_duration,
        'p_vsc_quant': p_vsc_quant,
        'avg_total_laps': avg_total_laps,
        'n_races_analyzed': n_races
    }


def pre_determine_sc_phases(fcy_params: dict, total_laps: int, rng: np.random.Generator = None) -> list:
    """Pre-determine SC phases for a single Monte Carlo iteration.
    
    Returns a list of dicts: [{'start_lap': int, 'duration': int}, ...]
    """
    if rng is None:
        rng = np.random.default_rng()

    p_sc_quant = fcy_params['p_sc_quant']
    p_sc_start = fcy_params['p_sc_start']
    p_sc_duration = fcy_params['p_sc_duration']

    # Determine number of SC phases
    n_sc = rng.choice([0, 1, 2, 3], p=p_sc_quant)
    
    if n_sc == 0:
        return []

    # Fallback: if start or duration distributions are all zeros (no historical data),
    # use uniform distributions instead of crashing
    if sum(p_sc_start) == 0:
        p_sc_start = [1/6] * 6
    if sum(p_sc_duration) == 0:
        p_sc_duration = [1/10] * 10

    sc_phases = []
    for _ in range(n_sc):
        # Determine which bin the SC starts in
        bin_idx = rng.choice(6, p=p_sc_start)

        # Convert bin to a specific lap
        if bin_idx == 0:
            start_lap = 1
        else:
            # Map bin to lap range
            bin_start = (bin_idx - 1) * 0.2 if bin_idx > 0 else 0
            bin_end = bin_idx * 0.2 if bin_idx < 5 else 1.0
            lap_start = max(2, int(bin_start * total_laps))
            lap_end = int(bin_end * total_laps)
            start_lap = rng.integers(lap_start, max(lap_start + 1, lap_end))

        # Determine duration
        duration = rng.choice(range(1, 11), p=p_sc_duration)

        sc_phases.append({
            'start_lap': int(start_lap),
            'duration': int(duration)
        })

    # Sort by start lap
    sc_phases.sort(key=lambda x: x['start_lap'])

    # Merge overlapping SC phases
    merged = [sc_phases[0]]
    for phase in sc_phases[1:]:
        prev = merged[-1]
        prev_end = prev['start_lap'] + prev['duration']
        if phase['start_lap'] <= prev_end:
            # Overlap detected — extend the previous phase
            new_end = max(prev_end, phase['start_lap'] + phase['duration'])
            prev['duration'] = new_end - prev['start_lap']
        else:
            merged.append(phase)

    return merged


if __name__ == "__main__":
    from data_pipeline import SEASONS

    test_gps = ['Bahrain']
    params = compute_fcy_probabilities(SEASONS, test_gps)

    if params:
        print(f"\nAnalyzed {params['n_races_analyzed']} races")
        print(f"Average total laps: {params['avg_total_laps']:.0f}")

        print("\n--- SC Quantity Probabilities ---")
        for i, p in enumerate(params['p_sc_quant']):
            print(f"  P({i} SC phases) = {p:.3f}")

        print("\n--- SC Start Bin Probabilities ---")
        labels = ['First Lap', '<20%', '<40%', '<60%', '<80%', '<100%']
        for label, p in zip(labels, params['p_sc_start']):
            print(f"  P(start in {label}) = {p:.3f}")

        print("\n--- SC Duration Probabilities ---")
        for i, p in enumerate(params['p_sc_duration']):
            print(f"  P(duration={i+1} laps) = {p:.3f}")

        print("\n--- VSC Quantity Probabilities ---")
        for i, p in enumerate(params['p_vsc_quant']):
            print(f"  P({i} VSC phases) = {p:.3f}")

        # Test pre-determination
        print("\n--- Sample Pre-determined SC Phases ---")
        for i in range(5):
            phases = pre_determine_sc_phases(params, 57)
            if phases:
                for ph in phases:
                    print(f"  Run {i+1}: SC starts lap {ph['start_lap']}, duration {ph['duration']} laps")
            else:
                print(f"  Run {i+1}: No SC")
