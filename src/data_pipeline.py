import fastf1
import pandas as pd
import numpy as np
from scipy.stats import norm, fisk
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('../cache')

SEASONS = [2023, 2024, 2025]

def fetch_and_clean_laps(year: int, gp_name: str) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, gp_name, 'R')
        session.load(telemetry = False, weather=False)
        laps = session.laps.copy()

        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

        dry_laps = laps[~laps['Compound'].isin(['INTERMEDIATE', 'WET'])].copy()

        dry_laps = dry_laps.dropna(subset = ['LapTimeSeconds'])

        clean_laps = dry_laps[dry_laps['TrackStatus'] == '1'].copy()

        clean_laps = clean_laps[clean_laps['PitInTime'].isna() & clean_laps['PitOutTime'].isna()]

        clean_laps = clean_laps[clean_laps['LapNumber'] > 2].copy()

        clean_laps['Year'] = year
        clean_laps['Track'] = gp_name
        
        return clean_laps

    except Exception as e:
        print(f"Error fetching laps for {gp_name} {year}: {e}")
        return pd.DataFrame()

def aggregate_seasons_data(years: list, gps:list) -> pd.DataFrame:
    all_laps = []

    for year in years:
        for gp in gps:
            print(f"Aggregating data for {year} {gp}...")
            df = fetch_and_clean_laps(year, gp)
            if not df.empty:
                all_laps.append(df)

    if all_laps:
        return pd.concat(all_laps, ignore_index=True)
    else:
        return pd.DataFrame()

def driver_stints(df: pd.DataFrame):
    driver_residuals = {}

    stint_groups = df.groupby(['Year', 'Track', 'Driver', 'Compound', 'Stint'])

    for stint_keys, stint_data in stint_groups:
        year, track, driver, compound, stint_number = stint_keys
        
        if len(stint_data) < 5:
            continue

        x = stint_data['TyreLife'].values
        y = stint_data['LapTimeSeconds'].values

        k2, k1, k0 = np.polyfit(x, y, 2)

        y_pred = k2*(x**2) + k1*x + k0

        residuals = y - y_pred

        if driver not in driver_residuals:
            driver_residuals[driver] = []
        driver_residuals[driver].extend(residuals)

    driver_fisk_params = {}
    for driver, residuals_list in driver_residuals.items():
        residuals_arr = np.array(residuals_list)
        if len(residuals_arr) < 10:
            continue
        # Fit the Fisk (log-logistic) distribution to the residuals
        c, loc, scale = fisk.fit(residuals_arr)
        driver_fisk_params[driver] = {'c': c, 'loc': loc, 'scale': scale}
        
    return driver_fisk_params, driver_residuals

def starting_performance(years, gps):
    grid_positions = []
    excess_times = []
    driver_residuals = {}

    for year in years:
        for gp in gps:
            try:

                clean_df = fetch_and_clean_laps(year, gp)
                if clean_df.empty:
                    continue

                driver_medians = clean_df.groupby('Driver')['LapTimeSeconds'].median().to_dict()
                
                session = fastf1.get_session(year, gp, 'R')
                session.load(telemetry = False, weather = False)

                lap1_df = session.laps[session.laps['LapNumber'] == 1].copy()
                lap1_df['LapTimeSeconds'] = lap1_df['LapTime'].dt.total_seconds()

                results_df = session.results[['Abbreviation', 'GridPosition']].copy()
                results_df = results_df.rename(columns={'Abbreviation': 'Driver'})

                merged = pd.merge(lap1_df[['Driver', 'LapTimeSeconds']], results_df, on='Driver')
                merged = merged.dropna(subset=['LapTimeSeconds', 'GridPosition'])

                merged = merged[merged['GridPosition'] > 0]
                for _, row in merged.iterrows():
                    driver = row['Driver']
                    grid_pos = row['GridPosition']
                    lap1_time = row['LapTimeSeconds']
                    
                    if driver in driver_medians:
                        lap1_excess = lap1_time - driver_medians[driver]
                        
                        if 0 < lap1_excess < 25:
                            grid_positions.append(grid_pos)
                            excess_times.append(lap1_excess)
                            
                            if driver not in driver_residuals:
                                driver_residuals[driver] = []
                            driver_residuals[driver].append((grid_pos, lap1_excess))

            except Exception as e:
                print(f"Error processing {year} {gp}: {e}")
                continue

    # 7. Check if we actually collected any starting data
    if not grid_positions:
        print("No starting performance data found!")
        return None, None

    # Convert to numpy arrays for calculation
    x_data = np.array(grid_positions)
    y_data = np.array(excess_times)

    # 8. Define the paper's physical reference curve (Equation 10)
    # distance = 8m * (grid_position - 0.8)
    # expected_time = sqrt(2 * distance / a_avg) + human_reaction_time (0.2s)
    def ref_curve(p_g, a_avg):
        return np.sqrt(16.0 * (p_g - 0.8) / a_avg) + 0.2

    # Fit the average acceleration parameter (a_avg) using least squares
    popt, pcov = curve_fit(ref_curve, x_data, y_data, p0=[11.2])
    a_avg_fitted = popt[0]
    print(f"\nFitted Start Line Acceleration (a_avg): {a_avg_fitted:.3f} m/s^2")

    # 9. Compute driver-specific start residuals and fit Normal(mu, sigma)
    driver_stats = {}
    for driver, data_list in driver_residuals.items():
        residuals = []
        for grid_pos, excess in data_list:
            expected = ref_curve(grid_pos, a_avg_fitted)
            residual = excess - expected
            residuals.append(residual)
            
        # We need at least 2 data points for standard deviation to be meaningful
        if len(residuals) >= 2:
            mu = np.mean(residuals)
            sigma = np.std(residuals)
            driver_stats[driver] = {'mu': mu, 'sigma': sigma}

    # 10. Generate and save the comparison plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_data, y_data, alpha=0.5, color='blue', label='Actual Start Delay')
    
    # Generate points to draw the reference curve line
    x_plot = np.linspace(1, 20, 100)
    plt.plot(x_plot, ref_curve(x_plot, a_avg_fitted), 'r-', lw=2, 
             label=f'Fitted Ref Curve (a_avg={a_avg_fitted:.2f} m/s^2)')
    
    plt.title('Starting Performance: Time to Start Line vs Grid Position')
    plt.xlabel('Grid Position')
    plt.ylabel('Start Delay (Lap 1 Excess Time in seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = 'starting_performance_ref_curve.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved reference curve plot to: {plot_path}")

    return a_avg_fitted, driver_stats


def compare_distributions(residuals, driver_name: str):
    residuals = np.array(residuals)
    
    # 1. Fit Normal Distribution
    mu_norm, std_norm = norm.fit(residuals)
    
    # 2. Fit Fisk (Log-Logistic) Distribution
    # Note: fisk.fit returns (c, loc, scale)
    c_fisk, loc_fisk, scale_fisk = fisk.fit(residuals)
    
    # 3. Calculate Log-Likelihoods
    log_lik_norm = np.sum(norm.logpdf(residuals, mu_norm, std_norm))
    log_lik_fisk = np.sum(fisk.logpdf(residuals, c_fisk, loc_fisk, scale_fisk))
    
    print(f"\n--- Distribution Comparison for {driver_name} ---")
    print(f"Number of data points (laps): {len(residuals)}")
    print(f"Normal Fit: mu={mu_norm:.3f}, std={std_norm:.3f}")
    print(f"Normal Log-Likelihood: {log_lik_norm:.2f}")
    print(f"Fisk Fit: c={c_fisk:.3f}, loc={loc_fisk:.3f}, scale={scale_fisk:.3f}")
    print(f"Fisk Log-Likelihood: {log_lik_fisk:.2f}")
    
    if log_lik_fisk > log_lik_norm:
         print("Result: Fisk distribution fits the residuals better (higher log-likelihood)!")
    else:
         print("Result: Normal distribution fits the residuals better!")

    # 4. Plotting the comparison
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of actual residuals
    count, bins, ignored = plt.hist(residuals, bins=30, density=True, alpha=0.6, color='gray', label='Actual Residuals')
    
    # Generate points for plotting the curves
    x = np.linspace(min(residuals) - 1, max(residuals) + 1, 1000)
    
    # Plot Normal PDF
    plt.plot(x, norm.pdf(x, mu_norm, std_norm), 'r-', lw=2, label=f'Normal Fit (LL={log_lik_norm:.1f})')
    
    # Plot Fisk PDF
    plt.plot(x, fisk.pdf(x, c_fisk, loc_fisk, scale_fisk), 'b-', lw=2, label=f'Fisk Fit (LL={log_lik_fisk:.1f})')
    
    plt.title(f'Lap Time Residuals Distribution Comparison - {driver_name}')
    plt.xlabel('Lap Time Residual (seconds)')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plot_path = 'residuals_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved comparison plot to: {plot_path}")


        

if __name__ == "__main__":
    test_gps = ['Bahrain']
    test_years = SEASONS

    # Section 3.1 - Starting Performance
    a_avg_fitted, driver_start_stats = starting_performance(test_years, test_gps)
    if driver_start_stats:
        print(f"\nFitted Acceleration (a_avg): {a_avg_fitted:.3f} m/s^2")
        print("\n--- Driver Starting Performance (mu, sigma) ---")
        for driver, stats in sorted(driver_start_stats.items()):
            print(f"  {driver}: mu={stats['mu']:.3f}s, sigma={stats['sigma']:.3f}s")

    # Section 3.2a - Lap Time Variability
    master_df = aggregate_seasons_data(test_years, test_gps)

    if not master_df.empty:
        fisk_params, residuals = driver_stints(master_df)
        
        print("\n--- Fisk (Log-Logistic) Parameters per Driver ---")
        for driver, params in sorted(fisk_params.items()):
            print(f"  {driver}: c={params['c']:.3f}, loc={params['loc']:.3f}, scale={params['scale']:.3f}")
        
        target_driver = 'VER'
        if target_driver in residuals:
            compare_distributions(residuals[target_driver], target_driver)


