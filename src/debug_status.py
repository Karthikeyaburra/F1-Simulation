import fastf1
import pandas as pd
from probability_model import classify_retirement

fastf1.Cache.enable_cache('../cache')

session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load(telemetry=False, weather=False)

print("=== Raw Status Values for Bahrain 2024 ===")
for idx, row in session.results.iterrows():
    status = row['Status']
    classification = classify_retirement(status)
    print(f"Driver: {row['Abbreviation']}, Team: {row['TeamName']}, Status: {status!r} -> Classified as: {classification}")
