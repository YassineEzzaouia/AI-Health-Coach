"""
Comprehensive Health Data EDA for AI Health Coach
Focus: Health metrics analysis for RL and LLM applications
Author: Collaborative EDA for FitBit & Nutrition5k
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("AI HEALTH COACH - COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
print("="*80)
print("\n")

# ============================================================================
# PART 1: LOAD AND EXPLORE ALL FITBIT DATASETS
# ============================================================================
print("PART 1: LOADING FITBIT DATASETS")
print("-"*80)

# Daily Level Data
daily_activity = pd.read_csv('FitBit Dataset/dailyActivity_merged.csv')
daily_calories = pd.read_csv('FitBit Dataset/dailyCalories_merged.csv')
daily_intensities = pd.read_csv('FitBit Dataset/dailyIntensities_merged.csv')
daily_steps = pd.read_csv('FitBit Dataset/dailySteps_merged.csv')
sleep_day = pd.read_csv('FitBit Dataset/sleepDay_merged.csv')
weight_log = pd.read_csv('FitBit Dataset/weightLogInfo_merged.csv')

# Hourly Level Data
hourly_calories = pd.read_csv('FitBit Dataset/hourlyCalories_merged.csv')
hourly_intensities = pd.read_csv('FitBit Dataset/hourlyIntensities_merged.csv')
hourly_steps = pd.read_csv('FitBit Dataset/hourlySteps_merged.csv')

# Minute Level Data
minute_calories = pd.read_csv('FitBit Dataset/minuteCaloriesNarrow_merged.csv')
minute_intensities = pd.read_csv('FitBit Dataset/minuteIntensitiesNarrow_merged.csv')
minute_steps = pd.read_csv('FitBit Dataset/minuteStepsNarrow_merged.csv')
minute_METs = pd.read_csv('FitBit Dataset/minuteMETsNarrow_merged.csv')
minute_sleep = pd.read_csv('FitBit Dataset/minuteSleep_merged.csv')

# Heart Rate Data (Second Level)
heartrate_seconds = pd.read_csv('FitBit Dataset/heartrate_seconds_merged.csv')

print(f"✓ Daily Activity: {daily_activity.shape}")
print(f"✓ Sleep Data: {sleep_day.shape}")
print(f"✓ Weight Log: {weight_log.shape}")
print(f"✓ Heart Rate (seconds): {heartrate_seconds.shape}")
print(f"✓ Minute-level data loaded (calories, intensities, steps, METs, sleep)")
print(f"✓ Hourly data loaded (calories, intensities, steps)")

print("\n" + "="*80)
print("PART 2: DATA STRUCTURE ANALYSIS")
print("-"*80)

# Daily Activity Detailed Analysis
print("\n📊 DAILY ACTIVITY DATASET")
print(f"Shape: {daily_activity.shape}")
print(f"\nColumns ({len(daily_activity.columns)}):")
print(daily_activity.columns.tolist())
print(f"\nData Types:")
print(daily_activity.dtypes)
print(f"\nUnique Users: {daily_activity['Id'].nunique()}")
print(f"\nDate Range: {daily_activity['ActivityDate'].min()} to {daily_activity['ActivityDate'].max()}")

print("\n📊 BASIC STATISTICS - DAILY ACTIVITY")
print(daily_activity.describe())

print("\n📊 SLEEP DATA ANALYSIS")
print(f"Shape: {sleep_day.shape}")
print(f"Columns: {sleep_day.columns.tolist()}")
print(f"Unique Users with Sleep Data: {sleep_day['Id'].nunique()}")
print("\nSleep Statistics:")
print(sleep_day.describe())

print("\n📊 HEART RATE DATA (SECONDS)")
print(f"Shape: {heartrate_seconds.shape}")
print(f"Columns: {heartrate_seconds.columns.tolist()}")
print(f"Unique Users: {heartrate_seconds['Id'].nunique()}")
print("\nHeart Rate Statistics:")
print(heartrate_seconds['Value'].describe())

# ============================================================================
# PART 3: DATA QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("PART 3: DATA QUALITY ASSESSMENT")
print("-"*80)

def analyze_data_quality(df, name):
    """Comprehensive data quality analysis"""
    print(f"\n🔍 {name}")
    print(f"  Total Records: {len(df):,}")
    print(f"  Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            pct = (count / len(df)) * 100
            print(f"    - {col}: {count} ({pct:.2f}%)")
    else:
        print("    ✓ No missing values")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    print(f"  Duplicates: {duplicates}")
    
    return missing.sum()

# Analyze all datasets
total_missing = 0
total_missing += analyze_data_quality(daily_activity, "Daily Activity")
total_missing += analyze_data_quality(sleep_day, "Sleep Data")
total_missing += analyze_data_quality(weight_log, "Weight Log")
total_missing += analyze_data_quality(heartrate_seconds, "Heart Rate (sample of 1000)")
total_missing += analyze_data_quality(hourly_steps, "Hourly Steps")
total_missing += analyze_data_quality(minute_steps, "Minute Steps (sample)")

print(f"\n📌 Total Missing Values Across All Datasets: {total_missing}")

# ============================================================================
# PART 4: DATE PARSING AND PREPARATION
# ============================================================================
print("\n" + "="*80)
print("PART 4: DATE PARSING AND TEMPORAL PREPARATION")
print("-"*80)

# Parse dates
daily_activity['ActivityDate'] = pd.to_datetime(daily_activity['ActivityDate'])
sleep_day['SleepDay'] = pd.to_datetime(sleep_day['SleepDay'])
weight_log['Date'] = pd.to_datetime(weight_log['Date'])
hourly_calories['ActivityHour'] = pd.to_datetime(hourly_calories['ActivityHour'])
hourly_steps['ActivityHour'] = pd.to_datetime(hourly_steps['ActivityHour'])
hourly_intensities['ActivityHour'] = pd.to_datetime(hourly_intensities['ActivityHour'])

# Parse minute-level dates
minute_calories['ActivityMinute'] = pd.to_datetime(minute_calories['ActivityMinute'])
minute_steps['ActivityMinute'] = pd.to_datetime(minute_steps['ActivityMinute'])
minute_intensities['ActivityMinute'] = pd.to_datetime(minute_intensities['ActivityMinute'])
minute_METs['ActivityMinute'] = pd.to_datetime(minute_METs['ActivityMinute'])

# Parse heart rate
heartrate_seconds['Time'] = pd.to_datetime(heartrate_seconds['Time'])

# Add temporal features
daily_activity['DayOfWeek'] = daily_activity['ActivityDate'].dt.day_name()
daily_activity['IsWeekend'] = daily_activity['ActivityDate'].dt.dayofweek.isin([5, 6])
daily_activity['Week'] = daily_activity['ActivityDate'].dt.isocalendar().week

print("✓ All dates parsed successfully")
print(f"✓ Date range: {daily_activity['ActivityDate'].min()} to {daily_activity['ActivityDate'].max()}")
print(f"✓ Total days: {(daily_activity['ActivityDate'].max() - daily_activity['ActivityDate'].min()).days}")

# ============================================================================
# PART 5: USER ACTIVITY PATTERNS
# ============================================================================
print("\n" + "="*80)
print("PART 5: USER ACTIVITY PATTERNS")
print("-"*80)

# Records per user
user_records = daily_activity.groupby('Id').size().reset_index(name='Records')
print(f"\n📊 RECORDS PER USER:")
print(f"  Mean: {user_records['Records'].mean():.1f} days")
print(f"  Median: {user_records['Records'].median():.0f} days")
print(f"  Min: {user_records['Records'].min()} days")
print(f"  Max: {user_records['Records'].max()} days")

# Active users (those with >20 days of data)
active_users = user_records[user_records['Records'] >= 20]['Id'].values
print(f"  Active Users (≥20 days): {len(active_users)}")

# Sleep tracking
sleep_users = sleep_day['Id'].nunique()
print(f"\n💤 SLEEP TRACKING:")
print(f"  Users tracking sleep: {sleep_users} / {daily_activity['Id'].nunique()}")

# Weight tracking
weight_users = weight_log['Id'].nunique()
print(f"\n⚖️  WEIGHT TRACKING:")
print(f"  Users tracking weight: {weight_users} / {daily_activity['Id'].nunique()}")

# Heart rate tracking
hr_users = heartrate_seconds['Id'].nunique()
print(f"\n❤️  HEART RATE TRACKING:")
print(f"  Users with heart rate data: {hr_users} / {daily_activity['Id'].nunique()}")

# ============================================================================
# PART 6: HEALTH METRICS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("PART 6: HEALTH METRICS ANALYSIS")
print("-"*80)

print("\n🏃 ACTIVITY METRICS:")
print(f"  Average Daily Steps: {daily_activity['TotalSteps'].mean():,.0f}")
print(f"  Average Distance: {daily_activity['TotalDistance'].mean():.2f} km")
print(f"  Average Active Minutes: {daily_activity['VeryActiveMinutes'].mean() + daily_activity['FairlyActiveMinutes'].mean() + daily_activity['LightlyActiveMinutes'].mean():.1f}")
print(f"  Average Sedentary Minutes: {daily_activity['SedentaryMinutes'].mean():.1f}")
print(f"  Average Calories Burned: {daily_activity['Calories'].mean():.0f}")

print("\n💤 SLEEP METRICS:")
print(f"  Average Sleep Duration: {sleep_day['TotalMinutesAsleep'].mean() / 60:.2f} hours")
print(f"  Average Time in Bed: {sleep_day['TotalTimeInBed'].mean() / 60:.2f} hours")
print(f"  Average Sleep Efficiency: {(sleep_day['TotalMinutesAsleep'] / sleep_day['TotalTimeInBed']).mean() * 100:.1f}%")

if not weight_log.empty:
    print("\n⚖️  WEIGHT METRICS:")
    print(f"  Average Weight: {weight_log['WeightKg'].mean():.1f} kg")
    print(f"  Average BMI: {weight_log['BMI'].mean():.1f}")

print("\n❤️  HEART RATE METRICS:")
print(f"  Average Heart Rate: {heartrate_seconds['Value'].mean():.1f} bpm")
print(f"  Min Heart Rate: {heartrate_seconds['Value'].min():.0f} bpm")
print(f"  Max Heart Rate: {heartrate_seconds['Value'].max():.0f} bpm")

# ============================================================================
# PART 7: HEALTH GUIDELINES COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PART 7: HEALTH GUIDELINES COMPARISON")
print("-"*80)

# WHO/CDC Guidelines
STEP_GOAL = 10000
SLEEP_MIN = 7 * 60  # 7 hours
SLEEP_MAX = 9 * 60  # 9 hours
ACTIVE_MIN_GOAL = 30  # 30 minutes moderate activity per day

# Calculate adherence
daily_activity['MeetsStepGoal'] = daily_activity['TotalSteps'] >= STEP_GOAL
daily_activity['TotalActiveMinutes'] = (daily_activity['VeryActiveMinutes'] + 
                                        daily_activity['FairlyActiveMinutes'] + 
                                        daily_activity['LightlyActiveMinutes'])
daily_activity['MeetsActivityGoal'] = daily_activity['TotalActiveMinutes'] >= ACTIVE_MIN_GOAL

sleep_day['HealthySleep'] = (sleep_day['TotalMinutesAsleep'] >= SLEEP_MIN) & \
                            (sleep_day['TotalMinutesAsleep'] <= SLEEP_MAX)

print(f"\n🎯 GOAL ADHERENCE:")
print(f"  Days meeting step goal (10,000 steps): {daily_activity['MeetsStepGoal'].mean() * 100:.1f}%")
print(f"  Days meeting activity goal (30+ min): {daily_activity['MeetsActivityGoal'].mean() * 100:.1f}%")
print(f"  Days with healthy sleep (7-9 hours): {sleep_day['HealthySleep'].mean() * 100:.1f}%")

print("\n📈 COMPLETION STATUS:")
print("✓ Part 1-7 completed successfully!")
print("  Next: Feature Engineering, Heart Rate Aggregation, and Visualizations")
