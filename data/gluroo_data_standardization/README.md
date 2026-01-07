# Data Standardization

This directory contains scripts for standardizing and merging health data from multiple sources with varying frequencies and file structures.

## Data Frequency and File Structure Challenges

The data sources have varying logging frequencies and file organization patterns that require special handling:

### 1. Logging Frequencies

#### Regular Intervals
- Blood Glucose (CGM): Every 5 minutes
- SpO2: Every minute during wear time
- Temperature: Every minute during wear time
- Stress Score: Every 5 minutes during wear time

#### Daily Aggregates
- Sleep Score: Once per day
- Daily Readiness: Once per day
- HRV Summary: Daily statistics
- Respiratory Rate Summary: Daily statistics

#### Event-Based (Irregular)
- Insulin Doses: When administered
- Meal Announcements: When food is consumed
- Finger Prick Readings: Manual checks
- HRV Details: During specific activities/rest

#### Event-Triggered Measurements
- Blood Glucose: Additional readings when meals/insulin are reported
- Heart Rate: Additional readings during exercise or stress events
- HRV: Additional measurements during significant stress changes
- Temperature: More frequent readings during detected fever/illness

These event-triggered measurements complement the regular interval data, providing additional context around important health events.

### 2. File Organization

#### Single File per Time Range
- Blood Glucose: CSV files covering a few months each
  - Example: `679372_5th-7th.csv`, `679372_7th-9th.csv`
- Sleep Profile: One consolidated file
  - Example: `Sleep Profile.csv`
- Stress Score: One consolidated file
  - Example: `Stress Score.csv`

#### Daily Files
- Daily Readiness Score
  - Example: `Daily Readiness Score - 2024-04-01.csv`
- HRV Details
  - Example: `Heart Rate Variability Details - 2024-04-03.csv`
- Respiratory Rate Summary
  - Example: `Daily Respiratory Rate Summary - 2024-04-02.csv`

#### Monthly Files
- SpO2 Measurements
  - Example: `Minute SpO2 - 2024-01.csv`
- Temperature Readings
  - Example: `Device Temperature - 2024-01.csv`
- HRV Summary
  - Example: `Daily Heart Rate Variability Summary - 2024-01.csv`

### 3. Handling Strategies

#### File Discovery and Loading
- Uses Python's `glob` to dynamically find and load files
  ```python
  files = glob.glob('data/fitness/Temperature/Device Temperature - *.csv')
  ```
- Handles different date formats in filenames
  ```python
  # Examples of date patterns in filenames:
  # - YYYY-MM-DD: Daily Readiness Score - 2024-04-01.csv
  # - YYYY-MM: Minute SpO2 - 2024-01.csv
  # - Custom: 679372_5th-7th.csv
  ```

#### Time Period Management
- Merges overlapping time periods carefully
  - Checks for duplicates across files
  - Validates data consistency in overlaps
- Maintains original granularity while aligning timestamps
  - Regular interval data: Kept at original frequency
  - Event data: Preserved at exact timestamps
  - Daily data: Aligned to start of day

#### Data Quality
- Special handling for gaps in wear time/device removal
  - Marks gaps in continuous data
  - Validates data before/after gaps
- Handles timezone inconsistencies
  - Standardizes all timestamps to UTC
  - Preserves original timezone information

## Scripts

- `standardize_health_data.py`: Main data processing script
- `validate_health_data.py`: Data validation and quality checks
- `visualize_health_data.py`: Interactive visualization