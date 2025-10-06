import pandas as pd
import streamlit as st
import numpy as np
from io import StringIO

def process_uploaded_data(uploaded_file):
    """
    Loads the uploaded CSV, performs necessary cleaning, type conversions,
    and calculates essential optimization metrics.
    """
    try:
        # Load the file into a Pandas DataFrame
        df = pd.read_csv(uploaded_file)

        # --- Core Column Validation ---
        required_cols = [
            'Resource ID', 'Service Name', 'Usage Quantity', 'Usage Unit',
            'Region / Zone', 'CPU Utilization (%)', 'Memory Utilization (%)',
            'Usage Start Date', 'Usage End Date', 'Rounded Cost ($)'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {', '.join(missing_cols)}. Check your file format.")
            return None

        # --- Data Cleaning and Type Conversion ---

        # 1. Date/Time Conversion
        df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'], errors='coerce', utc=True)
        df['Usage End Date'] = pd.to_datetime(df['Usage End Date'], errors='coerce', utc=True)

        # Drop rows where date conversion failed (should be minimal)
        df.dropna(subset=['Usage Start Date', 'Usage End Date'], inplace=True)

        # 2. Numeric Conversion (handle potential non-numeric entries like strings or errors)
        numeric_cols = ['CPU Utilization (%)', 'Memory Utilization (%)', 'Rounded Cost ($)']
        for col in numeric_cols:
            # Coerce errors to NaN, then fill NaN with 0 for utilization
            if col != 'Rounded Cost ($)':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                # Fill cost NaNs with 0.0, but log if many are missing
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # 3. Handle data types for utilization (ensure they are percentages)
        df['CPU Utilization (%)'] = df['CPU Utilization (%)'].clip(lower=0.0, upper=100.0)
        df['Memory Utilization (%)'] = df['Memory Utilization (%)'].clip(lower=0.0, upper=100.0)


        # --- Metric Calculation (Optimization Features) ---

        # 1. Calculate Usage Duration (in hours)
        df['Duration (Hours)'] = (df['Usage End Date'] - df['Usage Start Date']).dt.total_seconds() / 3600

        # 2. Calculate Wasted Cost Metric
        # Wasted Cost is based on resources running for many hours but being underutilized
        # Define 'Underutilized' as < 10% average utilization across CPU and Memory
        df['Avg Utilization (%)'] = (df['CPU Utilization (%)'] + df['Memory Utilization (%)']) / 2

        # Flag resources for rightsizing/termination
        LOW_UTIL_THRESHOLD = 10.0
        df['Optimization Flag'] = np.where(
            (df['Duration (Hours)'] > 100) & (df['Avg Utilization (%)'] < LOW_UTIL_THRESHOLD),
            'Rightsizing/Termination Candidate',
            'Optimal/Acceptable'
        )

        # Calculate a potential monthly savings estimate (simple linear projection)
        # Assuming the resource runs similarly for a 30-day month
        df['Monthly Cost Estimate ($)'] = (df['Rounded Cost ($)'] / df['Duration (Hours)']) * 720

        # Final cleanup: drop any rows that ended up with zero cost or zero duration after cleaning
        df = df[(df['Rounded Cost ($)'] > 0) & (df['Duration (Hours)'] > 0)].copy()

        st.info(f"Processed {len(df)} valid usage records.")
        return df

    except Exception as e:
        st.error(f"An unexpected error occurred during data processing: {e}")
        st.markdown("Please ensure the CSV is properly formatted and contains the required columns with valid data types.")
        return None
