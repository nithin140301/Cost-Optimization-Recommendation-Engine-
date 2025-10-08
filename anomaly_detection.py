import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import streamlit as st

def flag_anomalous_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects cost anomalies (spikes) using Isolation Forest based on cost and usage metrics 
    and adds an 'is_anomaly' column to the DataFrame. This function implements the 
    user's requested logic (Isolation Forest on daily records).

    The function flags the anomalies and returns the DataFrame for use in the main app.

    Returns:
        pd.DataFrame: Original DataFrame with an added 'is_anomaly' column (-1 for anomaly, 1 for normal).
                      Records treated as normal (1) if features are missing or non-numeric.
    """
    if df.empty:
        # Assign all as normal if empty
        return df.assign(is_anomaly=1) 

    # Define features for anomaly detection
    features = ['Rounded Cost ($)', 'Duration (Hours)', 'Avg Utilization (%)']

    # Check for missing features required by Isolation Forest
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        st.error(f"Anomaly detection failed: Missing required features: {', '.join(missing_features)}. All records will be treated as normal (1).")
        return df.assign(is_anomaly=1)


    # --- Data Preparation ---
    X = df[features].copy()

    # Handle NaNs and non-finite values by dropping records from the training set
    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    if X_clean.empty:
        st.error("Anomaly detection failed: No clean numeric data available after removing missing/non-finite values.")
        return df.assign(is_anomaly=1)

    # --- Model Training & Prediction ---
    # Contamination is set to 0.05 (5%)
    model = IsolationForest(
        random_state=42,
        contamination=0.05, 
        n_estimators=100
    )
    model.fit(X_clean)

    # Calculate prediction on the clean subset X_clean
    anomaly_predictions = model.predict(X_clean)

    # --- Re-merge results into the original DataFrame ---
    
    # Create a Series mapping the index of clean data to its anomaly prediction
    anomaly_series = pd.Series(anomaly_predictions, index=X_clean.index)
    
    # Initialize the new column in the original DataFrame to '1' (Normal)
    df['is_anomaly'] = 1 
    
    # Update the flag only for the records that were successfully processed
    df.loc[anomaly_series.index, 'is_anomaly'] = anomaly_series
    
    anomalies_count = len(df[df['is_anomaly'] == -1])
    
    if anomalies_count > 0:
        st.info(f"Isolation Forest identified **{anomalies_count}** daily usage records as anomalies and they will be removed before forecasting.")
        # Optional: Display the high-level anomaly list for user review
        anomalies = df[df['is_anomaly'] == -1]
        with st.expander("Review Daily Anomalies (Isolated Records)"):
            st.dataframe(
                anomalies[['Resource ID', 'Service Name', 'Rounded Cost ($)', 'Usage Start Date']].sort_values(by='Rounded Cost ($)', ascending=False)
            )
    else:
        st.success("No anomalies detected in the daily usage records.")
        
    return df
