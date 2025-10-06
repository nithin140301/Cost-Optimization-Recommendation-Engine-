import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

def run_anomaly_detection(df: pd.DataFrame):
    """
    Detects cost anomalies (spikes) using Isolation Forest based on cost and usage metrics.
    """
    if df.empty:
        st.warning("No valid data available for anomaly detection.")
        return

    st.info("Using Isolation Forest to find cost outliers...")

    # Define features for anomaly detection (Cost, Duration, and Utilization)
    features = ['Rounded Cost ($)', 'Duration (Hours)', 'Avg Utilization (%)']

    # --- Feature Engineering & Data Preparation ---
    # Ensure Duration (Hours) and Avg Utilization (%) columns are available and numerical
    # Note: Your initial data did not have these, they are likely calculated earlier in your app.
    # For this code to run, these columns must be added to 'df' before this function is called.

    # Check if the required features are present and handle non-numeric data
    try:
        # 1. Prepare data for model
        X = df[features].copy()

        # Drop rows with NaN or infinite values that would break the Isolation Forest model
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(inplace=True)

        if X.empty:
            st.error("Anomaly detection failed: No clean numeric data available in required features.")
            return

        # --- Model Training ---
        model = IsolationForest(
            random_state=42,
            contamination=0.05,
            n_estimators=100
        )
        model.fit(X)

        # --- FIX: Calculate prediction and score on the 3 features, then assign ---
        # Calculate scores and predictions using the 3 features in X
        anomaly_scores = model.decision_function(X)
        anomaly_predictions = model.predict(X)

        # Assign results back to X
        X['anomaly_score'] = anomaly_scores
        X['is_anomaly'] = anomaly_predictions
        # --- END FIX ---


        # Merge the results back to the original DataFrame
        df_with_anomalies = df.loc[X.index].copy()
        df_with_anomalies['is_anomaly'] = X['is_anomaly']

        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == -1]

        st.subheader("Results")
        st.metric(
            label="Total Cost Anomalies Detected",
            value=len(anomalies),
            delta_color="off"
        )

        # --- Visualization ---
        if not anomalies.empty:
            st.warning(f"🚨 **{len(anomalies)}** usage records detected as high-priority cost anomalies!")

            # Plot the anomalies against the Usage Start Date for better context
            plot_df = df_with_anomalies.copy()
            plot_df['Anomaly Type'] = plot_df['is_anomaly'].apply(
                lambda x: 'Anomaly' if x == -1 else 'Normal'
            )
            plot_df['Usage Start Date'] = pd.to_datetime(plot_df['Usage Start Date'])

            fig = px.scatter(
                plot_df,
                x='Usage Start Date', # Use the date column for a time series plot
                y='Rounded Cost ($)',
                color='Anomaly Type',
                color_discrete_map={'Anomaly': 'red', 'Normal': 'blue'},
                title="Cost vs. Time (Anomalies Highlighted)"
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Details of Anomalous Usage Records"):
                st.dataframe(
                    anomalies[[
                        'Resource ID', 'Service Name', 'Region / Zone', 'Rounded Cost ($)',
                        'Usage Start Date', 'Usage End Date'
                    ]].sort_values(by='Rounded Cost ($)', ascending=False)
                )
        else:
            st.success("✅ No significant cost anomalies detected using current thresholds.")

    except Exception as e:
        # Re-raise the error for better debugging if it's not the intended feature mismatch
        st.error(f"An unexpected error occurred during anomaly detection: {e}")
        st.info("Ensure the required features ('Rounded Cost ($)', 'Duration (Hours)', 'Avg Utilization (%)') are correctly calculated and contain numerical data.")