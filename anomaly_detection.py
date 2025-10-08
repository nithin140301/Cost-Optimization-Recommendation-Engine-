import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import streamlit as st
import plotly.express as px

# Helper function used by forecast_app.py to flag records
def flag_anomalous_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects cost anomalies (spikes) using Isolation Forest based on cost and usage metrics 
    and adds an 'is_anomaly' column to the DataFrame. This function implements the 
    user's requested logic (Isolation Forest on daily records).
    
    Returns:
        pd.DataFrame: Original DataFrame with an added 'is_anomaly' column (-1 for anomaly, 1 for normal).
    """
    if df.empty:
        return df.assign(is_anomaly=1) 

    # Define features for anomaly detection
    features = ['Rounded Cost ($)', 'Duration (Hours)', 'Avg Utilization (%)']

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        # Note: We won't error out here, just return all as normal if used by forecast_app
        return df.assign(is_anomaly=1)

    X = df[features].copy()
    X_clean = X.replace([np.inf, -np.inf], np.nan).dropna()
    
    if X_clean.empty:
        return df.assign(is_anomaly=1)

    # Model Training & Prediction
    model = IsolationForest(
        random_state=42,
        contamination=0.05, 
        n_estimators=100
    )
    model.fit(X_clean)

    anomaly_predictions = model.predict(X_clean)

    # Re-merge results into the original DataFrame
    anomaly_series = pd.Series(anomaly_predictions, index=X_clean.index)
    df['is_anomaly'] = 1 
    df.loc[anomaly_series.index, 'is_anomaly'] = anomaly_series
        
    return df

# Main function used by app.py to run and display anomaly detection
def run_anomaly_detection(df: pd.DataFrame):
    """
    Main execution function for the Anomaly Detection dashboard view.
    """
    st.title("🚨 Cloud Cost Anomaly Detection (Isolation Forest)")
    st.markdown("This module identifies outliers in daily cost using the **Isolation Forest** algorithm, leveraging cost, duration, and utilization features.")
    
    if df.empty:
        st.warning("Please upload and process data to begin anomaly detection.")
        return
        
    df_result = df.copy()
    
    # Ensure date column is ready
    if 'Usage Start Date' in df_result.columns:
        df_result['Usage Start Date'] = pd.to_datetime(df_result['Usage Start Date'])
    else:
        st.error("Missing 'Usage Start Date' column. Cannot perform time-series anomaly detection.")
        return

    # Aggregate data to Daily Cost
    daily_cost_df = df_result.groupby(df_result['Usage Start Date'].dt.date)['Rounded Cost ($)'].sum().reset_index()
    daily_cost_df.columns = ['Date', 'Rounded Cost ($)']
    daily_cost_df['Date'] = pd.to_datetime(daily_cost_df['Date'])
    
    st.subheader("1. Anomaly Flagging")
    
    # Run the flagging function
    df_flagged = flag_anomalous_records(df_result.copy())
    
    anomalies_count = len(df_flagged[df_flagged['is_anomaly'] == -1])
    
    if anomalies_count > 0:
        st.info(f"Isolation Forest identified **{anomalies_count}** daily usage records as anomalies.")
        
        # Merge flags to the daily aggregated data for plotting
        daily_cost_df['is_anomaly'] = 1
        
        # Since we ran flagging on raw data, we need to map the anomaly flag to the daily sum.
        # Simple approach: If any record on a given date is an anomaly, flag the entire day's sum.
        anomalous_dates = df_flagged[df_flagged['is_anomaly'] == -1]['Usage Start Date'].dt.date.unique()
        daily_cost_df.loc[daily_cost_df['Date'].dt.date.isin(anomalous_dates), 'is_anomaly'] = -1

        # --- Visualization ---
        st.subheader("2. Anomaly Visualization")
        
        daily_cost_df['Anomaly_Color'] = daily_cost_df['is_anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        fig = px.scatter(
            daily_cost_df,
            x='Date',
            y='Rounded Cost ($)',
            color='Anomaly_Color',
            title='Daily Cloud Cost and Detected Anomalies',
            labels={'Rounded Cost ($)': 'Daily Cost ($)', 'Date': 'Usage Date'},
            color_discrete_map={'Normal': '#00B894', 'Anomaly': '#D63031'} # Green for normal, Red for anomaly
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.8), selector=dict(mode='markers'))
        fig.update_layout(xaxis_title="Date", yaxis_title="Daily Cost ($)", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # --- Detailed List ---
        st.subheader(f"3. Detailed Anomaly Records ({anomalies_count} Records)")
        anomalies = df_flagged[df_flagged['is_anomaly'] == -1].copy()
        
        # Calculate derived metrics for better context
        anomalies['Cost Rank'] = anomalies['Rounded Cost ($)'].rank(ascending=False, method='min').astype(int)
        anomalies.sort_values(by='Rounded Cost ($)', ascending=False, inplace=True)
        
        st.dataframe(
            anomalies[[
                'Cost Rank', 
                'Usage Start Date', 
                'Rounded Cost ($)', 
                'Service Name', 
                'Resource ID',
                'Usage Quantity',
                'Usage Unit'
            ]].style.format({'Rounded Cost ($)': '${:,.2f}'}),
            use_container_width=True
        )
        
        st.markdown(
            """
            <div style='padding: 10px; background-color: #ffeaa7; border-radius: 8px; margin-top: 20px;'>
                ⚠️ **Next Steps:** Review these anomalous records. They suggest unusual cost spikes. 
                If they are true spikes (not planned), they indicate optimization opportunities or unexpected usage. 
                These records will be automatically **removed** before running the **Forecast** model for a cleaner prediction.
            </div>
            """,
            unsafe_allow_html=True
        )
        
    else:
        st.success("No significant daily cost anomalies detected using Isolation Forest (contamination=5%).")
