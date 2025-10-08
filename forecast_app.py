import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Check if Prophet is available
try:
    _ = Prophet()
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    # Only raise error in the module, let the calling function handle the UI warning


def run_cost_forecasting(df: pd.DataFrame):
    """
    Performs cost forecasting dynamically. It uses the input DataFrame (df) for historical
    training and accepts a second optional file for validation/back-testing.
    The prediction period is entirely controlled by user inputs in the sidebar.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Dynamic Cloud Cost Forecast")
    st.markdown("Use this tool to forecast future monthly costs based on your historical data. Adjust the sidebar parameters for dynamic prediction and trend tuning.")
    
    df_train = df.copy()

    if df_train.empty:
        st.error("The input historical training data is empty.")
        return

    # --- INPUT FOR VALIDATION DATA ---
    st.sidebar.header("Validation Data (Optional)")
    uploaded_validation_data = st.sidebar.file_uploader(
        "Upload separate Validation Data (CSV)",
        type=['csv'],
        help="Use this file for back-testing if you want to compare your forecast against actual costs for a past period."
    )
    
    # --- DYNAMIC FORECAST CONTROLS ---
    st.sidebar.header("Dynamic Prediction Horizon")
    
    prediction_start_date = st.sidebar.text_input(
        'Prediction Start Date (YYYY-MM-DD)',
        # Set a reasonable default (e.g., the day after the last date in the training data, if possible)
        value='2023-01-01', 
        help="The first day of the month you wish to forecast."
    )

    prediction_length_months = st.sidebar.number_input(
        'Prediction Length (Months)',
        min_value=1,
        max_value=60,
        value=12,
        step=1,
        help="The number of months to forecast from the start date."
    )
    
    # --- MODEL TUNING CONTROLS ---
    st.sidebar.header("Model Tuning Parameters")

    # ANOMALY REMOVAL CONTROL (CRITICAL FOR VOLATILE DATA)
    anomaly_removal_start_date = st.sidebar.text_input(
        'Anomaly Removal Start Date (YYYY-MM-DD)',
        value='2022-07-01',
        help="Crucial for fixing corrupted trends. Set this date to start **after** any extreme, volatile peaks in your historical data. Data before this date is ignored."
    )

    # Trend Sensitivity control
    changepoint_scale = st.sidebar.slider(
        'Trend Sensitivity (changepoint_prior_scale)',
        min_value=0.001, 
        max_value=0.5, 
        value=0.50,
        step=0.001,
        help="Controls how flexible the model is when fitting trends. Set high (like 0.50) to stabilize against short-term volatility."
    )

    # Manual Changepoint control 
    manual_changepoint = st.sidebar.text_input(
        'Manual Trend Change Date (YYYY-MM-DD)',
        value='',
        placeholder='2023-01-01',
        help="Enter a date if a major structural change (e.g., massive optimization, product launch) occurred to force a trend shift."
    )
    
    interval_conf = st.sidebar.slider(
        'Uncertainty Confidence (interval_width)',
        min_value=0.70, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        help="Controls the width of the lower/upper bounds. For back-testing, check if the Actual Cost falls within the bounds."
    )
    
    st.info(f"Model will train on history starting from **{anomaly_removal_start_date}** and predict **{prediction_length_months} months** starting **{prediction_start_date}**.")


    try:
        # --- Data Preparation ---
        required_cols = ['Usage Start Date', 'Rounded Cost ($)']
        if not all(col in df_train.columns for col in required_cols):
             st.error(f"Training data is missing required columns: {', '.join(required_cols)}.")
             return
             
        df_train['Usage Start Date'] = pd.to_datetime(df_train['Usage Start Date'])
        
        # --- 1. APPLY ANOMALY REMOVAL FILTER ---
        df_train_filtered = df_train.copy()
        if anomaly_removal_start_date:
            try:
                start_date = pd.to_datetime(anomaly_removal_start_date)
                df_train_filtered = df_train[df_train['Usage Start Date'] >= start_date].copy()
                st.markdown(f"***Training Data Filtered:*** *Only using data from **{start_date.strftime('%Y-%m-%d')}** onwards.*")
            except Exception:
                st.sidebar.error("Invalid date format for Anomaly Removal Start Date. Using all historical data.")

        # Re-check for enough data after filtering
        if df_train_filtered.empty:
            st.error("Training data is empty after applying the 'Anomaly Removal Start Date' filter.")
            return

        prophet_df = df_train_filtered.copy()
        # Aggregate by the start of the month
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        if len(prophet_df) < 2:
            st.error("Not enough historical monthly data (requires at least 2 months) to perform a reliable forecast after aggregation.")
            return

        # --- TRAIN THE MODEL (WITH DYNAMIC TUNING) ---
        changepoint_list = []
        if manual_changepoint:
            try:
                pd.to_datetime(manual_changepoint)
                changepoint_list.append(manual_changepoint)
            except:
                st.sidebar.error("Invalid date format for Manual Trend Change Date. Ignoring manual changepoint.")
                changepoint_list = []

        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_scale,
            interval_width=interval_conf,
            changepoints=changepoint_list if changepoint_list else None 
        )
        model.fit(prophet_df)

        # --- DYNAMIC PREDICTION ---
        
        # Calculate the number of total periods needed to generate history + forecast
        prediction_start_dt = pd.to_datetime(prediction_start_date)
        
        # Determine the total number of months to predict starting from the *earliest date* in history.
        # This simplifies the future DataFrame creation.
        
        # Find the difference in months between the last historical date and the desired prediction end date
        last_historical_date = prophet_df['ds'].max()
        prediction_end_date = prediction_start_dt + pd.DateOffset(months=prediction_length_months)
        
        # Ensure we have enough future periods to cover the prediction window
        total_future_periods = (prediction_end_date.to_period('M') - last_historical_date.to_period('M')).n
        
        # Prophet uses the number of future periods *after* the historical data
        future = model.make_future_dataframe(periods=total_future_periods, freq='M')
        
        # Clip the future dataframe to only include dates >= the desired prediction start date
        future_forecast = future[future['ds'] >= prediction_start_dt].copy()
        
        if future_forecast.empty:
            st.error("The prediction start date is not after the historical data end date, or the length is too short.")
            return
            
        # Generate the forecast
        forecast = model.predict(future)
        
        # Filter the final forecast results to the requested prediction window
        forecast_period = forecast[
            (forecast['ds'] >= prediction_start_dt) & 
            (forecast['ds'] < prediction_end_date)
        ].copy()

        # Format dates for the table
        forecast_period['ds'] = forecast_period['ds'].dt.to_period('M').dt.start_time

        # --- VALIDATION ---
        validation_df = None
        if uploaded_validation_data is not None:
            try:
                df_actual = pd.read_csv(uploaded_validation_data)
                
                if not all(col in df_actual.columns for col in required_cols):
                    st.error(f"Validation data is missing required columns: {', '.join(required_cols)}.")
                    raise ValueError("Missing columns in validation data.")

                df_actual['Usage Start Date'] = pd.to_datetime(df_actual['Usage Start Date'])
                
                # Aggregate actuals by month
                actual_data = df_actual.copy()
                actual_data['ds'] = actual_data['Usage Start Date'].dt.to_period('M').dt.start_time
                actual_data = actual_data.groupby('ds')['Rounded Cost ($)'].sum().reset_index()
                actual_data.columns = ['ds', 'Actual Cost']
                
                # Merge predicted and actual costs for validation only in the predicted period
                validation_df = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
                    actual_data, on='ds', how='left'
                )
                
            except Exception as e:
                st.error(f"An error occurred while processing the validation file: {e}")
        
        # --- DISPLAY RESULTS ---
        
        # Determine which dataframe to display
        if validation_df is not None:
            display_df = validation_df.copy()
            st.subheader("Forecast and Validation Results")
            display_df.rename(
                columns={
                    'ds': 'Date', 
                    'yhat': 'Predicted Cost', 
                    'yhat_lower': 'Lower Bound', 
                    'yhat_upper': 'Upper Bound'
                },
                inplace=True
            )
            display_df['Error ($)'] = (display_df['Predicted Cost'] - display_df['Actual Cost']).abs()

            st.dataframe(
                display_df.style.format(
                    {'Predicted Cost': '${:,.2f}', 
                     'Lower Bound': '${:,.2f}', 
                     'Upper Bound': '${:,.2f}', 
                     'Actual Cost': '${:,.2f}',
                     'Error ($)': '${:,.2f}'}
                ),
                use_container_width=True
            )
            
        else:
            display_df = forecast_period.copy()
            st.subheader("Dynamic Forecast Results")
            display_df.rename(
                columns={'ds': 'Date', 'yhat': 'Predicted Cost', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
            ).style.format(
                {'Predicted Cost': '${:,.2f}', 'Lower Bound': '${:,.2f}', 'Upper Bound': '${:,.2f}'}
            )
            st.dataframe(
                display_df.style.format(
                    {'Predicted Cost': '${:,.2f}', 'Lower Bound': '${:,.2f}', 'Upper Bound': '${:,.2f}'}
                ),
                use_container_width=True
            )


        # --- VISUALIZATION and Component Breakdown ---
        st.subheader("Forecast Visuals")
        st.write("The chart shows the historical costs (black dots) and the forecast period (blue line).")
        
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An unexpected error occurred during forecasting: {e}")
        st.warning("Please verify all date inputs are in 'YYYY-MM-DD' format and that the required columns ('Usage Start Date', 'Rounded Cost ($)') are present.")
