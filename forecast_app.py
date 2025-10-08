import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Import the anomaly processing function that flags records using Isolation Forest
from anomaly_detection import flag_anomalous_records

# Check if Prophet is available
try:
    _ = Prophet()
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def run_cost_forecasting(df: pd.DataFrame):
    """
    Performs cost forecasting dynamically. It first cleans the data by using the 
    Isolation Forest results (daily records flagged as anomalies are removed) 
    before training the final forecast model.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Dynamic Cloud Cost Forecast (Isolation Forest Cleaned)")
    st.markdown("This forecast removes records flagged as anomalies by the **Isolation Forest** model before aggregating historical data for prediction.")
    
    df_train = df.copy()

    required_cols = ['Usage Start Date', 'Rounded Cost ($)']
    if not all(col in df_train.columns for col in required_cols):
         st.error(f"Training data is missing required columns: {', '.join(required_cols)}.")
         return
         
    df_train['Usage Start Date'] = pd.to_datetime(df_train['Usage Start Date'])

    if df_train.empty:
        st.error("The input historical training data is empty.")
        return

    # Calculate date range for guidance
    min_date = df_train['Usage Start Date'].min().strftime('%Y-%m-%d')
    max_date = df_train['Usage Start Date'].max().strftime('%Y-%m-%d')
    st.sidebar.markdown(f"**Full Historical Range:** {min_date} to {max_date}")

    # --- INPUT FOR VALIDATION DATA ---
    st.sidebar.header("Validation Data (Optional)")
    uploaded_validation_data = st.sidebar.file_uploader(
        "Upload separate Validation Data (CSV)",
        type=['csv'],
        help="Use this file for back-testing if you want to compare your forecast against actual costs for a past period."
    )
    
    # --- DYNAMIC FORECAST CONTROLS ---
    st.sidebar.header("Dynamic Prediction Horizon")
    
    default_prediction_start = (df_train['Usage Start Date'].max() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    
    prediction_start_date = st.sidebar.text_input(
        'Prediction Start Date (YYYY-MM-DD)',
        value=default_prediction_start,
        help="The first day of the month you wish to forecast. Must be after the end of historical data."
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
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Anomaly Removal:** Performed automatically via Isolation Forest on daily records.")

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
        placeholder='e.g., 2023-01-01',
        help="Enter a date if a major structural change (e.g., massive optimization, product launch) occurred to force a trend shift."
    )
    
    interval_conf = st.sidebar.slider(
        'Uncertainty Confidence (interval_width)',
        min_value=0.70, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        help="Controls the width of the lower/upper bounds."
    )
    
    st.info(f"Model will train on **Isolation Forest cleaned history** and predict **{prediction_length_months} months** starting **{prediction_start_date}**.")


    try:
        # --- 1. ISOLATION FOREST ANOMALY CLEANING ---
        st.subheader("🧹 Anomaly Detection & Cleaning (Daily Records)")
        
        # Use the Isolation Forest function to flag anomalous daily records
        # This function also displays a summary of detected anomalies in the Streamlit UI.
        df_flagged = flag_anomalous_records(df_train)
        
        # Filter out the anomalous records (is_anomaly == -1)
        df_clean_daily = df_flagged[df_flagged['is_anomaly'] != -1].copy()
        
        if df_clean_daily.empty:
            st.error("No data remains after anomaly removal. Please check your Isolation Forest contamination setting.")
            return

        # --- 2. AGGREGATE TO MONTHLY FOR PROPHET ---
        prophet_df = df_clean_daily.copy()
        
        # Aggregate the CLEANED daily costs to monthly (ds, y)
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        if len(prophet_df) < 2:
            st.error("Not enough monthly data (requires at least 2 clean months) to perform a reliable forecast after anomaly removal and aggregation.")
            return
            
        st.success(f"Training data aggregated. Using {len(prophet_df)} monthly observations for final forecast model training.")


        # --- 3. TRAIN THE FINAL MODEL ---
        changepoint_list = []
        if manual_changepoint:
            try:
                pd.to_datetime(manual_changepoint)
                changepoint_list.append(manual_changepoint)
            except ValueError:
                st.sidebar.error("Invalid date format for Manual Trend Change Date. Ignoring manual changepoint.")

        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_scale,
            interval_width=interval_conf,
            changepoints=changepoint_list if changepoint_list else None 
        )
        model.fit(prophet_df)

        # --- 4. DYNAMIC PREDICTION ---
        
        try:
            prediction_start_dt = pd.to_datetime(prediction_start_date)
        except ValueError:
            st.error("Invalid date format entered for Prediction Start Date. Please use YYYY-MM-DD format.")
            return

        
        last_historical_date = prophet_df['ds'].max()
        prediction_end_date = prediction_start_dt + pd.DateOffset(months=prediction_length_months)
        
        total_future_periods = (prediction_end_date.to_period('M') - last_historical_date.to_period('M')).n
        
        if total_future_periods <= 0:
             st.error("The Prediction Start Date must be *after* your last historical data point, or the Prediction Length is too short.")
             return
            
        future = model.make_future_dataframe(periods=total_future_periods, freq='M')
        
        # Generate the forecast
        forecast = model.predict(future)
        
        # Filter the final forecast results to the requested prediction window
        forecast_period = forecast[
            (forecast['ds'] >= prediction_start_dt) & 
            (forecast['ds'] < prediction_end_date)
        ].copy()

        forecast_period['ds'] = forecast_period['ds'].dt.to_period('M').dt.start_time

        # --- 5. VALIDATION (OPTIONAL) ---
        validation_df = None
        if uploaded_validation_data is not None:
            required_cols_val = ['Usage Start Date', 'Rounded Cost ($)']
            try:
                df_actual = pd.read_csv(uploaded_validation_data)
                
                if not all(col in df_actual.columns for col in required_cols_val):
                    st.error(f"Validation data is missing required columns: {', '.join(required_cols_val)}.")
                    raise ValueError("Missing columns in validation data.")

                df_actual['Usage Start Date'] = pd.to_datetime(df_actual['Usage Start Date'])
                
                # Aggregate actuals by month
                actual_data = df_actual.copy()
                actual_data['ds'] = actual_data['Usage Start Date'].dt.to_period('M').dt.start_time
                actual_data = actual_data.groupby('ds')['Rounded Cost ($)'].sum().reset_index()
                actual_data.columns = ['ds', 'Actual Cost']
                
                # Merge predicted and actual costs for validation
                validation_df = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
                    actual_data, on='ds', how='left'
                )
                
            except Exception as e:
                st.error(f"An error occurred while processing the validation file. Please check file format and date columns. ({e})")
        
        # --- 6. DISPLAY RESULTS ---
        
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
            if 'Actual Cost' in display_df.columns:
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
                columns={'ds': 'Date', 'yhat': 'Predicted Cost', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'},
                inplace=True
            )
            st.dataframe(
                display_df.style.format(
                    {'Predicted Cost': '${:,.2f}', 'Lower Bound': '${:,.2f}', 'Upper Bound': '${:,.2f}'}
                ),
                use_container_width=True
            )


        # --- 7. VISUALIZATION and Component Breakdown ---
        st.subheader("Forecast Visuals")
        st.write("The chart shows the clean historical costs (black dots) and the forecast period (blue line).")
        
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An unexpected error occurred during model processing: {e}")
        st.warning("Please verify your data preparation steps, especially date and cost column names.")
