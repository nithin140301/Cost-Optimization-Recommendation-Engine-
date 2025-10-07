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
    # st.error("Prophet library not found. Please install it using 'pip install prophet' to enable forecasting.")


def run_cost_forecasting(df: pd.DataFrame):
    """
    Performs cost forecasting specifically for the back-testing scenario:
    1. Trains on the data passed in 'df' (assumed to be 2022 data).
    2. Uses a Streamlit file uploader to get the validation data (2023 data).
    3. Predicts 12 months and displays a comparison table of Predicted vs. Actual 2023 costs.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Cloud Cost Forecast (Back-Testing: 2022 -> Predict 2023)")
    
    # RENAME INPUT DF FOR CLARITY: df is now the 2022 Training Data
    df_train = df.copy()

    if df_train.empty:
        st.warning("The input data (expected 2022 training data) is empty.")
        return

    # --- INPUT FOR VALIDATION DATA (2023) ---
    st.sidebar.header("Validation Data")
    uploaded_file_2023 = st.sidebar.file_uploader(
        "Upload Actual 2023 Data (data_2023.csv)",
        type=['csv'],
        help="This file is used to validate the 2023 predictions made by the model trained on 2022 data."
    )
    
    # --- MODEL TUNING CONTROLS ---
    st.sidebar.header("Forecast Tuning Parameters")
    
    changepoint_scale = st.sidebar.slider(
        'Trend Sensitivity (changepoint_prior_scale)',
        min_value=0.001, 
        max_value=0.5, 
        value=0.001, 
        step=0.001,
        help="Lower values make the forecast trend smoother and prevent extreme predictions (like $17M or $-9M$). Use a very low value (e.g., 0.001) for stable trends."
    )
    
    interval_conf = st.sidebar.slider(
        'Uncertainty Confidence (interval_width)',
        min_value=0.70, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        help="Controls the width of the lower/upper bounds. For back-testing, check if the Actual Cost falls within the Lower and Upper Bound."
    )
    
    st.info("Model training on the uploaded 2022 data. Predicting the **12 months of 2023**.")


    try:
        # --- Data Preparation ---
        # Robustly check for required columns
        required_cols = ['Usage Start Date', 'Rounded Cost ($)']
        if not all(col in df_train.columns for col in required_cols):
             st.error(f"Training data is missing required columns: {', '.join(required_cols)}.")
             return
             
        df_train['Usage Start Date'] = pd.to_datetime(df_train['Usage Start Date'])

        prophet_df = df_train.copy()
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']

        # Group and sum the total monthly costs
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure enough data
        if len(prophet_df) < 2:
            st.error("Not enough historical monthly data (requires at least 2 months) in the 2022 training data to perform a reliable forecast.")
            return

        # --- TRAIN THE MODEL (WITH DYNAMIC TUNING) ---
        st.markdown(
            f"***Model Tuning Note:*** *Trend sensitivity set to **{changepoint_scale}**. Confidence interval set to **{interval_conf}**.*"
        )
        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=changepoint_scale,
            interval_width=interval_conf
        )
        model.fit(prophet_df)

        # --- PREDICTION (12 months of 2023) ---
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        
        # Filter down to just the 12 forecast months
        forecast_2023 = forecast.tail(12).copy()
        forecast_2023['ds'] = forecast_2023['ds'].dt.to_period('M').dt.start_time

        # --- VALIDATION (Load Actual 2023 Data from Uploader) ---
        if uploaded_file_2023 is not None:
            try:
                # Load the 2023 validation data from the Streamlit uploader
                df_actual_2023 = pd.read_csv(uploaded_file_2023)
                
                # Check required columns for validation data
                if not all(col in df_actual_2023.columns for col in required_cols):
                    st.error(f"Validation data is missing required columns: {', '.join(required_cols)}.")
                    raise ValueError("Missing columns in validation data.")

                df_actual_2023['Usage Start Date'] = pd.to_datetime(df_actual_2023['Usage Start Date'])
                
                # Aggregate 2023 actuals by month
                actual_2023 = df_actual_2023.copy()
                actual_2023['ds'] = actual_2023['Usage Start Date'].dt.to_period('M').dt.start_time
                actual_2023 = actual_2023.groupby('ds')['Rounded Cost ($)'].sum().reset_index()
                actual_2030.columns = ['ds', 'Actual Cost']
                
                # Merge predicted and actual costs
                validation_df = forecast_2023[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
                    actual_2023, on='ds', how='left'
                )
                
                validation_df.rename(
                    columns={
                        'ds': 'Date', 
                        'yhat': 'Predicted Cost', 
                        'yhat_lower': 'Lower Bound', 
                        'yhat_upper': 'Upper Bound'
                    },
                    inplace=True
                )
                
                # Calculate the prediction error (absolute difference)
                validation_df['Error ($)'] = (validation_df['Predicted Cost'] - validation_df['Actual Cost']).abs()
                
                st.subheader("Model Validation: Predicted vs. Actual (2023)")
                st.dataframe(
                    validation_df.style.format(
                        {'Predicted Cost': '${:,.2f}', 
                         'Lower Bound': '${:,.2f}', 
                         'Upper Bound': '${:,.2f}', 
                         'Actual Cost': '${:,.2f}',
                         'Error ($)': '${:,.2f}'}
                    )
                )
                st.success(f"Validation against 2023 data complete. Use the **Trend Sensitivity** slider to minimize the average **Error (\$)**.")
                
            except Exception as e:
                # Fall-through to display forecast if validation fails
                st.error(f"An error occurred while processing the 2023 validation file. Displaying forecast only. ({e})")

                # Display forecast table without validation data
                forecast_output = forecast_2023
                st.dataframe(
                    forecast_output[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                        columns={'ds': 'Date', 'yhat': 'Predicted Cost', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
                    ).style.format(
                        {'Predicted Cost': '{:,.2f}', 'Lower Bound': '{:,.2f}', 'Upper Bound': '{:,.2f}'}
                    )
                )

        
        else:
            st.warning("Please upload your **Actual 2023 Data (data_2023.csv)** using the file uploader in the sidebar to perform validation.")
            # Display forecast table without validation data
            forecast_output = forecast_2023
            st.dataframe(
                forecast_output[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Predicted Cost', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'}
                ).style.format(
                    {'Predicted Cost': '{:,.2f}', 'Lower Bound': '{:,.2f}', 'Upper Bound': '{:,.2f}'}
                )
            )


        # --- VISUALIZATION and Component Breakdown ---
        st.subheader("Forecast Visuals (Trained on 2022, Predicted 2023)")
        st.write("The chart shows the 2022 historical costs (black dots) and the 2023 prediction (blue line).")
        
        # Plot only the historical and forecast data
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.warning("Ensure the 'Usage Start Date' and 'Rounded Cost ($)' columns are present and correctly formatted in your training data.")
