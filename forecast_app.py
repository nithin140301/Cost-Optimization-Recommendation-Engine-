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
    st.error("Prophet library not found. Please install it using 'pip install prophet' to enable forecasting.")


def run_cost_forecasting(df: pd.DataFrame):
    """
    Performs cost forecasting specifically for the back-testing scenario:
    1. Trains on 2022 data ('data_2022.csv').
    2. Predicts 12 months (for validation against 2023).
    3. Displays a comparison table of Predicted vs. Actual 2023 costs.
    
    NOTE: The 'df' argument passed to this function is ignored, as it loads the
    training data directly from 'data_2022.csv'.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Cloud Cost Forecast (Back-Testing: 2022 -> Predict 2023)")
    
    # --- Data Loading (Train on 2022) ---
    @st.cache_data
    def _load_training_data():
        """Loads and prepares 2022 training data."""
        try:
            # Explicitly load data_2022.csv for training
            df_2022 = pd.read_csv("data_2022.csv")
            st.sidebar.success("Loaded 'data_2022.csv' for training (Historical Data).")
            return df_2022
        except FileNotFoundError as e:
            st.error(f"Error loading training file: {e}. Ensure 'data_2022.csv' is available.")
            return pd.DataFrame()

    df_train = _load_training_data()
    
    if df_train.empty:
        return

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
        help="Controls the width of the lower/upper bounds (e.g., 0.95 for 95% confidence)."
    )
    
    st.info("Training model on **2022 data** only. Predicting the **12 months of 2023**.")


    try:
        # --- Data Preparation ---
        df_train['Usage Start Date'] = pd.to_datetime(df_train['Usage Start Date'])

        prophet_df = df_train.copy()
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']

        # Group and sum the total monthly costs
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        # Ensure enough data
        if len(prophet_df) < 2:
            st.error("Not enough historical monthly data (requires at least 2 months in 'data_2022.csv') to perform a reliable forecast.")
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
        # Predict 12 periods, covering 2023
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)
        
        # Filter down to just the 12 forecast months
        forecast_2023 = forecast.tail(12).copy()
        # Convert date to start of month for clean merging
        forecast_2023['ds'] = forecast_2023['ds'].dt.to_period('M').dt.start_time

        # --- VALIDATION (Load Actual 2023 Data) ---
        try:
            # Explicitly load data_2023.csv for validation
            df_actual_2023 = pd.read_csv("data_2023.csv")
            df_actual_2023['Usage Start Date'] = pd.to_datetime(df_actual_2023['Usage Start Date'])
            
            # Aggregate 2023 actuals by month
            actual_2023 = df_actual_2023.copy()
            actual_2023['ds'] = actual_2023['Usage Start Date'].dt.to_period('M').dt.start_time
            actual_2023 = actual_2023.groupby('ds')['Rounded Cost ($)'].sum().reset_index()
            actual_2023.columns = ['ds', 'Actual Cost']
            
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
            st.success(f"Validation against 'data_2023.csv' complete. Use the **Trend Sensitivity** slider to minimize the average **Error (\$)**.")
            
        except FileNotFoundError:
            st.warning("Could not load 'data_2023.csv'. Displaying 2023 forecast without actuals for validation.")
            # Fallback to display the forecast without validation data
            forecast_output = forecast.tail(12)
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
        st.warning("Ensure the 'Usage Start Date' and 'Rounded Cost ($)' columns are present and correctly formatted in 'data_2022.csv'.")
