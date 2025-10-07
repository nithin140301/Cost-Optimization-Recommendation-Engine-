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
    Performs cost forecasting using Facebook Prophet, provided the library is available,
    predicting total monthly cost for the next 12 months.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Cloud Cost Forecast")
    
    # --- MODEL TUNING CONTROLS ---
    st.sidebar.header("Forecast Tuning Parameters")
    
    # Allows tuning the trend sensitivity (to fix extreme negative or positive forecasts)
    changepoint_scale = st.sidebar.slider(
        'Trend Sensitivity (changepoint_prior_scale)',
        min_value=0.001, 
        max_value=0.5, 
        value=0.001, # Defaulting to a very low value to stabilize aggressive trends
        step=0.001,
        help="Lower values make the forecast trend smoother and less likely to swing aggressively (e.g., $17M or $-9M$ predictions). Start low and increase if the forecast is too flat."
    )
    
    # Allows tuning the uncertainty width (to fix collapsed bounds)
    interval_conf = st.sidebar.slider(
        'Uncertainty Confidence (interval_width)',
        min_value=0.70, 
        max_value=0.99, 
        value=0.95, 
        step=0.01,
        help="Controls the width of the lower/upper bounds. Use 0.95 for 95% confidence intervals."
    )
    
    st.info("Forecasting **total monthly cost** for the next 12 months using tuned parameters...")


    try:
        # --- MONTHLY DATA PREPARATION ---
        df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'])

        prophet_df = df.copy()
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']

        # Group and sum the total monthly costs
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']

        # Ensure enough data
        if len(prophet_df) < 2:
            st.error("Not enough historical monthly data (requires at least 2 months) to perform a reliable forecast.")
            return

        # --- TRAIN THE MODEL (WITH DYNAMIC TUNING) ---
        st.markdown(
            f"***Model Tuning Note:*** *The trend sensitivity (`changepoint_prior_scale`) is currently set to **{changepoint_scale}**. "
            f"The uncertainty confidence (`interval_width`) is set to **{interval_conf}**.*"
        )
        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
            # FIXED: Now using the user-defined slider values
            changepoint_prior_scale=changepoint_scale,
            interval_width=interval_conf
        )
        model.fit(prophet_df)

        # --- FUTURE DATES ---
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        st.subheader("Monthly Cost Forecast (Next 1 Year)")
        st.write("The shaded region represents the uncertainty bounds (confidence interval).")
        
        # Plot using Plotly (works in Streamlit)
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
        st.write("Review the Trend component to ensure it is not wildly positive or negative.")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)  # matplotlib figure

        # Display key figures
        last_month_cost = forecast.iloc[-1]['yhat']
        
        # Calculate the trend component for the last forecast month (for verification)
        last_trend = forecast.iloc[-1]['trend']
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        col1.metric(
            label="Predicted Monthly Cost (12 Months Out)",
            value=f"${last_month_cost:,.0f}"
        )
        col2.metric(
            label="Projected Trend Component (12 Months Out)",
            value=f"${last_trend:,.0f}"
        )
        st.markdown("---")


        st.dataframe(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']].tail(13).rename(
                columns={'ds': 'Date', 'yhat': 'Predicted Cost', 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound', 'trend': 'Trend Component'}
            ).style.format(
                {'Predicted Cost': '{:,.2f}', 'Lower Bound': '{:,.2f}', 'Upper Bound': '{:,.2f}', 'Trend Component': '{:,.2f}'}
            )
        )

        st.success("Monthly forecast analysis complete. Use the sidebar controls to tune the model.")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.warning("This error usually means the data format is inconsistent or Prophet ran into numerical issues.")
