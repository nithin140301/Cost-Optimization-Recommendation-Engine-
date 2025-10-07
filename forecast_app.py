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
    
    FIX 1: The changepoint_prior_scale has been reduced to 0.005 to prevent the model
    from overreacting to the steep $2M cost cut observed in December 2023.
    
    FIX 2: The interval_width is set to 0.95 to ensure the uncertainty bounds (yhat_lower/upper) 
    are visible and not collapsed onto the prediction line.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.title("💰 Cloud Cost Forecast")
    st.info("Forecasting **total monthly cost** for the next 12 months...")

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

        # --- TRAIN THE MODEL (WITH FIX) ---
        st.markdown(
            "***Model Tuning Note:*** *The `changepoint_prior_scale` was set to `0.005` to smooth out the trend "
            "and prevent the large December cost reduction from causing an unrealistic negative forecast. The `interval_width` "
            "is set to `0.95` to display the $95\%$ confidence bounds.*"
        )
        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
            # CRITICAL FIX 1: Reduce sensitivity to late-stage trend changes
            changepoint_prior_scale=0.005,
            # CRITICAL FIX 2: Set explicit uncertainty interval width for visible bounds
            interval_width=0.95
        )
        model.fit(prophet_df)

        # --- FUTURE DATES ---
        future = model.make_future_dataframe(periods=12, freq='M')
        forecast = model.predict(future)

        st.subheader("Monthly Cost Forecast (Next 1 Year)")

        # Plot using Plotly (works in Streamlit)
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
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

        st.success("Monthly forecast analysis complete. The trend and uncertainty bounds should now be stable!")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.warning("This error usually means the data format is inconsistent or Prophet ran into numerical issues.")
