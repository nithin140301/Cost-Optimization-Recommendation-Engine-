import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly


def run_cost_forecasting(df):
    st.info("Forecasting **total monthly cost** for the next 12 months...")
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=12, freq='M')
    forecast = m.predict(future)
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)

# Attempt to import Prophet, which is a complex external dependency
try:
    from forecast_app import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.error("Prophet library not found. Please install it using 'pip install prophet' to enable forecasting.")
except Exception as e:
    PROPHET_AVAILABLE = False
    st.error(f"Error loading Prophet: {e}. Forecasting tab disabled.")


def run_cost_forecasting(df: pd.DataFrame):
    """
    Performs cost forecasting using Facebook Prophet, provided the library is available,
    predicting total monthly cost for the next 12 months.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library, which is not currently installed or failed to load.")
        st.markdown("Please install it to use this feature: `pip install prophet`")
        return

    st.info("Forecasting **total monthly cost** for the next 12 months...")

    try:
        # --- MONTHLY DATA PREPARATION ---
        # 1. Prepare data for Prophet: requires columns 'ds' (datestamp) and 'y' (value)
        # We aggregate the total cost by the month of the 'Usage Start Date'.
        df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'])

        # Set 'ds' to the first day of the month for each usage record
        prophet_df = df.copy()
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']

        # Group and sum the total monthly costs
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()
        prophet_df.columns = ['ds', 'y']

        # Ensure we have enough data (at least 2 months) for Prophet
        if len(prophet_df) < 2:
            st.error("Not enough historical monthly data (requires at least 2 months) to perform a reliable forecast.")
            return

        # 2. Train the Prophet Model
        model = Prophet(
            # Adjust seasonality for monthly data
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True # Relevant for monthly forecasts
        )
        model.fit(prophet_df)

        # 3. Create future dates for prediction (next 12 months, using month-end frequency 'M')
        future = model.make_future_dataframe(periods=12, freq='M')

        # 4. Predict the cost
        forecast = model.predict(future)

        st.subheader("Monthly Cost Forecast (Next 1 Year)")

        # Plotting the forecast
        fig1 = model.plot(forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Component Breakdown")
        fig2 = model.plot_components(forecast)
        st.plotly_chart(fig2, use_container_width=True)

        # Display key figures
        last_month_cost = forecast.iloc[-1]['yhat']
        st.metric(
            label="Predicted Cost 12 Months Out (Monthly Total)",
            value=f"${last_month_cost:,.2f}",
            delta=f"Based on trends, prepare for this monthly total cost."
        )

        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13))

        st.success("Monthly forecast analysis complete.")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.warning("This error usually means the data format is inconsistent or Prophet ran into numerical issues.")

# --- End of run_cost_forecasting function ---