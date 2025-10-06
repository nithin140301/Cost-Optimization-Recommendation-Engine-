import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly


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

        # --- TRAIN THE MODEL ---
        model = Prophet(
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True
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
        st.metric(
            label="Predicted Cost 12 Months Out (Monthly Total)",
            value=f"${last_month_cost:,.2f}",
            delta="Based on trends, prepare for this monthly total cost."
        )

        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(13))

        st.success("Monthly forecast analysis complete.")

    except Exception as e:
        st.error(f"An error occurred during forecasting: {e}")
        st.warning("This error usually means the data format is inconsistent or Prophet ran into numerical issues.")
