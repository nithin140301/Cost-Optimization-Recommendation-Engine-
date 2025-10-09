import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

from anomaly_detection import flag_anomalous_records  # Isolation Forest cleaning

# Check Prophet availability
try:
    _ = Prophet()
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def run_cost_forecasting(df: pd.DataFrame):
    """
    Simplified cost forecasting using Prophet.
    Anomaly removal is automatic; no model tuning or validation upload.
    """
    if not PROPHET_AVAILABLE:
        st.warning("Cost forecasting requires the 'prophet' library.")
        st.markdown("Install it: `pip install prophet`")
        return

    st.title("💰 Cloud Cost Forecast (Isolation Forest Cleaned)")
    st.markdown("Forecast uses historical costs cleaned via Isolation Forest and aggregated monthly.")

    required_cols = ['Usage Start Date', 'Rounded Cost ($)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing required columns: {', '.join(required_cols)}")
        return

    df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'])

    if df.empty:
        st.error("Input historical data is empty.")
        return

    min_date = df['Usage Start Date'].min().strftime('%Y-%m-%d')
    max_date = df['Usage Start Date'].max().strftime('%Y-%m-%d')
    st.sidebar.markdown(f"**Historical Range:** {min_date} to {max_date}")

    # --- Prediction horizon input ---
    default_start = (df['Usage Start Date'].max() + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    prediction_start_date = st.sidebar.text_input(
        "Prediction Start Date (YYYY-MM-DD)",
        value=default_start
    )
    prediction_length_months = st.sidebar.number_input(
        "Prediction Length (Months)",
        min_value=1,
        max_value=60,
        value=12,
        step=1
    )

    st.info(f"Forecasting {prediction_length_months} months starting {prediction_start_date}")

    try:
        # --- Anomaly Detection ---
        df_flagged = flag_anomalous_records(df)
        df_clean = df_flagged[df_flagged['is_anomaly'] != -1].copy()

        if df_clean.empty:
            st.error("No data remains after anomaly removal.")
            return

        # --- Aggregate to monthly ---
        prophet_df = df_clean.copy()
        prophet_df['ds'] = prophet_df['Usage Start Date'].dt.to_period('M').dt.start_time
        prophet_df['y'] = prophet_df['Rounded Cost ($)']
        prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()

        if len(prophet_df) < 2:
            st.error("Not enough monthly data to forecast.")
            return

        st.success(f"Using {len(prophet_df)} monthly observations for model training.")

        # --- Train Prophet ---
        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)

        # --- Prepare future dataframe ---
        prediction_start_dt = pd.to_datetime(prediction_start_date)
        last_historical = prophet_df['ds'].max()
        prediction_end = prediction_start_dt + pd.DateOffset(months=prediction_length_months)
        total_periods = (prediction_end.to_period('M') - last_historical.to_period('M')).n

        if total_periods <= 0:
            st.error("Prediction start date must be after last historical date.")
            return

        future = model.make_future_dataframe(periods=total_periods, freq='M')
        forecast = model.predict(future)

        # Filter for requested forecast window
        forecast_period = forecast[
            (forecast['ds'] >= prediction_start_dt) &
            (forecast['ds'] < prediction_end)
        ].copy()
        forecast_period['ds'] = forecast_period['ds'].dt.to_period('M').dt.start_time

        # --- Display results ---
        st.subheader("Forecast Results")
        display_df = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
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

        # --- Visualization ---
        st.subheader("Forecast Visuals")
        st.write("Black dots = historical costs, blue line = forecast")
        fig1 = plot_plotly(model, forecast)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Forecast Components")
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error during forecasting: {e}")
