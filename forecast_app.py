import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric

def run_cost_forecasting(df: pd.DataFrame):
    """
    Forecast future costs using Prophet, including cross-validation and performance metrics.
    """
    st.subheader("Cost Forecasting (Next 12 Months)")

    required_cols = ['Usage Start Date', 'Rounded Cost ($)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"The dataset must contain the following columns: {', '.join(required_cols)}")
        return

    # --- Prepare data ---
    prophet_df = df[['Usage Start Date', 'Rounded Cost ($)']].copy()
    prophet_df.rename(columns={'Usage Start Date': 'ds', 'Rounded Cost ($)': 'y'}, inplace=True)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    prophet_df['ds'] = prophet_df['ds'].dt.to_period('M').dt.start_time
    prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()

    if len(prophet_df) < 2:
        st.error("Not enough monthly data to forecast.")
        return

    # --- Prophet Model ---
    model = Prophet()
    model.fit(prophet_df)

    # --- Forecast next 12 months ---
    future_dates = model.make_future_dataframe(periods=12, freq='MS')
    prediction = model.predict(future_dates)

    last_historical = prophet_df['ds'].max()
    forecast_period = prediction[prediction['ds'] > last_historical].copy()
    forecast_period['ds'] = forecast_period['ds'].dt.to_period('M').dt.start_time

    # --- Display forecast table ---
    st.subheader("Forecast Results (Next 12 Months)")
    display_df = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    display_df.rename(
        columns={'ds': 'Date', 'yhat': 'Predicted Cost',
                 'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'},
        inplace=True
    )
    st.dataframe(
        display_df.style.format(
            {'Predicted Cost': '${:,.2f}',
             'Lower Bound': '${:,.2f}',
             'Upper Bound': '${:,.2f}'}
        ),
        use_container_width=True
    )

    # --- Forecast Plots ---
    st.subheader("Forecast Plot")
    st.plotly_chart(plot_plotly(model, prediction), use_container_width=True)

    st.subheader("Forecast Components")
    st.pyplot(model.plot_components(prediction))

    # --- Prophet Cross-Validation ---
    st.subheader("Cross-Validation & Performance Metrics")
    st.markdown("This performs time-series cross-validation to evaluate forecast accuracy.")

   # --- Prophet Cross-Validation ---
st.subheader("Cross-Validation & Performance Metrics")
st.markdown("This performs time-series cross-validation to evaluate forecast accuracy.")

try:
    # Run cross-validation
    df_cv = cross_validation(model, initial='180 days', period='90 days', horizon='180 days', parallel="processes")
    
    st.markdown("**Full Cross-Validation Table:**")
    st.dataframe(df_cv, use_container_width=True)  # Display full table

    # Performance metrics
    df_p = performance_metrics(df_cv)
    st.markdown("**Performance Metrics:**")
    st.dataframe(df_p, use_container_width=True)

    # Plot RMSE over horizon
    st.markdown("**Cross-Validation Metric Plot (RMSE over horizon):**")
    fig = plot_cross_validation_metric(df_cv, metric='rmse')
    st.pyplot(fig)

except Exception as e:
    st.warning(f"Cross-validation could not be performed: {e}")
    st.info("Cross-validation requires sufficient historical data (at least ~6 months).")
