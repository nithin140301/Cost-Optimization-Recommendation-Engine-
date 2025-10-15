# import streamlit as st
# import pandas as pd
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from prophet.diagnostics import cross_validation, performance_metrics
# from prophet.plot import plot_cross_validation_metric

# def run_cost_forecasting(df: pd.DataFrame):
#     """
#     Forecast future costs using Prophet, including cross-validation and performance metrics.
#     """
#     st.subheader("Cost Forecasting (Next 12 Months)")

#     required_cols = ['Usage Start Date', 'Rounded Cost ($)']
#     if not all(col in df.columns for col in required_cols):
#         st.error(f"The dataset must contain the following columns: {', '.join(required_cols)}")
#         return

#     # --- Prepare data ---
#     prophet_df = df[['Usage Start Date', 'Rounded Cost ($)']].copy()
#     prophet_df.rename(columns={'Usage Start Date': 'ds', 'Rounded Cost ($)': 'y'}, inplace=True)
#     prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

#     prophet_df['ds'] = prophet_df['ds'].dt.to_period('M').dt.start_time
#     prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()

#     if len(prophet_df) < 2:
#         st.error("Not enough monthly data to forecast.")
#         return

#     # --- Prophet Model ---
#     model = Prophet()
#     model.fit(prophet_df)

#     # --- Forecast next 12 months ---
#     future_dates = model.make_future_dataframe(periods=12, freq='MS')
#     prediction = model.predict(future_dates)

#     last_historical = prophet_df['ds'].max()
#     forecast_period = prediction[prediction['ds'] > last_historical].copy()
#     forecast_period['ds'] = forecast_period['ds'].dt.to_period('M').dt.start_time

#     # --- Display forecast table ---
#     st.subheader("Forecast Results (Next 12 Months)")
#     display_df = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
#     display_df.rename(
#         columns={'ds': 'Date', 'yhat': 'Predicted Cost',
#                  'yhat_lower': 'Lower Bound', 'yhat_upper': 'Upper Bound'},
#         inplace=True
#     )
#     st.dataframe(
#         display_df.style.format(
#             {'Predicted Cost': '${:,.2f}',
#              'Lower Bound': '${:,.2f}',
#              'Upper Bound': '${:,.2f}'}
#         ),
#         use_container_width=True
#     )

#     # --- Forecast Plots ---
#     st.subheader("Forecast Plot")
#     st.plotly_chart(plot_plotly(model, prediction), use_container_width=True)

#     st.subheader("Forecast Components")
#     st.pyplot(model.plot_components(prediction))

#     # --- Prophet Cross-Validation ---
#     st.subheader("Cross-Validation & Performance Metrics")
#     st.markdown("This performs time-series cross-validation to evaluate forecast accuracy.")

#     try:
#         # Run cross-validation
#         df_cv = cross_validation(model, initial='180 days', period='90 days', horizon='180 days', parallel="processes")
#         st.markdown("**Cross-validation sample:**")
#         st.dataframe(df_cv)

#         # Performance metrics
#         df_p = performance_metrics(df_cv)
#         st.markdown("**Performance metrics:**")
#         st.dataframe(df_p)

#         # Plot RMSE over horizon
#         st.markdown("**Cross-validation metric plot (RMSE over horizon):**")
#         fig = plot_cross_validation_metric(df_cv, metric='rmse')
#         st.pyplot(fig)

#     except Exception as e:
#         st.warning(f"Cross-validation could not be performed: {e}")
#         st.info("Cross-validation requires sufficient historical data (at least ~6 months).")


import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt # Import necessary for plot_components

def run_cost_forecasting(df: pd.DataFrame):
    """
    Forecast future costs using Prophet, including a single-run cross-validation test.
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

    # Group by month and sum cost
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
    # Ensure the plot is drawn to the streamlit-managed pyplot figure
    fig_components = model.plot_components(prediction)
    st.pyplot(fig_components)
    plt.close(fig_components) # Close the figure to free memory

    # --- Prophet Cross-Validation (Modified for a "Single Run") ---
    st.subheader("Single-Run Performance Metrics")
    st.markdown("This performs a **single cross-validation fold** (test run) by forcing only one cutoff point to evaluate forecast accuracy.")

    try:
        # **MODIFICATION FOR "SINGLE RUN":**
        # Set 'period' to a very large number of days (e.g., 10000 days or ~27 years).
        # This ensures the cross_validation function finds only ONE cutoff,
        # effectively giving one test run for the given 'horizon'.
        df_cv = cross_validation(
            model, 
            initial='180 days',    # Train on the first 6 months of data
            period='10000 days',   # Set a period much larger than the data to force only one test fold
            horizon='180 days',    # Test the next 6 months of data
            parallel="processes"
        )
        st.markdown("**Cross-validation sample (Showing only one test run):**")
        st.dataframe(df_cv)

        # Performance metrics
        # Calculate the overall performance for this single test fold
        df_p = performance_metrics(df_cv)
        st.markdown("**Overall Performance Metrics (Averaged over the single run):**")
        st.dataframe(df_p)

        # Plot RMSE over horizon
        st.markdown("**Cross-validation metric plot (RMSE over horizon):**")
        fig_rmse = plot_cross_validation_metric(df_cv, metric='rmse')
        st.pyplot(fig_rmse)
        plt.close(fig_rmse) # Close the figure to free memory

    except Exception as e:
        st.warning(f"Cross-validation could not be performed: {e}")
        st.info("Cross-validation requires sufficient historical data (at least ~6 months).")

# You would call run_cost_forecasting(df) after loading your data.
