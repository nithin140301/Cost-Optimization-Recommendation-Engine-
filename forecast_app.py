import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def run_cost_forecasting(df: pd.DataFrame):
    """
    Forecast future costs using Prophet.
    Automatically picks 'Usage Start Date' and 'Rounded Cost ($)' columns from the uploaded dataframe.
    """

    st.subheader("Cost Forecasting (Next 12 Months)")

    # --- Check required columns exist ---
    required_cols = ['Usage Start Date', 'Rounded Cost ($)']
    if not all(col in df.columns for col in required_cols):
        st.error(f"The dataset must contain the following columns: {', '.join(required_cols)}")
        return

    # --- Extract relevant columns and rename ---
    prophet_df = df[['Usage Start Date', 'Rounded Cost ($)']].copy()
    prophet_df.rename(columns={'Usage Start Date': 'ds', 'Rounded Cost ($)': 'y'}, inplace=True)
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    # --- Aggregate monthly ---
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

    # --- Filter only future months ---
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

    # --- Forecast Plot ---
    st.subheader("Forecast Plot")
    fig1 = plot_plotly(model, prediction)
    st.plotly_chart(fig1, use_container_width=True)

    # --- Forecast Components ---
    st.subheader("Forecast Components")
    st.pyplot(model.plot_components(prediction))
