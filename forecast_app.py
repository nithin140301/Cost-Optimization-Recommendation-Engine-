import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly


st.title("💰 Cloud Cost Forecast (Dynamic CSV Upload)")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your historical cost CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # --- Check required columns ---
        required_cols = ['Usage Start Date', 'Rounded Cost ($)']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must have columns: {', '.join(required_cols)}")
        else:
            df['Usage Start Date'] = pd.to_datetime(df['Usage Start Date'])

            if df.empty:
                st.error("Uploaded CSV is empty.")
            else:
                # --- Prepare data for Prophet ---
                prophet_df = df[['Usage Start Date', 'Rounded Cost ($)']].copy()
                prophet_df.rename(
                    columns={'Usage Start Date': 'ds', 'Rounded Cost ($)': 'y'}, 
                    inplace=True
                )

                # --- Aggregate monthly ---
                prophet_df['ds'] = prophet_df['ds'].dt.to_period('M').dt.start_time
                prophet_df = prophet_df.groupby('ds')['y'].sum().reset_index()

                if len(prophet_df) < 2:
                    st.error("Not enough monthly data to forecast.")
                else:
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
                    fig2 = model.plot_components(prediction)
                    st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing uploaded CSV: {e}")
