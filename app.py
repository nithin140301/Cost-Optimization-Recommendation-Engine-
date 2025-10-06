import forecast_app
import streamlit as st
import pandas as pd
# IMPORTANT: Import initialize_session_state to ensure state exists on startup
from auth import login_page, show_logout_button, initialize_session_state
from data_processor import  process_uploaded_data
from prophet import Prophet
from prophet.plot import plot_plotly


from recommendations import generate_recommendations
from anomaly_detection import run_anomaly_detection
from clustering import run_clustering_analysis
from forecast_app import run_cost_forecasting

# --- Page Configuration ---
st.set_page_config(
    page_title="Cloud Cost Optimization Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_main_app():
    """Renders the main application interface after successful login and data upload."""

    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your cloud usage CSV file (data2.csv)",
        type=["csv"]
    )

    if uploaded_file is not None:
        # Check if data is already processed in session state
        if 'df' not in st.session_state or st.session_state.get('df_file_name') != uploaded_file.name:
            st.info("Processing data... Please wait.")
            df = process_uploaded_data(uploaded_file)

            if df is not None:
                st.session_state['df'] = df
                st.session_state['df_file_name'] = uploaded_file.name
                st.success("Data loaded and processed successfully!")
            else:
                st.error("Failed to process the uploaded file. Check the file format.")
                # We stop the execution here if data processing fails
                st.stop()

        df = st.session_state['df']

        st.header("Cloud Cost Optimization Recommendation Engine")

        # --- Tabbed Interface ---
        tab_recommend, tab_forecast, tab_anomaly, tab_cluster, tab_data = st.tabs([
            "💰 Recommendations",
            "📈 Future Forecast",
            "🚨 Anomaly Detection",
            "📊 Workload Clustering",
            "🗂️ Raw Data & Metrics"
        ])

        with tab_recommend:
            st.title("Optimization Recommendations")
            st.markdown("Actionable insights for rightsizing, termination, and pricing model changes.")
            # Ensure df is passed to all analysis functions
            generate_recommendations(df)

        with tab_forecast:
            st.title("Cost Forecasting")
            st.markdown("Predicting future spending to help set budgets and prevent bill shock.")
            run_cost_forecasting(df)

        with tab_anomaly:
            st.title("Cost Anomaly Detection")
            st.markdown("Identifying unusual cost spikes that require immediate investigation.")
            run_anomaly_detection(df)

        with tab_cluster:
            st.title("Workload Clustering")
            st.markdown("Segmenting resources based on cost and usage patterns for targeted optimization strategies.")
            run_clustering_analysis(df)

        with tab_data:
            st.title("Processed Data Overview")
            st.markdown("Review the raw data along with the calculated optimization metrics.")
            st.dataframe(df)

    else:
        # Prompt user to upload data
        st.info("Please upload your cloud usage CSV file (`data2.csv`) in the sidebar to begin analysis.")
        # Only pop 'df' if no file is uploaded to avoid unnecessary processing after a file is removed
        st.session_state.pop('df', None)

# --- Main Logic Flow ---
def main():
    # 1. Ensure all session state variables are initialized before use.
    # We keep the initializer for consistency, but rely on .get() below for safety.
    initialize_session_state()

    # 2. Display logout button (relies on initialized state)
    show_logout_button()

    # 3. Check the authenticated status using .get() with a default value (False).
    # This prevents the KeyError if the key is missing on the first run.
    if st.session_state.get('logged_in', False):
        render_main_app()
    else:
        login_page()

if __name__ == '__main__':
    main()
