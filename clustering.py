# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import streamlit as st
# import plotly.express as px
# import numpy as np

# def run_clustering_analysis(df):
#     """
#     Performs K-Means clustering on Cost, Utilization, and Quantity to segment workloads.
#     """

#     st.subheader("📊 Workload Segmentation using K-Means Clustering")
#     st.markdown(
#         "Clustering helps group resources with similar spending and usage patterns. "
#         "We cluster based on **Total Cost**, **Usage Quantity**, and **Average Utilization**."
#     )

#     # Select features for clustering
#     features = ['Unrounded Cost ($)', 'Usage Quantity', 'Avg Utilization (%)']

#     # Filter out rows with zero cost or usage quantity, as they skew the data
#     # Also, ensure we drop any rows that resulted in NaN/Inf in previous steps (like anomaly detection)
#     cluster_df = df[(df['Unrounded Cost ($)'] > 0) & (df['Usage Quantity'] > 0)].copy()

#     # Clean data before scaling
#     for col in features:
#         if col in cluster_df.columns:
#             cluster_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
#     cluster_df.dropna(subset=features, inplace=True)


#     if cluster_df.shape[0] < 5:
#         st.warning("Not enough data points with non-zero cost/usage to perform meaningful clustering.")
#         return

#     X = cluster_df[features]

#     # 1. Scale Data
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # 2. Perform K-Means Clustering
#     K = 4
#     kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
#     cluster_df['Cluster_ID'] = kmeans.fit_predict(X_scaled)


#     # 3. Assign Descriptive Cluster Names (The requested update!)

#     # Calculate mean values for each cluster to profile them
#     profile = cluster_df.groupby('Cluster_ID')[features].mean()

#     # We want a high score for optimization candidates (High Cost, Low Util)
#     # Calculate a simple "Optimization Index": Cost Rank - Utilization Rank

#     # Rank by Cost (descending)
#     profile['Cost_Rank'] = profile['Unrounded Cost ($)'].rank(ascending=False)

#     # Rank by Utilization (ascending, since low util is bad)
#     profile['Util_Rank'] = profile['Avg Utilization (%)'].rank(ascending=True)

#     # Optimization Score: Higher score = more wasteful/higher priority
#     profile['Optimization_Score'] = profile['Cost_Rank'] + profile['Util_Rank']

#     # Sort by Optimization Score
#     profile = profile.sort_values('Optimization_Score', ascending=False).reset_index()

#     # Define names based on the score ranking
#     cluster_names = {
#         profile.loc[0, 'Cluster_ID']: "**1. High Priority Waste (Rightsizing Target)**", # Highest Score
#         profile.loc[1, 'Cluster_ID']: "2. Moderate Efficiency Concern",
#         profile.loc[2, 'Cluster_ID']: "3. Stable/Efficient Workload",
#         profile.loc[3, 'Cluster_ID']: "4. Low Cost/Low Priority",  # Lowest Score
#     }

#     # Map the descriptive names back to the main DataFrame
#     cluster_df['Cluster'] = cluster_df['Cluster_ID'].map(cluster_names)

#     st.success(f"Clustering complete with K={K} workload segments.")

#     # 4. Visualization
#     # Define a target column for the bubble size (e.g., Total Cost)
#     cluster_df['Size_Metric'] = cluster_df['Unrounded Cost ($)'] / cluster_df['Unrounded Cost ($)'].max() * 50

#     fig = px.scatter_3d(
#         cluster_df,
#         x='Avg Utilization (%)',
#         y='Unrounded Cost ($)',
#         z='Usage Quantity',
#         color='Cluster',
#         size='Size_Metric',
#         hover_data=['Resource ID', 'Service Name'],
#         title="3D Visualization of Workload Clusters"
#     )
#     fig.update_layout(height=700)
#     st.plotly_chart(fig, use_container_width=True)

#     # 5. Cluster Summaries
#     st.subheader("Cluster Profiles")
#     cluster_summary = cluster_df.groupby('Cluster')[features].mean()
#     cluster_summary['Resource Count'] = cluster_df['Cluster'].value_counts()

#     # Clean up the output dataframe
#     cluster_summary = cluster_summary.sort_values('Unrounded Cost ($)', ascending=False).round(2)
#     cluster_summary.index.name = "Workload Segment"

#     st.dataframe(cluster_summary)

#     st.markdown(
#         "**Actionable Insights:** Focus your efforts on the **High Priority Waste** segment. Resources in these clusters "
#         "have the highest combined cost and lowest utilization, making them prime candidates for immediate rightsizing or termination."
#     )

#     # Add a quick peek at the most wasteful resources
#     wasteful_cluster_name = cluster_names[profile.loc[0, 'Cluster_ID']]
#     wasteful_resources = cluster_df[cluster_df['Cluster'] == wasteful_cluster_name].sort_values(
#         'Unrounded Cost ($)', ascending=False
#     ).head(10)

#     if not wasteful_resources.empty:
#         with st.expander(f"View Top 10 Resources in {wasteful_cluster_name}"):
#             st.dataframe(wasteful_resources[['Resource ID', 'Service Name', 'Unrounded Cost ($)', 'Avg Utilization (%)']].round(2))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import streamlit as st
import plotly.express as px
import numpy as np

def run_clustering_analysis(df):
    """
    Performs K-Means clustering on Cost, Utilization, and Quantity to segment workloads.
    High Priority Waste is dynamically assigned based on high cost + low utilization.
    """
    st.subheader("📊 Workload Segmentation using K-Means Clustering")
    st.markdown(
        "Clustering helps group resources with similar spending and usage patterns. "
        "We cluster based on **Total Cost**, **Usage Quantity**, and **Average Utilization**."
    )

    features = ['Unrounded Cost ($)', 'Usage Quantity', 'Avg Utilization (%)']

    # Filter out rows with zero cost or usage quantity
    cluster_df = df[(df['Unrounded Cost ($)'] > 0) & (df['Usage Quantity'] > 0)].copy()

    # Clean data: replace inf with NaN and drop NaNs
    for col in features:
        if col in cluster_df.columns:
            cluster_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    cluster_df.dropna(subset=features, inplace=True)

    if cluster_df.shape[0] < 5:
        st.warning("Not enough data points with non-zero cost/usage to perform meaningful clustering.")
        return

    X = cluster_df[features]

    # 1. Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Perform K-Means Clustering
    K = 4
    kmeans = KMeans(n_clusters=K, random_state=42, n_init='auto')
    cluster_df['Cluster_ID'] = kmeans.fit_predict(X_scaled)

    # 3. Profile Clusters
    profile = cluster_df.groupby('Cluster_ID')[features].mean()

    # Compute optimization score: high cost + low utilization
    profile['Cost_Rank'] = profile['Unrounded Cost ($)'].rank(ascending=False)
    profile['Util_Rank'] = profile['Avg Utilization (%)'].rank(ascending=True)
    profile['Optimization_Score'] = profile['Cost_Rank'] + profile['Util_Rank']

    # Sort clusters by Optimization Score
    profile = profile.sort_values('Optimization_Score').reset_index()

    # Assign meaningful cluster names
    cluster_names = {
        profile.loc[0, 'Cluster_ID']: "**1. High Priority Waste (Rightsizing Target)**",
        profile.loc[1, 'Cluster_ID']: "**2. Moderate Efficiency Concern**",
        profile.loc[2, 'Cluster_ID']: "**3. Stable/Efficient Workload**",
        profile.loc[3, 'Cluster_ID']: "**4. Low Cost/Low Priority**",
    }

    cluster_df['Cluster'] = cluster_df['Cluster_ID'].map(cluster_names)

    # Optional: Resource-level Waste Score
    cluster_df['Waste_Score'] = (cluster_df['Unrounded Cost ($)'] / cluster_df['Unrounded Cost ($)'].max()) \
                                 * (1 - cluster_df['Avg Utilization (%)'] / 100)

    st.success(f"Clustering complete with K={K} workload segments.")

    # 4. Visualization
    cluster_df['Size_Metric'] = cluster_df['Unrounded Cost ($)'] / cluster_df['Unrounded Cost ($)'].max() * 50

    fig = px.scatter_3d(
        cluster_df,
        x='Avg Utilization (%)',
        y='Unrounded Cost ($)',
        z='Usage Quantity',
        color='Cluster',
        size='Size_Metric',
        hover_data=['Resource ID', 'Service Name', 'Waste_Score'],
        title="3D Visualization of Workload Clusters"
    )
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    # 5. Cluster Summaries
    st.subheader("Cluster Profiles")
    cluster_summary = cluster_df.groupby('Cluster')[features].mean()
    cluster_summary['Resource Count'] = cluster_df['Cluster'].value_counts()
    cluster_summary = cluster_summary.sort_values('Unrounded Cost ($)', ascending=False).round(2)
    cluster_summary.index.name = "Workload Segment"
    st.dataframe(cluster_summary)

    st.markdown(
        "**Actionable Insights:** Focus your efforts on the **High Priority Waste** segment. "
        "Resources in this cluster have the highest combined cost and lowest utilization, "
        "making them prime candidates for immediate rightsizing or termination."
    )

    # 6. Top wasteful resources
    wasteful_cluster_name = cluster_names[profile.loc[0, 'Cluster_ID']]
    wasteful_resources = cluster_df[cluster_df['Cluster'] == wasteful_cluster_name].sort_values(
        'Waste_Score', ascending=False
    ).head(10)

    if not wasteful_resources.empty:
        with st.expander(f"View Top 10 Resources in {wasteful_cluster_name}"):
            st.dataframe(
                wasteful_resources[['Resource ID', 'Service Name', 'Unrounded Cost ($)', 'Avg Utilization (%)', 'Waste_Score']].round(2)
            )
