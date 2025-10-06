import streamlit as st
import pandas as pd
import plotly.express as px

def generate_recommendations(df: pd.DataFrame):
    """
    Generates and displays actionable cost optimization recommendations based on
    rightsizing, termination, and service cost analysis.
    """
    if df.empty:
        st.warning("No valid data available after processing. Please check your uploaded file.")
        return

    try:
        # --- 1. Rightsizing & Termination Candidates (Using Optimization Flag) ---

        candidates = df[df['Optimization Flag'] == 'Rightsizing/Termination Candidate'].copy()

        st.subheader("1. Rightsizing and Termination Opportunities")
        st.markdown(
            """
            Resources flagged below have high duration (over 100 hours) but extremely low average utilization 
            (CPU and Memory combined are less than 10%). These are immediate savings opportunities.
            """
        )

        if not candidates.empty:
            # Group candidates by Service and Region to see total savings potential
            savings_summary = candidates.groupby(['Service Name', 'Region / Zone']).agg(
                potential_monthly_savings=('Monthly Cost Estimate ($)', 'sum'),
                resource_count=('Resource ID', 'count')
            ).reset_index().sort_values(by='potential_monthly_savings', ascending=False)

            total_potential_savings = savings_summary['potential_monthly_savings'].sum()

            st.metric(
                label="Total Estimated Monthly Rightsizing Savings",
                value=f"${total_potential_savings:,.2f}",
                delta="Focus on these to cut waste immediately."
            )

            # Display top candidates visually
            fig_savings = px.bar(
                savings_summary.head(10),
                x='Service Name',
                y='potential_monthly_savings',
                color='Region / Zone',
                title="Top 10 Services by Potential Monthly Savings",
                labels={'potential_monthly_savings': 'Estimated Savings ($)', 'Service Name': 'Cloud Service'}
            )
            fig_savings.update_layout(xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig_savings, use_container_width=True)

            with st.expander(f"View Details for {len(candidates)} Rightsizing Candidates"):
                # Display the most expensive candidates first
                st.dataframe(
                    candidates[[
                        'Resource ID', 'Service Name', 'Region / Zone', 'Avg Utilization (%)',
                        'Duration (Hours)', 'Monthly Cost Estimate ($)', 'Rounded Cost ($)'
                    ]].sort_values(by='Monthly Cost Estimate ($)', ascending=False)
                )
        else:
            st.success("🎉 No significant rightsizing or termination candidates found based on current usage and low utilization thresholds.")

        st.markdown("---")

        # --- 2. High-Cost Service Analysis ---
        st.subheader("2. Identify Most Expensive Services")

        # Aggregate total cost by service
        cost_by_service = df.groupby('Service Name')['Rounded Cost ($)'].sum().reset_index()
        cost_by_service = cost_by_service.sort_values(by='Rounded Cost ($)', ascending=False)

        st.info("High-cost services should be reviewed for potential pricing model changes (e.g., Reserved Instances).")

        fig_cost = px.pie(
            cost_by_service.head(5),
            values='Rounded Cost ($)',
            names='Service Name',
            title='Top 5 Services by Total Cost',
            hole=0.4
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        # Recommendation based on high usage duration
        long_running_resources = df[
            (df['Duration (Hours)'] > (30 * 24)) & # Running longer than a month
            (df['Service Name'].isin(cost_by_service.head(5)['Service Name']))
            ].copy()

        if not long_running_resources.empty:
            st.markdown(
                f"""
                **Pricing Model Recommendation:** We identified **{len(long_running_resources)}** long-running resources 
                within your top 5 most expensive services.
                
                For these stable workloads, you should investigate converting from **On-Demand** to 
                **Reserved Instances (RIs)** or **Committed Use Discounts (CUDs)** to save 30-70%.
                """
            )

        st.markdown("---")

        # --- 3. Regional Cost Comparison ---
        st.subheader("3. Regional Cost Arbitrage (Cheapest Locations)")

        # Calculate average cost per hour by service and region
        df['Cost per Hour ($)'] = df['Rounded Cost ($)'] / df['Duration (Hours)']

        regional_cost = df.groupby(['Service Name', 'Region / Zone'])['Cost per Hour ($)'].mean().reset_index()

        # Only analyze services present in multiple regions
        multi_region_services = regional_cost['Service Name'].value_counts()
        multi_region_services = multi_region_services[multi_region_services > 1].index

        if not multi_region_services.empty:
            st.info("The table below shows which regions offer the same service for less, suggesting potential cost migration.")

            comparison_list = []
            for service in multi_region_services:
                service_data = regional_cost[regional_cost['Service Name'] == service].sort_values('Cost per Hour ($)')
                cheapest_region = service_data.iloc[0]
                most_expensive_region = service_data.iloc[-1]

                # Calculate the percentage difference
                cost_diff = most_expensive_region['Cost per Hour ($)'] - cheapest_region['Cost per Hour ($)']
                if most_expensive_region['Cost per Hour ($)'] > 0:
                    savings_potential = (cost_diff / most_expensive_region['Cost per Hour ($)']) * 100
                else:
                    savings_potential = 0

                comparison_list.append({
                    'Service Name': service,
                    'Cheapest Region': cheapest_region['Region / Zone'],
                    'Cheapest Cost/Hr ($)': f"{cheapest_region['Cost per Hour ($)']:.4f}",
                    'Expensive Region': most_expensive_region['Region / Zone'],
                    'Expensive Cost/Hr ($)': f"{most_expensive_region['Cost per Hour ($)']:.4f}",
                    'Max Savings Potential (%)': f"{savings_potential:.1f}%"
                })

            st.dataframe(pd.DataFrame(comparison_list).sort_values('Max Savings Potential (%)', ascending=False))

        else:
            st.info("Not enough services spanning multiple regions to perform regional cost arbitrage analysis.")

    except KeyError as e:
        st.error(f"Dataframe error in recommendations: Required column {e} not found. Ensure 'data_processor.py' runs correctly.")
    except Exception as e:
        st.error(f"An unexpected error occurred during recommendation generation: {e}")
