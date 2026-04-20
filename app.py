import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np


# Page config
st.set_page_config(page_title="Crime Analysis Dashboard", layout="wide")


@st.cache_data
def load_data():
    try:
        # Try local first (for fast dev)
        df = pd.read_csv("cleaned_crime_data.csv")
    except:
        # Fallback to Google Drive
        url = "https://drive.google.com/uc?id=1uEJlp6R7IEzoRLPLpL2CyGn9iDjEnsL3"
        df = pd.read_csv(url)
    return df


@st.cache_resource
def load_models():
    models = {}

    try:
        models["geo_kmeans"] = joblib.load("models/geo_kmeans.pkl")
        models["temporal_kmeans"] = joblib.load("models/temporal_kmeans.pkl")
        models["geo_scaler"] = joblib.load("geo_scaler.pkl")
        models["temporal_scaler"] = joblib.load("temporal_scaler.pkl")
        models["pca_model"] = joblib.load("pca_model.pkl")
        models["pca_scaler"] = joblib.load("pca_scaler.pkl")

    except Exception as e:
        st.error(f"Error loading models: {e}")

    return models


df = load_data()
models = load_models()


# Sidebar navigation
st.sidebar.title("🧭Navigation")

page = st.sidebar.radio('Choose below⬇️',
    ["🏠Home", "🌍Geo Analysis", "⏱️Temporal Analysis", "📈PCA Insights"]
)

# ---------------- HOME ----------------
if page == "🏠Home":

    st.title("🚓 Crime Pattern Analysis Dashboard")

    st.markdown("### 📊 Turning Data into Actionable Insights")

    st.markdown("""
    This intelligent dashboard helps analyze and understand crime patterns using advanced machine learning techniques.
    
    It enables law enforcement agencies to:
    - Identify crime hotspots  
    - Understand temporal trends  
    - Optimize patrol strategies  
    - Improve response efficiency  
    """)

    st.divider()

    # ============================
    # 🔥 Key Modules (Cards Style)
    # ============================
    st.subheader("🔍 Key Analysis Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🌍 Geo Analysis
        - Clustering using KMeans  
        - Identify high-risk locations  
        - District & beat-level insights  
        """)

        st.markdown("""
        ### ⏱ Temporal Analysis
        - Crime trends by hour & month  
        - Peak crime detection  
        - Smart patrol planning  
        """)

    with col2:
        st.markdown("""
        ### 📉 PCA Insights
        - Feature importance analysis  
        - Dimensionality reduction  
        - Pattern discovery  
        """)

        st.markdown("""
        ### 🤖 MLflow Tracking
        - Model experiment tracking  
        - Performance comparison  
        - Reproducible workflows  
        """)

    st.divider()

    # ============================
    # 💼 Business Impact
    # ============================
    st.subheader("💼 Business Impact")

    col1, col2, col3 = st.columns(3)

    col1.metric("📍 Hotspot Detection", "High Accuracy")
    col2.metric("⏱ Response Time", "Reduced")
    col3.metric("🚓 Resource Allocation", "Optimized")

    st.markdown("""
    - Enables *data-driven policing strategies*  
    - Improves *crime prevention planning*  
    - Supports *efficient patrol deployment*  
    """)

    st.divider()

    # ============================
    # 🚀 Call to Action
    # ============================
    st.info("👈 Use the navigation panel to explore Geo, Temporal, and PCA insights.")

# ---------------- GEO ----------------
elif page == "🌍Geo Analysis":

    st.title("📍 Geographic Crime Analysis")

    # ---------------- CLUSTER DISTRIBUTION ----------------
    if "geo_cluster_tuned" in df.columns:
        st.subheader("📊 Cluster Distribution")
        st.bar_chart(df["geo_cluster_tuned"].value_counts())
    else:
        st.warning("geo_cluster_tuned column not found in dataset")

    # ---------------- CLUSTER FILTER ----------------
    st.subheader("🎯 Filter by Cluster")

    if "geo_cluster_tuned" in df.columns:

        cluster_options = sorted(df["geo_cluster_tuned"].dropna().unique())

        selected_cluster = st.selectbox(
            "Select Cluster",
            options=["All"] + list(cluster_options)
        )

        if selected_cluster != "All":
            filtered_df = df[df["geo_cluster_tuned"] == selected_cluster]
        else:
            filtered_df = df

    else:
        st.warning("geo_cluster_tuned column not found")
        filtered_df = df

    # ---------------- MAP VISUALIZATION ----------------

    st.subheader("🗺️ Crime Locations by Geographic Clusters")

    if all(col in filtered_df.columns for col in ["latitude", "longitude", "geo_cluster_tuned"]):

        map_df = filtered_df[["latitude", "longitude", "geo_cluster_tuned"]].dropna()

        # Optional sampling for performance
        if len(map_df) > 10000:
            map_df = map_df.sample(10000, random_state=42)

        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="geo_cluster_tuned",
            zoom=10,
            height=600,
        )

        fig.update_layout(mapbox_style="open-street-map")

        st.plotly_chart(fig)

    else:
        st.warning("Required columns not found for map")

    # ---------------- CLUSTER SUMMARY ----------------
    st.subheader("📊 Cluster Summary")

    if "geo_cluster_tuned" in df.columns:

        summary_df = df.groupby("geo_cluster_tuned").agg({
            "latitude": "mean",
            "longitude": "mean",
            "geo_cluster_tuned": "count"
        }).rename(columns={"geo_cluster_tuned": "crime_count"})

        summary_df = summary_df.reset_index()

        # 🔥 Rename column
        summary_df = summary_df.rename(columns={
            "geo_cluster_tuned": "Cluster ID"
        })

        # 🔥 Sort by importance
        summary_df = summary_df.sort_values(by="crime_count", ascending=False)

        # 🔥 Add Rank column
        summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

        # 🔥 Clean index
        summary_df = summary_df.reset_index(drop=True)

        st.dataframe(summary_df)

    else:
        st.warning("geo_cluster_tuned column not found")

    st.subheader("🧠 Cluster Insights")

    if "geo_cluster_tuned" in df.columns:

    # Use same summary logic
        insights_df = df.groupby("geo_cluster_tuned").size().reset_index(name="crime_count")

        insights_df = insights_df.sort_values(by="crime_count", ascending=False).reset_index(drop=True)

        for i, row in insights_df.iterrows():

            cluster_id = row["geo_cluster_tuned"]
            count = row["crime_count"]

        # 🧠 Simple interpretation logic
            if i == 0:
              insight = "🔥 High crime dense region"
            elif i < 3:
              insight = "⚠️ Moderate crime zone"
            else:
              insight = "🟢 Low crime / sparse activity area"

            st.write(f"*Cluster {cluster_id}* → {insight} ({count} crimes)")

    else:
        st.warning("geo_cluster_tuned column not found")
    
    

# ---------------- TEMPORAL ----------------

elif page == "⏱️Temporal Analysis":

    st.title("⏱️ Temporal Crime Analysis Dashboard")

    # Load dataset
    # Load temporal dataset (cloud-safe)
    @st.cache_data
    def load_temporal_data():
        url = "https://drive.google.com/uc?id=1pf0e01Q0Xp7mzZIGQnhjkdLb23gcAJRt"
        return pd.read_csv(url)

    df_temp = load_temporal_data()

    # ===============================
    # 🎯 KPI CARDS (TOP SUMMARY)
    # ===============================
    peak_hour = df_temp["hour"].value_counts().idxmax() if "hour" in df_temp.columns else "N/A"
    peak_month = df_temp["month"].value_counts().idxmax() if "month" in df_temp.columns else "N/A"

    top_district = df_temp["district"].value_counts().idxmax() if "district" in df_temp.columns else "N/A"
    top_beat = df_temp["beat"].value_counts().idxmax() if "beat" in df_temp.columns else "N/A"

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("🕒 Peak Hour", f"{peak_hour}:00")
    k2.metric("📅 Peak Month", peak_month)
    k3.metric("🏙️ Top District", top_district)
    k4.metric("🚓 Top Beat", top_beat)

    st.markdown("---")

    # ===============================
    # 📂 TABS
    # ===============================
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Trends",
        "🔥 Heatmap",
        "📈 Insights",
        "🚨 Smart Patrol"
    ])

    # =====================================================
    # 📊 TAB 1 — TRENDS
    # =====================================================
    with tab1:
        st.subheader("Crime Distribution Patterns")

        col1, col2 = st.columns(2)

        with col1:
            if "hour" in df_temp.columns:
                st.markdown("*Crimes by Hour*")
                hourly_counts = df_temp["hour"].value_counts().sort_index()
                st.bar_chart(hourly_counts)

        with col2:
            if "month" in df_temp.columns:
                st.markdown("*Crimes by Month*")
                monthly_counts = df_temp["month"].value_counts().sort_index()
                st.bar_chart(monthly_counts)

    # =====================================================
    # 🔥 TAB 2 — HEATMAP
    # =====================================================
    with tab2:
        st.subheader("Crime Intensity Heatmap")

        if "hour" in df_temp.columns and "month" in df_temp.columns:
            heatmap_data = pd.crosstab(df_temp["hour"], df_temp["month"])
            st.dataframe(heatmap_data)

            st.info("Darker / higher values indicate peak crime combinations")

    # =====================================================
    # 📈 TAB 3 — INSIGHTS + STRATEGY
    # =====================================================
    with tab3:

        # Cluster Summary
        if "temporal_cluster" in df_temp.columns:
            st.subheader("Cluster-Based Behavior Analysis")

            summary = df_temp.groupby("temporal_cluster").agg({
                "hour": "mean",
                "day_num": "mean",
                "month": "mean"
            }).reset_index()

            summary.columns = ["Cluster", "Avg Hour", "Avg Day", "Avg Month"]
            st.dataframe(summary)

        # Insights Row
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Key Insights")

            st.success(f"Peak crime hour: {peak_hour}:00")
            st.success(f"Peak crime month: {peak_month}")

        with col2:
            st.subheader("Pattern Understanding")

            if peak_hour >= 18:
                shift = "Evening / Night Patrol"
            elif peak_hour >= 12:
                shift = "Afternoon Patrol"
            else:
                shift = "Morning Patrol"

            st.info(f"High activity period: {shift}")
            st.info("Crime shows strong temporal clustering patterns")

        # Risk Scoring
        if "district" in df_temp.columns:
            st.subheader("Risk Scoring by District")

            risk_df = df_temp["district"].value_counts().reset_index()
            risk_df.columns = ["District", "Crime Count"]
            risk_df["Risk Score"] = (risk_df["Crime Count"] / risk_df["Crime Count"].max()) * 100

            st.dataframe(risk_df)

        # Strategy Layout
        st.subheader("Police Strategy Framework")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🎯 Strategic Insights")
            st.write(f"- Peak Hour: {peak_hour}:00")
            st.write(f"- Peak Month: {peak_month}")
            st.write(f"- High Risk District: {top_district}")
            st.write(f"- Critical Beat: {top_beat}")

        with col2:
            st.markdown("### 🚔 Recommended Actions")
            st.write(f"- Increase {shift} patrol deployment")
            st.write(f"- Prioritize District {top_district}")
            st.write(f"- Strengthen monitoring in Beat {top_beat}")
            st.write("- Allocate resources dynamically")

        st.markdown("### 📊 Expected Impact")
        st.write("✅ Faster response time")
        st.write("✅ Improved crime prevention")
        st.write("✅ Optimized patrol allocation")

    # =====================================================
    # 🚨 TAB 4 — SMART PATROL + ALERTS
    # =====================================================
    with tab4:

        st.subheader("Smart Patrol Planning System")

        col1, col2 = st.columns(2)

        with col1:
            selected_hour = st.selectbox(
                "Select Hour",
                sorted(df_temp["hour"].dropna().unique())
            )

        with col2:
            selected_month = st.selectbox(
                "Select Month",
                sorted(df_temp["month"].dropna().unique())
            )

        combined_data = df_temp[
            (df_temp["hour"] == selected_hour) &
            (df_temp["month"] == selected_month)
        ]

        st.markdown(f"### 🎯 Deployment Recommendation")

        if not combined_data.empty:
            top_beat_combined = combined_data["beat"].value_counts().idxmax() if "beat" in df_temp.columns else "N/A"
            top_district_combined = combined_data["district"].value_counts().idxmax() if "district" in df_temp.columns else "N/A"

            st.success(f"Deploy units in Beat {top_beat_combined}")
            st.success(f"Focus on District {top_district_combined}")
            st.info("Optimized patrol for selected time window")

        else:
            st.warning("No sufficient data for this combination")

        # Predictive Alerts
        st.subheader("Real-Time Risk Alerts")

        if "district" in df_temp.columns:
            risk_df = df_temp["district"].value_counts().reset_index()
            risk_df.columns = ["District", "Crime Count"]
            risk_df["Risk Score"] = (risk_df["Crime Count"] / risk_df["Crime Count"].max()) * 100

            high_risk = risk_df[risk_df["Risk Score"] > 70]

            if not high_risk.empty:
                st.error("🚨 High Risk Districts Identified")

                for _, row in high_risk.iterrows():
                    st.write(f"⚠️ District {row['District']} → Risk Score: {row['Risk Score']:.2f}")
            else:
                st.success("All districts within safe thresholds")

# ---------------- PCA ----------------
elif page == "📈PCA Insights":

    st.title("📉 PCA Insights Dashboard") 
    # ============================
    # Load Model
    # ============================
    pca = joblib.load("pca_model.pkl")

    components = pca.components_
    total_components = components.shape[0]
    variance = sum(pca.explained_variance_ratio_)

    # ============================
    # Clean Feature Names (UI ONLY)
    # ============================
    clean_names = [
        "Hour", "Month", "District", "Beat",
        "Latitude", "Longitude",
        "Crime Count", "Geo Cluster",
        "Day", "Season", "Region", "Location Type"
    ]

    feature_names = clean_names[:components.shape[1]]

    comp_df = pd.DataFrame(components, columns=feature_names)

    # ============================
    # KPI
    # ============================
    col1, col2 = st.columns(2)
    col1.metric("Total Components", total_components)
    col2.metric("Variance Captured", f"{variance:.2f}")

    st.divider()

    # ============================
    # Tabs
    # ============================
    tab1, tab2, tab3 = st.tabs([
        "📊 Variance Analysis",
        "📋 Feature Importance",
        "💡 Insights"
    ])

    # ============================
    # TAB 1 → Variance
    # ============================
    with tab1:

        st.subheader("Explained Variance by Components")
        st.bar_chart(pca.explained_variance_ratio_)

        st.subheader("Cumulative Variance")
        st.line_chart(np.cumsum(pca.explained_variance_ratio_))

    # ============================
    # TAB 2 → Feature Importance
    # ============================
    with tab2:

        st.subheader("Feature Contribution to Principal Components")
        st.dataframe(comp_df)

        st.info("Higher absolute values indicate stronger influence")

        # ==========================================
        # 🔥 Top Influential Features (GRID FIXED)
        # ==========================================
        st.subheader("🔥 Top Influential Features per Component")

        num_cols = 3
        cols = st.columns(num_cols)

        for i in range(len(comp_df)):

            top_features = (
                comp_df.iloc[i]
                .abs()
                .sort_values(ascending=False)
                .head(3)
            )

            with cols[i % num_cols]:
                st.markdown(f"""
                *PC{i}*
                - {top_features.index[0]}
                - {top_features.index[1]}
                - {top_features.index[2]}
                """)

    # ============================
    # TAB 3 → Insights
    # ============================
    with tab3:

        st.subheader("Key PCA Insights")

        st.success("Most important component: PC1")
        st.warning("Consider increasing components for better variance")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔍 Interpretation")
            st.markdown("""
            - PCA reduces dimensionality while retaining key patterns  
            - First few components capture majority of variance  
            - Helps simplify clustering and visualization  
            - Removes noise and redundant features  
            """)

        with col2:
            st.markdown("### 💼 Business Impact")
            st.markdown("""
            - Faster model computation  
            - Better clustering performance  
            - Improved pattern detection  
            - Simplified feature space for decision making  
            """)
