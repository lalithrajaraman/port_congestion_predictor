import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="US Vessel Dwell Time Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dark-themed CSS
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stApp {
        background-color: #0f1116;
    }
    h1, h2, h3 {
        color: #38bdf8 !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.875rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions to load data and models
@st.cache_data
def load_data():
    data_path = "data/processed_features.csv"
    registry_path = "data/port_registry.csv"
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    ports = pd.read_csv(registry_path) if os.path.exists(registry_path) else None
    if df is not None:
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    return df, ports

@st.cache_resource
def load_model_assets():
    model_path = "models/best_model.pkl"
    meta_path = "models/model_metadata.pkl"
    if os.path.exists(model_path) and os.path.exists(meta_path):
        model = joblib.load(model_path)
        meta = joblib.load(meta_path)
        return model, meta
    return None, None

df, ports_df = load_data()
model, meta = load_model_assets()

# --- Sidebar Controls ---
st.sidebar.markdown("<h1 style='text-align: center; font-size: 4rem; margin-top: -20px; margin-bottom: 10px;'>🚢</h1>", unsafe_allow_html=True)
st.sidebar.title("Port Registry Navigation")
st.sidebar.markdown("Filter vessels by port region, speed class, and operational status.")

if df is not None and ports_df is not None:
    # Port Selection
    port_names = ["All Ports (National View)"] + list(ports_df['port_name'].unique())
    selected_port = st.sidebar.selectbox("Select Port Focus", port_names)
    
    # Filter by Vessel Type
    vessel_types = sorted([t for t in df['VesselType'].dropna().unique() if t != -1])
    type_options = ["All"] + [str(int(t)) for t in vessel_types]
    selected_type = st.sidebar.selectbox("Vessel Type Code", type_options)
    
    # Filter by SOG
    min_sog, max_sog = float(df['SOG'].min()), float(df['SOG'].max())
    sog_range = st.sidebar.slider("Speed Over Ground (SOG)", min_sog, max_sog, (min_sog, max_sog))
    
    # Filter dataset
    filtered_df = df.copy()
    
    if selected_port != "All Ports (National View)":
        # Find port IDs corresponding to the selected name
        port_ids = ports_df[ports_df['port_name'] == selected_port]['port_id'].values
        filtered_df = filtered_df[filtered_df['nearest_port_id'].isin(port_ids)]
        
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df['VesselType'] == float(selected_type)]
        
    filtered_df = filtered_df[(filtered_df['SOG'] >= sog_range[0]) & (filtered_df['SOG'] <= sog_range[1])]
else:
    filtered_df = pd.DataFrame()
    selected_port = "All Ports (National View)"

st.sidebar.markdown("---")
st.sidebar.markdown("**Automated Port Discovery**")
st.sidebar.markdown(
    "Designed for **MathCo** as a generic, automated port discovery ML engine. "
    "Reads raw GPS traces, groups mooring events, and forecasts delays across all discovered harbors."
)

# --- Main Dashboard ---
st.title("🚢 Nationwide US Vessel Dwell Time Prediction Dashboard")
st.markdown("Monitoring vessel densities, waiting queues, and predicting berth delays across automatically discovered US port zones.")

if df is None or ports_df is None:
    st.error("Processed features or Port Registry CSV not found. Please run the data processing pipeline first.")
    st.stop()

# --- Row 1: KPI Cards ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_ships = filtered_df['MMSI'].nunique()
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{total_ships}</div><div class="metric-lbl">Ships in Port Zone</div></div>',
        unsafe_allow_html=True
    )

with col2:
    active_ships = filtered_df[filtered_df['SOG'] >= 0.5]['MMSI'].nunique()
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{active_ships}</div><div class="metric-lbl">Active Arrivals</div></div>',
        unsafe_allow_html=True
    )

with col3:
    avg_sog = filtered_df['SOG'].mean()
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{avg_sog:.1f} kts</div><div class="metric-lbl">Avg Fleet Speed</div></div>',
        unsafe_allow_html=True
    )

with col4:
    avg_dwell = filtered_df['dwell_time'].mean()
    st.markdown(
        f'<div class="metric-card"><div class="metric-val">{avg_dwell:.1f} hrs</div><div class="metric-lbl">Avg Wait to Berth</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# --- Row 2: Map & Port-level Analysis ---
tab1, tab2 = st.tabs(["🗺️ Regional Traffic Map", "📊 Nationwide Port Comparison"])

with tab1:
    st.subheader(f"Vessel Positions - {selected_port}")
    st.markdown("Red dots represent ships moving at speed. Blue/cyan dots indicate anchored or stationary vessels.")
    
    # Configure Map center based on selected port
    if selected_port == "All Ports (National View)":
        map_lat, map_lon, map_zoom = 38.0, -97.0, 3.5
        # Limit data size for rendering performance in national view
        map_df = filtered_df.sample(min(len(filtered_df), 8000), random_state=42)
    else:
        # Get coordinates of selected port
        port_row = ports_df[ports_df['port_name'] == selected_port].iloc[0]
        map_lat, map_lon, map_zoom = float(port_row['lat']), float(port_row['lon']), 10.0
        map_df = filtered_df
        
    view_state = pdk.ViewState(
        latitude=map_lat,
        longitude=map_lon,
        zoom=map_zoom,
        pitch=40
    )
    
    vessel_layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position=["LON", "LAT"],
        get_color="[SOG * 20, 100, 255 - (SOG * 20), 180]",
        get_radius=200 if selected_port != "All Ports (National View)" else 4000,
        pickable=True,
    )
    
    deck = pdk.Deck(
        layers=[vessel_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v10",
        tooltip={"text": "MMSI: {MMSI}\nSpeed: {SOG} knots\nPort wait: {dwell_time} hrs"}
    )
    st.pydeck_chart(deck)

with tab2:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Busiest Port Regions by Transmission Volume")
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0f1116')
        ax.set_facecolor('#1e293b')
        
        # Merge port names to count transmissions
        df_named = df.merge(ports_df, left_on='nearest_port_id', right_on='port_id', how='left')
        port_counts = df_named['port_name'].value_counts().head(10)
        
        port_counts.plot(kind='barh', color='#38bdf8', ax=ax)
        ax.set_title("Top 10 Most Active Port Zones", color='#38bdf8')
        ax.set_xlabel("Transmissions Count", color='#e2e8f0')
        ax.tick_params(colors='#e2e8f0', labelsize=8)
        st.pyplot(fig)

    with col_chart2:
        st.subheader("Average Dwell Time by Port Region")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        fig2.patch.set_facecolor('#0f1116')
        ax2.set_facecolor('#1e293b')
        
        # Average wait times per port
        port_waits = df_named.groupby('port_name')['dwell_time'].mean().sort_values(ascending=False).head(10)
        port_waits.plot(kind='bar', color='#f43f5e', ax=ax2)
        
        ax2.set_title("Top 10 Ports by Average Berth Wait Time", color='#38bdf8')
        ax2.set_ylabel("Wait Time (hours)", color='#e2e8f0')
        ax2.tick_params(colors='#e2e8f0', labelsize=8)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig2)

st.markdown("<br>", unsafe_allow_html=True)

# --- Row 3: Predictive Engine & Explainability ---
st.markdown("---")
st.header("🎯 Nationwide Dwell Time Predictive Engine")
st.markdown("Calculate the estimated waiting time for any ship approaching any of the 15 discovered port areas.")

if model is None or meta is None:
    st.warning("Model binaries not found. Please run the model training script first.")
else:
    predict_col, result_col = st.columns([1.5, 2])
    
    with predict_col:
        st.subheader("Vessel Parameters")
        select_method = st.radio("Vessel Selection Method", ["Select Active Vessel in Port Zone", "Input Custom Parameters"])
        
        if select_method == "Select Active Vessel in Port Zone":
            sample_vessels = sorted(filtered_df['MMSI'].unique())[:20]
            if not sample_vessels:
                st.warning("No ships available under current filters. Showing fallback inputs.")
                selected_mmsi = None
            else:
                selected_mmsi = st.selectbox("Select Ship MMSI", sample_vessels)
                
            if selected_mmsi:
                ship_data = filtered_df[filtered_df['MMSI'] == selected_mmsi].iloc[0]
                lat_input = float(ship_data['LAT'])
                lon_input = float(ship_data['LON'])
                sog_input = float(ship_data['SOG'])
                cog_input = float(ship_data['COG'])
                heading_input = float(ship_data['Heading'])
                vtype_input = float(ship_data['VesselType'])
                len_input = float(ship_data['Length'])
                wid_input = float(ship_data['Width'])
                draft_input = float(ship_data['Draft'])
                cargo_input = float(ship_data['Cargo']) if not pd.isna(ship_data['Cargo']) else -1.0
                port_id_input = int(ship_data['nearest_port_id'])
                
                # Fetch port name
                p_name = ports_df[ports_df['port_id'] == port_id_input]['port_name'].iloc[0]
                st.info(f"Target Port: **{p_name}** | Length={len_input}m | Draft={draft_input}m")
            else:
                # Mock inputs if list empty
                lat_input, lon_input, sog_input = 29.7, -95.0, 5.0
                cog_input, heading_input = 180.0, 180.0
                vtype_input, len_input, wid_input, draft_input, cargo_input = 1024.0, 150.0, 20.0, 8.0, 70.0
                port_id_input = 0
        else:
            # Manual inputs
            port_selection = st.selectbox(
                "Target Port Destination", 
                ports_df['port_name'].unique()
            )
            port_row = ports_df[ports_df['port_name'] == port_selection].iloc[0]
            port_id_input = int(port_row['port_id'])
            
            # Autocomplete coordinates close to the selected port
            lat_input = st.number_input("LAT Coordinate", value=float(port_row['lat'] + 0.05))
            lon_input = st.number_input("LON Coordinate", value=float(port_row['lon'] - 0.05))
            
            sog_input = st.slider("Current Speed (SOG knots)", 0.0, 25.0, 5.0)
            cog_input = st.slider("Course Over Ground (COG degrees)", 0.0, 360.0, 180.0)
            heading_input = st.slider("Heading (degrees)", 0.0, 360.0, 180.0)
            vtype_input = st.selectbox("Vessel Class Type", [1004, 1012, 1019, 1024, 1025])
            len_input = st.number_input("Vessel Length (m)", min_value=10.0, max_value=400.0, value=180.0)
            wid_input = st.number_input("Vessel Width (m)", min_value=5.0, max_value=60.0, value=25.0)
            draft_input = st.number_input("Vessel Draft (m)", min_value=1.0, max_value=25.0, value=9.0)
            cargo_input = st.selectbox("Cargo Category", [-1.0, 30.0, 50.0, 70.0, 80.0])

        hour_input = st.slider("Hour of Day (0-23)", 0, 23, 12)
        
        # Distance calculation to target port center
        p_lat = float(ports_df[ports_df['port_id'] == port_id_input]['lat'].iloc[0])
        p_lon = float(ports_df[ports_df['port_id'] == port_id_input]['lon'].iloc[0])
        
        from src.data_processing import haversine_distance
        dist_port = haversine_distance(lat_input, lon_input, p_lat, p_lon)
        
        size_area = len_input * wid_input
        draft_ratio = draft_input / len_input if len_input > 0 else 0.0
        
        # Estimate congestion for selected port at selected hour
        matched_hour_data = df[(df['BaseDateTime'].dt.hour == hour_input) & (df['nearest_port_id'] == port_id_input)]
        if not matched_hour_data.empty:
            port_active_density = float(matched_hour_data['port_active_density'].mean())
            port_stationary_density = float(matched_hour_data['port_stationary_density'].mean())
            port_total_density = float(matched_hour_data['port_total_density'].mean())
        else:
            port_active_density = 30.0
            port_stationary_density = 50.0
            port_total_density = 80.0

    with result_col:
        st.subheader("Nationwide Wait Predictions")
        
        input_data = pd.DataFrame([{
            'LAT': lat_input,
            'LON': lon_input,
            'SOG': sog_input,
            'COG': cog_input,
            'Heading': heading_input,
            'VesselType': vtype_input,
            'Length': len_input,
            'Width': wid_input,
            'Draft': draft_input,
            'Cargo': cargo_input,
            'port_active_density': port_active_density,
            'port_stationary_density': port_stationary_density,
            'port_total_density': port_total_density,
            'dist_to_port': dist_port,
            'size_area': size_area,
            'draft_ratio': draft_ratio,
            'nearest_port_id': port_id_input,
            'hour': hour_input
        }])
        
        # Align features with model requirement
        input_data = input_data[meta['feature_names']]
        
        # Inference
        predicted_dwell = model.predict(input_data)[0]
        predicted_dwell = max(0.0, predicted_dwell)
        
        st.markdown(f"""
        <div style="background-color:#1e293b; padding: 25px; border-radius: 12px; border: 2px solid #38bdf8; text-align: center; margin-bottom: 25px;">
            <span style="font-size: 1.25rem; color:#94a3b8; text-transform: uppercase;">Predicted Berth wait time</span>
            <h2 style="font-size: 3.5rem; margin: 10px 0; color:#38bdf8;">{predicted_dwell:.2f} Hours</h2>
            <p style="color:#94a3b8; font-size:0.9rem;">Estimated remaining transit and wait time before docking.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Global Model Feature Importances (All Ports combined)**")
        st.markdown("Relative feature importances driving predictions across the nationwide port registry:")
        
        fig_imp, ax_imp = plt.subplots(figsize=(6, 3.5))
        fig_imp.patch.set_facecolor('#0f1116')
        ax_imp.set_facecolor('#1e293b')
        
        sorted_fi = sorted(meta["feature_importances"].items(), key=lambda x: x[1], reverse=True)[:8]
        feats, imps = zip(*sorted_fi)
        
        ax_imp.barh(feats[::-1], imps[::-1], color='#38bdf8')
        ax_imp.tick_params(colors='#e2e8f0', labelsize=8)
        st.pyplot(fig_imp)
