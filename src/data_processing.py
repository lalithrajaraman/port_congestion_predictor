import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0 # km
    return c * r

def get_port_name(lat, lon):
    port_geofences = [
        {"name": "Houston & Galveston", "lat_range": (29.2, 30.0), "lon_range": (-95.3, -94.6)},
        {"name": "Miami & Port Everglades", "lat_range": (25.7, 26.3), "lon_range": (-80.3, -80.0)},
        {"name": "Seattle & Puget Sound", "lat_range": (47.0, 48.0), "lon_range": (-122.6, -122.1)},
        {"name": "New Orleans & Mississippi River", "lat_range": (29.5, 30.2), "lon_range": (-91.0, -89.5)},
        {"name": "San Diego Harbor", "lat_range": (32.5, 32.8), "lon_range": (-117.3, -117.0)},
        {"name": "New York & New Jersey", "lat_range": (40.4, 40.9), "lon_range": (-74.3, -73.8)},
        {"name": "LA & Long Beach", "lat_range": (33.6, 33.8), "lon_range": (-118.3, -118.1)},
        {"name": "San Francisco & Oakland", "lat_range": (37.5, 38.0), "lon_range": (-122.5, -122.1)},
        {"name": "Detroit & Great Lakes River", "lat_range": (42.0, 43.0), "lon_range": (-83.5, -82.0)},
        {"name": "Duluth & Great Lakes Harbor", "lat_range": (46.5, 46.9), "lon_range": (-92.2, -91.9)},
        {"name": "Jacksonville", "lat_range": (30.2, 30.6), "lon_range": (-81.7, -81.3)},
        {"name": "Corpus Christi", "lat_range": (27.6, 27.9), "lon_range": (-97.5, -97.0)},
        {"name": "Mobile", "lat_range": (30.2, 30.8), "lon_range": (-88.1, -87.8)},
        {"name": "Savannah", "lat_range": (31.9, 32.2), "lon_range": (-81.2, -80.8)},
        {"name": "Boston Harbor", "lat_range": (42.2, 42.5), "lon_range": (-71.1, -70.8)}
    ]
    for p in port_geofences:
        if (p["lat_range"][0] <= lat <= p["lat_range"][1]) and (p["lon_range"][0] <= lon <= p["lon_range"][1]):
            return p["name"]
    return f"Port Region ({lat:.2f}, {lon:.2f})"

def discover_ports(raw_csv_path, max_ports=15):
    print("Pre-scanning dataset to automatically discover ports...")
    chunk_size = 500000
    moored_coords = []
    
    for i, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunk_size)):
        chunk.loc[chunk['Status'].isna() & (chunk['SOG'] < 0.5), 'Status'] = 5.0
        stationary = chunk[chunk['Status'].isin([1.0, 5.0]) | (chunk['SOG'] < 0.5)][['LAT', 'LON']].dropna()
        moored_coords.append(stationary)
        
    df_moored = pd.concat(moored_coords, ignore_index=True)
    print(f"Total stationary records found: {len(df_moored)}")
    
    df_moored['lat_round'] = df_moored['LAT'].round(1)
    df_moored['lon_round'] = df_moored['LON'].round(1)
    
    clusters = df_moored.groupby(['lat_round', 'lon_round']).agg(
        mean_lat=('LAT', 'mean'),
        mean_lon=('LON', 'mean'),
        count=('LAT', 'count')
    ).reset_index()
    
    top_clusters = clusters.sort_values(by='count', ascending=False).head(max_ports).reset_index(drop=True)
    
    ports = []
    for idx, row in top_clusters.iterrows():
        lat, lon = row['mean_lat'], row['mean_lon']
        name = get_port_name(lat, lon)
        ports.append({
            "port_id": idx,
            "port_name": name,
            "lat": lat,
            "lon": lon,
            "count": row['count']
        })
        print(f"Detected Port {idx}: {name} at ({lat:.4f}, {lon:.4f}) with {row['count']} transmissions")
        
    return pd.DataFrame(ports)

def process_raw_data(raw_csv_path, output_dir):
    # Step 1: Discover port registries
    ports_df = discover_ports(raw_csv_path)
    os.makedirs(output_dir, exist_ok=True)
    ports_df.to_csv(os.path.join(output_dir, "port_registry.csv"), index=False)
    
    print("\nStarting main feature extraction pipeline...")
    chunk_size = 500000
    filtered_chunks = []
    
    port_lats = ports_df['lat'].values
    port_lons = ports_df['lon'].values
    port_ids = ports_df['port_id'].values
    
    # Step 2: Read chunks, associate pings to nearest port via numpy broadcasting (ultra-fast)
    for i, chunk in enumerate(pd.read_csv(raw_csv_path, chunksize=chunk_size)):
        chunk = chunk.dropna(subset=['LAT', 'LON']).copy()
        if len(chunk) == 0:
            continue
            
        # Extract latitudes and longitudes from chunk
        c_lats = chunk['LAT'].values[:, np.newaxis] # Shape (N, 1)
        c_lons = chunk['LON'].values[:, np.newaxis] # Shape (N, 1)
        
        # Broadcast against port list (shape: 1, M)
        p_lats_rad = np.radians(port_lats[np.newaxis, :])
        p_lons_rad = np.radians(port_lons[np.newaxis, :])
        c_lats_rad = np.radians(c_lats)
        c_lons_rad = np.radians(c_lons)
        
        dlat = p_lats_rad - c_lats_rad
        dlon = p_lons_rad - c_lons_rad
        
        a = np.sin(dlat/2)**2 + np.cos(c_lats_rad) * np.cos(p_lats_rad) * np.sin(dlon/2)**2
        c_dist = 2 * np.arcsin(np.sqrt(a))
        dists_matrix = c_dist * 6371.0 # Distance in km (Shape: N, M)
        
        # Find closest port ID and distance
        min_dists = np.min(dists_matrix, axis=1)
        min_idx = np.argmin(dists_matrix, axis=1)
        
        chunk['nearest_port_id'] = port_ids[min_idx]
        chunk['dist_to_port'] = min_dists
        
        # Filter for ships within 50km zone of any port
        filtered_chunk = chunk[chunk['dist_to_port'] <= 50.0].copy()
        filtered_chunks.append(filtered_chunk)
        print(f"Processed chunk {i+1}, kept {len(filtered_chunk)} rows within port zones")
        
    df = pd.concat(filtered_chunks, ignore_index=True)
    print(f"Total nationwide rows in port zones: {len(df)}")
    
    if len(df) == 0:
        print("Error: No data rows found within 50km of any port.")
        return
        
    print("Cleaning and sorting dataset...")
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
    df = df.sort_values(by=['MMSI', 'BaseDateTime']).reset_index(drop=True)
    
    # Impute Status
    df.loc[df['Status'].isna() & (df['SOG'] < 0.5), 'Status'] = 5.0
    df.loc[df['Status'].isna() & (df['SOG'] >= 0.5), 'Status'] = 0.0
    
    # Impute Heading
    df.loc[(df['Heading'].isna()) | (df['Heading'] == 511.0), 'Heading'] = df['COG']
    df['Heading'] = df['Heading'].fillna(0)
    
    print("Calculating port congestion metrics...")
    df['TimeBin'] = df['BaseDateTime'].dt.round('10min')
    
    # Calculate congestion per time bin per port
    active_mask = df['SOG'] >= 0.5
    congestion_active = df[active_mask].groupby(['TimeBin', 'nearest_port_id'])['MMSI'].nunique().to_dict()
    congestion_stationary = df[~active_mask].groupby(['TimeBin', 'nearest_port_id'])['MMSI'].nunique().to_dict()
    
    # Create key tuples for lookup
    lookup_keys = list(zip(df['TimeBin'], df['nearest_port_id']))
    
    df['port_active_density'] = [congestion_active.get(k, 0) for k in lookup_keys]
    df['port_stationary_density'] = [congestion_stationary.get(k, 0) for k in lookup_keys]
    df['port_total_density'] = df['port_active_density'] + df['port_stationary_density']
    
    print("Computing target variable (dwell time to berth globally)...")
    vessels = []
    grouped = df.groupby('MMSI')
    
    for mmsi, group in grouped:
        mooring_points = group[group['Status'] == 5.0]
        if mooring_points.empty:
            continue
            
        first_mooring_time = mooring_points['BaseDateTime'].min()
        
        # Keep points before mooring
        approach_group = group[group['BaseDateTime'] <= first_mooring_time].copy()
        approach_group['dwell_time'] = (first_mooring_time - approach_group['BaseDateTime']).dt.total_seconds() / 3600.0
        vessels.append(approach_group)
        
    if not vessels:
        print("Error: No vessels achieved mooring status in the port zones.")
        return
        
    df_modeling = pd.concat(vessels, ignore_index=True)
    print(f"Modeling dataset rows (approaching vessels): {len(df_modeling)}")
    
    print("Engineering features...")
    df_modeling['Length'] = df_modeling['Length'].fillna(df_modeling['Length'].median())
    df_modeling['Width'] = df_modeling['Width'].fillna(df_modeling['Width'].median())
    df_modeling['Draft'] = df_modeling['Draft'].fillna(df_modeling['Draft'].median())
    
    df_modeling['size_area'] = df_modeling['Length'] * df_modeling['Width']
    df_modeling['draft_ratio'] = np.where(
        df_modeling['Length'] > 0, 
        df_modeling['Draft'] / df_modeling['Length'], 
        0.0
    )
    
    output_path = os.path.join(output_dir, "processed_features.csv")
    columns_to_save = [
        'MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading', 
        'VesselType', 'Length', 'Width', 'Draft', 'Cargo', 
        'port_active_density', 'port_stationary_density', 'port_total_density', 
        'dist_to_port', 'size_area', 'draft_ratio', 'nearest_port_id', 'dwell_time'
    ]
    df_modeling[columns_to_save].to_csv(output_path, index=False)
    print(f"Feature engineering finished! Features saved to {output_path}")

if __name__ == "__main__":
    default_root = Path(__file__).resolve().parent.parent
    default_out_dir = default_root / "data"
    
    parser = argparse.ArgumentParser(description="Vessel Dwell Time Data Processing Pipeline")
    parser.add_argument(
        "--raw-path", 
        type=str, 
        default=None,
        help="Path to the raw AIS CSV file (e.g. AIS_2017_01_10.csv)"
    )
    parser.add_argument(
        "--out-dir", 
        type=str, 
        default=str(default_out_dir),
        help="Directory to save the processed output features"
    )
    
    args = parser.parse_args()
    
    # Determine raw path with fallbacks
    raw_path = args.raw_path
    if raw_path is None:
        # Fallback 1: Check user's local path
        user_local_path = r"C:\Users\rlali\OneDrive\Documents\AIS_2017_01_10\AIS_2017_01_10.csv"
        # Fallback 2: Check repo data directory
        repo_data_path = default_root / "data" / "raw_data.csv"
        
        if os.path.exists(user_local_path):
            raw_path = user_local_path
        elif repo_data_path.exists():
            raw_path = str(repo_data_path)
        else:
            print("Error: Raw dataset path not specified and default fallbacks do not exist.")
            print("Please run this script specifying the path using: --raw-path <path_to_csv>")
            exit(1)
    
    # Verify file exists
    if not os.path.exists(raw_path):
        print(f"Error: Specified raw dataset does not exist: {raw_path}")
        exit(1)
            
    process_raw_data(raw_path, args.out_dir)
