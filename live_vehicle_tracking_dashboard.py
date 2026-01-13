import streamlit as st
import psycopg2
import pandas as pd
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import time
import math
import os
from geopy.geocoders import Nominatim
from streamlit_autorefresh import st_autorefresh
from dotenv import load_dotenv
import pytz

# Load environment variables from .env file
load_dotenv()

# Vehicles to exclude from the dashboard (normalized format - uppercase, no spaces/dashes)
EXCLUDED_VEHICLES = [
    'NL01AB2275',
]

# Custom owner name mappings (normalized vehicle_no -> owner_name)
CUSTOM_OWNER_MAPPING = {
    'HR55AM1370': 'Ranjeet Singh Logistics',
    'HR55AP1974': 'Ranjeet Singh Logistics',
    'HR55AM9667': 'Ranjeet Singh Logistics',
    'HR55AM2340': 'Ranjeet Singh Logistics',
    'NL01Q8157': 'Ranjeet Singh Logistics',
    'HR55AM6059': 'Ranjeet Singh Logistics',
    'HR55AM8703': 'Ranjeet Singh Logistics',
    'HR55AN5406': 'Ranjeet Singh Logistics',
    'HR55AM0907': 'Ranjeet Singh Logistics',
}

# Page config
st.set_page_config(
    page_title="Swift Live Tracking",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* PREVENT SCREEN BLUR/FLASH DURING AUTO-REFRESH */
    /* Hide all loading indicators */
    .stSpinner, [data-testid="stStatusWidget"], .stProgress, div[data-testid="stSpinner"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Force all content to stay visible during rerun */
    .main, .main .block-container, [data-testid="stAppViewContainer"],
    [data-testid="stAppViewBlockContainer"], .stApp, section.main {
        opacity: 1 !important;
        visibility: visible !important;
        filter: none !important;
        -webkit-filter: none !important;
    }

    /* Override Streamlit's fade/blur transitions */
    *, *::before, *::after {
        transition: none !important;
        animation: none !important;
    }

    /* Prevent the "Running..." overlay */
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        opacity: 1 !important;
    }

    /* Keep iframes visible (for HTML components) */
    iframe {
        opacity: 1 !important;
        visibility: visible !important;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: left;
        padding: 1rem;
        margin-top: 0rem;
    }

    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #FFFFFF !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #B0B0B0 !important;
    }

    [data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid rgba(28, 131, 225, 0.3);
    }

    /* Reduce size of auto-refresh controls */
    [data-testid="stCheckbox"] label {
        font-size: 0.85rem !important;
    }

    [data-testid="stCheckbox"] input[type="checkbox"] {
        width: 16px !important;
        height: 16px !important;
    }

    [data-testid="stSelectbox"] label {
        font-size: 0.9rem !important;
        margin-bottom: 0.3rem !important;
        font-weight: 600 !important;
    }

    [data-testid="stSelectbox"] > div > div {
        font-size: 0.9rem !important;
        padding: 0.75rem 1rem !important;
        min-height: 3.5rem !important;
        height: auto !important;
        width: 100% !important;
        line-height: 1.5 !important;
    }

    [data-testid="stSelectbox"] input {
        font-size: 0.9rem !important;
        color: #FFFFFF !important;
    }

    /* Ensure dropdown text is fully visible */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        overflow: visible !important;
    }

    /* Fix selected value display - prevent truncation and ensure visibility */
    [data-testid="stSelectbox"] div[data-baseweb="select"] span {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: nowrap !important;
        color: #FFFFFF !important;
        font-size: 0.9rem !important;
        letter-spacing: normal !important;
        display: inline-block !important;
        line-height: 1.5 !important;
    }

    /* Ensure selectbox selected value is visible in dark mode */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        color: #FFFFFF !important;
        font-size: 0.9rem !important;
        height: auto !important;
        line-height: 1.5 !important;
    }

    [data-testid="stSelectbox"] [role="option"] {
        white-space: normal !important;
        word-wrap: break-word !important;
        padding: 0.5rem !important;
    }

    button[kind="primary"] {
        font-size: 0.85rem !important;
        padding: 0.35rem 0.8rem !important;
        height: auto !important;
        min-height: 2.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula (in km)"""
    R = 6371  # Earth's radius in kilometers

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c
    return distance

def geocode_location(place_name):
    """Convert place name to coordinates using geocoding"""
    try:
        geolocator = Nominatim(user_agent="axestrack_vehicle_tracking")
        location = geolocator.geocode(place_name, timeout=10)

        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None, None

def get_database_connection():
    """Create database connection using environment variables"""
    return psycopg2.connect(
        host=os.getenv("Host"),
        user=os.getenv("UserName"),
        password=os.getenv("Password"),
        database=os.getenv("database_name"),
        port=int(os.getenv("Port", 5432)),
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )

@st.cache_data(ttl=600, show_spinner=False)
def load_owner_mapping():
    """Load owner name mapping from Excel file"""
    try:
        excel_path = os.path.join(os.path.dirname(__file__), 'party name map.xlsx')
        owner_df = pd.read_excel(excel_path)
        owner_df['normalized_vehicle_no'] = owner_df['RegistrationNo'].apply(
            lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
        )
        owner_df['owner_name'] = owner_df['PartyName'].apply(
            lambda x: 'Own Vehicle' if x in ['Swift Road Link Pvt. Ltd.', 'Nishant Saini Associates'] else x
        )
        return owner_df[['normalized_vehicle_no', 'owner_name']]
    except Exception as e:
        return pd.DataFrame(columns=['normalized_vehicle_no', 'owner_name'])

@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes, silent refresh
def load_vehicle_data():
    """Load latest vehicle tracking data"""
    connection = None
    try:
        connection = get_database_connection()

        # Get latest location for each vehicle (unique vehicles only)
        # Also get the last time each vehicle was moving (speed > 0) for idle time calculation
        query = """
            WITH ranked_records AS (
                SELECT
                    id,
                    vehicle_no,
                    imei,
                    location,
                    date_time,
                    temperature,
                    ignition,
                    latitude,
                    longitude,
                    speed,
                    angle,
                    odometer,
                    pincode,
                    recorded_at,
                    ROW_NUMBER() OVER (
                        PARTITION BY vehicle_no
                        ORDER BY date_time DESC, id DESC
                    ) as rn
                FROM fvts_vehicles
                WHERE latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND latitude != 0
                    AND longitude != 0
            ),
            last_moving AS (
                SELECT DISTINCT ON (vehicle_no)
                    vehicle_no,
                    date_time as last_moving_time
                FROM fvts_vehicles
                WHERE speed > 0
                    AND date_time >= NOW() - INTERVAL '7 days'
                ORDER BY vehicle_no, date_time DESC
            )
            SELECT
                r.id,
                r.vehicle_no,
                r.imei,
                r.location,
                r.date_time,
                r.temperature,
                r.ignition,
                r.latitude,
                r.longitude,
                r.speed,
                r.angle,
                r.odometer,
                r.pincode,
                r.recorded_at,
                lm.last_moving_time
            FROM ranked_records r
            LEFT JOIN last_moving lm ON r.vehicle_no = lm.vehicle_no
            WHERE r.rn = 1
            ORDER BY r.vehicle_no;
        """

        df = pd.read_sql_query(query, connection)

        # Filter out excluded vehicles
        if EXCLUDED_VEHICLES:
            df['normalized_vehicle'] = df['vehicle_no'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            df = df[~df['normalized_vehicle'].isin(EXCLUDED_VEHICLES)]
            df = df.drop(columns=['normalized_vehicle'])

        # Convert timestamps
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        if 'recorded_at' in df.columns:
            df['recorded_at'] = pd.to_datetime(df['recorded_at'], errors='coerce')
        if 'last_moving_time' in df.columns:
            df['last_moving_time'] = pd.to_datetime(df['last_moving_time'], errors='coerce')

        # Calculate time since last update
        now = datetime.now()
        df['minutes_ago'] = df['date_time'].apply(
            lambda x: int((now - x).total_seconds() / 60) if pd.notna(x) else None
        )

        df['hours_ago'] = df['date_time'].apply(
            lambda x: (now - x).total_seconds() / 3600 if pd.notna(x) else None
        )

        # Determine status with updated logic
        def get_status(row):
            if pd.isna(row['hours_ago']):
                return 'Unknown'
            # Stopped: Last update not received for 6 hours AND ignition is OFF
            elif row['hours_ago'] >= 6 and row['ignition'] == 0:
                return 'Stopped'
            # Moving: Speed > 0 (regardless of ignition status)
            elif row['speed'] > 0:
                return 'Moving'
            # Idle: Speed == 0 (ignition can be ON or OFF)
            elif row['speed'] == 0:
                return 'Idle'
            else:
                return 'Unknown'

        df['status'] = df.apply(get_status, axis=1)

        # Calculate ACTUAL idle time (time since vehicle was last moving with speed > 0)
        def calculate_idle_time(row):
            if row['status'] != 'Idle':
                return None

            # If vehicle is currently moving, no idle time
            if row['speed'] > 0:
                return None

            # If we have last_moving_time, calculate actual idle duration
            if pd.notna(row.get('last_moving_time')):
                idle_hours = (now - row['last_moving_time']).total_seconds() / 3600
                if idle_hours < 0:
                    idle_hours = 0

                # Format based on duration
                if idle_hours >= 24:
                    # Show in days and hours if >= 24 hours
                    days = int(idle_hours // 24)
                    remaining_hours = int(idle_hours % 24)
                    return f"{days}d {remaining_hours}h"
                else:
                    # Show in hours and minutes if < 24 hours
                    hours = int(idle_hours)
                    minutes = int((idle_hours % 1) * 60)
                    return f"{hours}h {minutes}m"
            else:
                # No movement data in last 7 days, show as unknown
                return ">7 days"

        df['idle_time'] = df.apply(calculate_idle_time, axis=1)

        # Add color coding for map
        def get_color(status):
            if status == 'Moving':
                return [0, 255, 0, 160]  # Green
            elif status == 'Idle':
                return [255, 165, 0, 160]  # Orange
            elif status == 'Stopped':
                return [255, 0, 0, 160]  # Red
            else:
                return [128, 128, 128, 160]  # Gray

        df['color'] = df['status'].apply(get_color)

        # Fetch mileage based on current month's data from intangles table
        # Mileage = Current Month Distance / (Current Month Engine Hours × 6 liters/hour)
        # Also fetch latest fuel_amount for each vehicle
        try:
            fuel_query = """
                WITH monthly_data AS (
                    SELECT
                        plate,
                        UPPER(REPLACE(REPLACE(plate, ' ', ''), '-', '')) as normalized_plate,
                        MIN(odometer_km) as month_start_odometer,
                        MAX(odometer_km) as month_end_odometer,
                        MIN(engine_hours) as month_start_engine_hours,
                        MAX(engine_hours) as month_end_engine_hours
                    FROM intangles
                    WHERE odometer_km IS NOT NULL
                        AND engine_hours IS NOT NULL
                        AND recorded_at >= DATE_TRUNC('month', CURRENT_DATE)
                    GROUP BY plate, UPPER(REPLACE(REPLACE(plate, ' ', ''), '-', ''))
                ),
                latest_fuel AS (
                    SELECT DISTINCT ON (plate)
                        plate,
                        UPPER(REPLACE(REPLACE(plate, ' ', ''), '-', '')) as normalized_plate,
                        fuel_amount
                    FROM intangles
                    WHERE fuel_amount IS NOT NULL
                    ORDER BY plate, recorded_at DESC
                )
                SELECT
                    COALESCE(m.plate, f.plate) as plate,
                    COALESCE(m.normalized_plate, f.normalized_plate) as normalized_plate,
                    (m.month_end_odometer - m.month_start_odometer) as distance_km,
                    (m.month_end_engine_hours - m.month_start_engine_hours) as engine_hours_used,
                    CASE
                        WHEN (m.month_end_engine_hours - m.month_start_engine_hours) > 0
                             AND (m.month_end_odometer - m.month_start_odometer) > 0
                        THEN (m.month_end_odometer - m.month_start_odometer) / ((m.month_end_engine_hours - m.month_start_engine_hours) * 6)
                        ELSE NULL
                    END as fuel_economy_kmpl,
                    f.fuel_amount as current_fuel
                FROM monthly_data m
                FULL OUTER JOIN latest_fuel f ON m.normalized_plate = f.normalized_plate
                WHERE (m.month_end_odometer - m.month_start_odometer) > 0
                   OR (m.month_end_engine_hours - m.month_start_engine_hours) > 0
                   OR f.fuel_amount IS NOT NULL;
            """
            fuel_df = pd.read_sql_query(fuel_query, connection)

            # Normalize vehicle numbers in both dataframes for matching
            if not fuel_df.empty:
                # Create normalized vehicle_no in df
                df['normalized_vehicle_no'] = df['vehicle_no'].apply(
                    lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
                )
                # Merge on normalized vehicle numbers
                fuel_df_merge = fuel_df[['normalized_plate', 'fuel_economy_kmpl', 'current_fuel']].copy()
                fuel_df_merge.columns = ['normalized_vehicle_no', 'fuel_economy_kmpl', 'current_fuel']
                df = df.merge(fuel_df_merge, on='normalized_vehicle_no', how='left')
                # Drop the normalized column
                df = df.drop(columns=['normalized_vehicle_no'])
            else:
                df['fuel_economy_kmpl'] = None
                df['current_fuel'] = None
        except Exception as e:
            df['fuel_economy_kmpl'] = None
            df['current_fuel'] = None

        # Fetch trip info from swift_trip_log
        try:
            trip_query = """
                WITH normalized_trips AS (
                    SELECT
                        CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                UPPER(REPLACE(SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1), ' ', ''))
                            ELSE UPPER(REPLACE(vehicle_no, ' ', ''))
                        END as normalized_vehicle_no,
                        trip_status,
                        route,
                        new_party_name as party,
                        loading_date,
                        driver_name,
                        driver_phone_no,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                CASE
                                    WHEN vehicle_no LIKE '% %' THEN
                                        UPPER(REPLACE(SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1), ' ', ''))
                                    ELSE UPPER(REPLACE(vehicle_no, ' ', ''))
                                END
                            ORDER BY loading_date DESC NULLS LAST
                        ) as rn
                    FROM swift_trip_log
                    WHERE vehicle_no IS NOT NULL
                )
                SELECT normalized_vehicle_no, trip_status, route, party, loading_date, driver_name, driver_phone_no
                FROM normalized_trips
                WHERE rn = 1
            """
            trip_df = pd.read_sql_query(trip_query, connection)

            if len(trip_df) > 0:
                df['normalized_vehicle_no'] = df['vehicle_no'].apply(
                    lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
                )
                df = df.merge(trip_df, on='normalized_vehicle_no', how='left')
                df = df.drop(columns=['normalized_vehicle_no'])
            else:
                df['trip_status'] = None
                df['route'] = None
                df['party'] = None
                df['loading_date'] = None
                df['driver_name'] = None
                df['driver_phone_no'] = None
        except Exception as e:
            df['trip_status'] = None
            df['route'] = None
            df['party'] = None
            df['loading_date'] = None
            df['driver_name'] = None
            df['driver_phone_no'] = None

        # Add owner name from Excel file
        try:
            owner_df = load_owner_mapping()
            df['normalized_vehicle_no'] = df['vehicle_no'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            df = df.merge(owner_df, on='normalized_vehicle_no', how='left')
            # Apply custom owner mappings
            df['owner_name'] = df.apply(
                lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no'], row['owner_name']),
                axis=1
            )
            df = df.drop(columns=['normalized_vehicle_no'], errors='ignore')
            df['owner_name'] = df['owner_name'].fillna('-')
        except Exception as e:
            df['owner_name'] = '-'

        return df

    except psycopg2.OperationalError as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("Please check your database connection and try refreshing the page.")
        return pd.DataFrame()
    except psycopg2.DatabaseError as e:
        st.error(f"Database error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error loading data: {str(e)}")
        return pd.DataFrame()
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

def show_overview_metrics(df):
    """Display overview metrics"""
    # First row of metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    total_vehicles = len(df)
    moving = len(df[df['status'] == 'Moving'])
    idle = len(df[df['status'] == 'Idle'])
    stopped = len(df[df['status'] == 'Stopped'])
    avg_speed = float(df['speed'].mean()) if 'speed' in df.columns else 0

    # Calculate vehicle bifurcation by owner
    own_vehicles = len(df[df['owner_name'] == 'Own Vehicle']) if 'owner_name' in df.columns else 0
    swaraj_vehicles = len(df[df['owner_name'].str.contains('SWARAJ', case=False, na=False)]) if 'owner_name' in df.columns else 0
    ranjeet_vehicles = len(df[df['owner_name'] == 'Ranjeet Singh Logistics']) if 'owner_name' in df.columns else 0
    other_vehicles = total_vehicles - own_vehicles - swaraj_vehicles - ranjeet_vehicles

    with col1:
        st.metric(
            label="🚛 Total Vehicles",
            value=f"{total_vehicles:,}"
        )
        # Show bifurcation below total
        st.markdown(f"""
        <div style="font-size: 1.1rem; color: #888; margin-top: -10px; line-height: 1.4;">
            Own: {own_vehicles} | Swaraj: {swaraj_vehicles} | Ranjeet: {ranjeet_vehicles} | Others: {other_vehicles}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.metric(
            label="🟢 Moving",
            value=f"{moving:,}"
        )

    with col3:
        st.metric(
            label="🟠 Idle",
            value=f"{idle:,}"
        )

    with col4:
        st.metric(
            label="🔴 Stopped",
            value=f"{stopped:,}"
        )

    with col5:
        st.metric(
            label="📊 Avg Speed",
            value=f"{avg_speed:.0f} km/h"
        )

def show_map(df):
    """Display vehicle tracking map using Folium"""

    if len(df) == 0:
        st.warning("No vehicle data available")
        return

    # Calculate center point
    center_lat = float(df['latitude'].mean())
    center_lon = float(df['longitude'].mean())

    # Create Folium map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )

    # Add markers for each vehicle
    for idx, row in df.iterrows():
        # Determine color based on status
        if row['status'] == 'Moving':
            color = 'green'
            icon = 'play'
        elif row['status'] == 'Idle':
            color = 'orange'
            icon = 'pause'
        elif row['status'] == 'Stopped':
            color = 'red'
            icon = 'stop'
        else:
            color = 'gray'
            icon = 'question'

        # Create popup content with trip info
        # Show Speed for Moving, Idle Time for Idle, Stopped Time for Stopped
        if row['status'] == 'Moving':
            speed_or_time_label = "Speed"
            speed_or_time_value = f"{row['speed']} km/h"
        elif row['status'] == 'Idle':
            speed_or_time_label = "Idle Time"
            speed_or_time_value = row.get('idle_time', '-') if pd.notna(row.get('idle_time')) else '-'
        elif row['status'] == 'Stopped':
            speed_or_time_label = "Stopped Time"
            # Calculate stopped time from Last Update Time (date_time) to current time
            if pd.notna(row.get('date_time')):
                try:
                    last_update = pd.to_datetime(row['date_time'])
                    stopped_seconds = (datetime.now() - last_update).total_seconds()
                    stopped_hours = stopped_seconds / 3600
                    if stopped_hours >= 24:
                        days = int(stopped_hours // 24)
                        hours = int(stopped_hours % 24)
                        speed_or_time_value = f"{days}d {hours}h"
                    else:
                        hours = int(stopped_hours)
                        minutes = int((stopped_seconds % 3600) / 60)
                        speed_or_time_value = f"{hours}h {minutes}m"
                except:
                    speed_or_time_value = '-'
            else:
                speed_or_time_value = '-'
        else:
            speed_or_time_label = "Speed"
            speed_or_time_value = f"{row['speed']} km/h"

        # Get trip info with safe defaults
        trip_status = row.get('trip_status', '-') if pd.notna(row.get('trip_status')) else '-'
        route = row.get('route', '-') if pd.notna(row.get('route')) else '-'
        party = row.get('party', '-') if pd.notna(row.get('party')) else '-'
        loading_date = row.get('loading_date', '-')
        if pd.notna(loading_date) and loading_date != '-':
            loading_date = pd.to_datetime(loading_date).strftime('%d-%b-%Y') if hasattr(loading_date, 'strftime') or isinstance(loading_date, str) else str(loading_date)[:10]
        else:
            loading_date = '-'
        driver_name = row.get('driver_name', '-') if pd.notna(row.get('driver_name')) else '-'
        driver_phone = row.get('driver_phone_no', '-') if pd.notna(row.get('driver_phone_no')) else '-'
        owner_name = row.get('owner_name', '-') if pd.notna(row.get('owner_name')) else '-'

        popup_html = f"""
        <div style="font-family: Arial; width: 280px;">
            <h4 style="margin: 0; color: {color};">🚛 {row['vehicle_no']}</h4>
            <hr style="margin: 5px 0;">
            <p style="margin: 3px 0;"><b>Status:</b> <span style="color: {color};">{row['status']}</span></p>
            <p style="margin: 3px 0;"><b>{speed_or_time_label}:</b> {speed_or_time_value}</p>
            <p style="margin: 3px 0;"><b>Location:</b> {row['location']}</p>
            <hr style="margin: 5px 0; border-style: dashed;">
            <p style="margin: 3px 0;"><b>Trip Status:</b> {trip_status}</p>
            <p style="margin: 3px 0;"><b>Route:</b> {route}</p>
            <p style="margin: 3px 0;"><b>Party:</b> {party}</p>
            <p style="margin: 3px 0;"><b>Loading Date:</b> {loading_date}</p>
            <p style="margin: 3px 0;"><b>Driver:</b> {driver_name}</p>
            <p style="margin: 3px 0;"><b>Driver Phone:</b> {driver_phone}</p>
            <p style="margin: 3px 0;"><b>Owner Name:</b> {owner_name}</p>
        </div>
        """

        # Add marker
        folium.Marker(
            location=[float(row['latitude']), float(row['longitude'])],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['vehicle_no']} - {row['status']}",
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m)

    # Display map
    st_folium(m, width=1400, height=600, returned_objects=[])

def show_vehicle_list(df):
    """Display vehicle list with details"""
    import streamlit.components.v1 as components

    st.subheader("🚛 Live Vehicle Details")

    # Prepare display dataframe - check if fuel_economy_kmpl exists
    if 'fuel_economy_kmpl' in df.columns:
        display_df = df[[
            'vehicle_no', 'status', 'speed', 'fuel_economy_kmpl', 'current_fuel', 'idle_time', 'location',
            'ignition', 'odometer', 'date_time', 'latitude', 'longitude'
        ]].copy()

        display_df.columns = [
            'Vehicle No', 'Status', 'Speed (km/h)', 'Mileage', 'Current Fuel (L)', 'Idle Time', 'Location',
            'Ignition', 'Odometer (km)', 'Last Update Time', 'Latitude', 'Longitude'
        ]
    else:
        display_df = df[[
            'vehicle_no', 'status', 'speed', 'idle_time', 'location',
            'ignition', 'odometer', 'date_time', 'latitude', 'longitude'
        ]].copy()

        display_df.columns = [
            'Vehicle No', 'Status', 'Speed (km/h)', 'Idle Time', 'Location',
            'Ignition', 'Odometer (km)', 'Last Update Time', 'Latitude', 'Longitude'
        ]

    # Format columns
    display_df['Ignition'] = display_df['Ignition'].apply(lambda x: 'ON' if x == 1 else 'OFF')
    display_df['Odometer (km)'] = display_df['Odometer (km)'].apply(lambda x: f"{float(x):,.2f}" if pd.notna(x) else "N/A")
    display_df['Idle Time'] = display_df['Idle Time'].apply(lambda x: x if pd.notna(x) else '-')
    display_df['Speed (km/h)'] = display_df['Speed (km/h)'].apply(lambda x: f"{x}" if pd.notna(x) else "0")
    if 'Mileage' in display_df.columns:
        display_df['Mileage'] = display_df['Mileage'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
    if 'Current Fuel (L)' in display_df.columns:
        display_df['Current Fuel (L)'] = display_df['Current Fuel (L)'].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "-")

    # Build HTML table with clickable location cells
    num_rows = len(display_df)
    table_height = min(600, 60 + num_rows * 45)

    # Rename Location to Live Location
    display_df = display_df.rename(columns={'Location': 'Live Location'})

    # Columns to display (excluding lat/lon from visible columns)
    visible_columns = [col for col in display_df.columns if col not in ['Latitude', 'Longitude']]

    html_table = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background-color: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        .vehicle-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .vehicle-table th {{
            background-color: rgba(80, 80, 80, 0.9);
            color: #FFFFFF;
            font-weight: bold;
            text-align: center;
            border: 1px solid rgba(200, 200, 200, 0.5);
            padding: 10px;
            font-size: 14px;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        .vehicle-table td {{
            border: 1px solid rgba(200, 200, 200, 0.5);
            padding: 8px;
            color: #000000;
        }}
        .vehicle-table tr:hover td {{
            background-color: rgba(33, 150, 243, 0.15);
        }}
        .vehicle-table .vehicle-no {{
            text-align: center;
            font-weight: 600;
        }}
        .vehicle-table .center {{
            text-align: center;
        }}
        .vehicle-table .left {{
            text-align: left;
        }}
        .moving {{
            background-color: rgba(144, 238, 144, 0.5);
        }}
        .idle {{
            background-color: rgba(255, 213, 128, 0.6);
        }}
        .stopped {{
            background-color: rgba(220, 120, 130, 0.75);
        }}
        .location-link {{
            color: #FFFFFF;
            text-decoration: underline;
            cursor: pointer;
            font-weight: 500;
        }}
        .location-link:hover {{
            text-decoration: underline;
            color: #E0E0E0;
        }}
        .table-container {{
            max-height: {table_height - 20}px;
            overflow-y: auto;
            overflow-x: auto;
        }}
        /* Modal styles */
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.6);
        }}
        .modal-content {{
            background-color: #fefefe;
            margin: 5% auto;
            padding: 0;
            border-radius: 10px;
            width: 80%;
            max-width: 800px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}
        .modal-header {{
            padding: 15px 20px;
            background-color: #1a73e8;
            color: white;
            border-radius: 10px 10px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .modal-header h3 {{
            margin: 0;
            font-size: 18px;
        }}
        .close {{
            color: white;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #ddd;
        }}
        .modal-body {{
            padding: 0;
        }}
        .modal-body iframe {{
            width: 100%;
            height: 400px;
            border: none;
        }}
        .modal-footer {{
            padding: 15px 20px;
            text-align: center;
            background-color: #1a73e8;
            border-radius: 0 0 10px 10px;
            position: relative;
            z-index: 10;
        }}
        .modal-footer a {{
            text-decoration: none;
            font-weight: 600;
            font-size: 16px;
            padding: 10px 25px;
            background-color: #fff;
            color: #1a73e8;
            border-radius: 5px;
            display: inline-block;
        }}
        .modal-footer a:hover {{
            background-color: #e8f0fe;
        }}
    </style>
    </head>
    <body>

    <!-- Modal for map popup -->
    <div id="mapModal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">📍 Vehicle Location</h3>
                <span class="close" onclick="closeModal()">&times;</span>
            </div>
            <div class="modal-body">
                <iframe id="mapFrame" src=""></iframe>
            </div>
            <div class="modal-footer">
                <a id="googleMapsLink" href="" target="_blank">🗺️ Open in Google Maps</a>
            </div>
        </div>
    </div>

    <div class="table-container">
    <table class="vehicle-table">
    <thead><tr>
    '''

    # Add headers
    for col in visible_columns:
        html_table += f'<th>{col}</th>'
    html_table += '</tr></thead><tbody>'

    # Add rows
    for idx, row in display_df.iterrows():
        status = row['Status']
        status_class = ''
        if status == 'Moving':
            status_class = 'moving'
        elif status == 'Idle':
            status_class = 'idle'
        elif status == 'Stopped':
            status_class = 'stopped'

        lat = row['Latitude']
        lon = row['Longitude']

        html_table += f'<tr class="{status_class}">'
        for col in visible_columns:
            val = row[col]
            if col == 'Vehicle No':
                html_table += f'<td class="vehicle-no">{val}</td>'
            elif col == 'Live Location':
                # Make location clickable with map popup
                if pd.notna(lat) and pd.notna(lon) and lat != 0 and lon != 0:
                    vehicle_no = row['Vehicle No']
                    html_table += f'''<td class="left">
                        <a class="location-link" href="javascript:void(0)"
                           onclick="openMapPopup({lat}, {lon}, '{vehicle_no}', '{val}')">{val}</a>
                    </td>'''
                else:
                    html_table += f'<td class="left">{val}</td>'
            elif col in ['Speed (km/h)', 'Odometer (km)', 'Mileage', 'Idle Time', 'Ignition', 'Status']:
                html_table += f'<td class="center">{val}</td>'
            else:
                html_table += f'<td class="left">{val}</td>'
        html_table += '</tr>'

    html_table += '''</tbody></table></div>

    <script>
        function openMapPopup(lat, lon, vehicleNo, location) {
            var modal = document.getElementById("mapModal");
            var mapFrame = document.getElementById("mapFrame");
            var modalTitle = document.getElementById("modalTitle");
            var googleMapsLink = document.getElementById("googleMapsLink");

            // Set modal title
            modalTitle.innerHTML = "📍 " + vehicleNo + " - " + location;

            // Set iframe source to OpenStreetMap embed
            var osmUrl = "https://www.openstreetmap.org/export/embed.html?bbox=" +
                         (lon - 0.01) + "," + (lat - 0.01) + "," + (lon + 0.01) + "," + (lat + 0.01) +
                         "&layer=mapnik&marker=" + lat + "," + lon;
            mapFrame.src = osmUrl;

            // Set Google Maps link
            googleMapsLink.href = "https://www.google.com/maps?q=" + lat + "," + lon;

            modal.style.display = "block";
        }

        function closeModal() {
            var modal = document.getElementById("mapModal");
            modal.style.display = "none";
            document.getElementById("mapFrame").src = "";
        }

        // Close modal when clicking outside
        window.onclick = function(event) {
            var modal = document.getElementById("mapModal");
            if (event.target == modal) {
                closeModal();
            }
        }
    </script>
    </body></html>'''

    components.html(html_table, height=table_height + 50, scrolling=True)

    # Download button for the data
    download_df = display_df[visible_columns].copy()
    csv_data = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Vehicle Data",
        data=csv_data,
        file_name=f"live_vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        key="download_vehicle_data"
    )

def create_sparkline_svg(values, dates):
    """Create an SVG line chart sparkline with points, labels, and tooltips"""
    if not values or len(values) == 0 or not dates or len(dates) == 0:
        return '-'

    # Filter out None values for scaling
    valid_values = [v for v in values if v is not None]
    if len(valid_values) == 0:
        return '-'

    # SVG dimensions - wider to show KM labels
    n = len(values)
    width = max(280, n * 35)  # Dynamic width based on number of points
    height = 55
    padding_x = 15
    padding_top = 18  # More space at top for labels
    padding_bottom = 12

    min_val = min(valid_values) if valid_values else 0
    max_val = max(valid_values) if valid_values else 1

    if max_val == min_val:
        max_val = min_val + 1  # Avoid division by zero

    # Calculate points
    points = []
    point_data = []

    for i, (v, d) in enumerate(zip(values, dates)):
        x = padding_x + (i / max(n - 1, 1)) * (width - 2 * padding_x) if n > 1 else width / 2
        if v is not None and v >= 0:
            y = padding_top + (1 - (v - min_val) / (max_val - min_val)) * (height - padding_top - padding_bottom)
        else:
            y = height - padding_bottom
            v = 0
        points.append((x, y))
        point_data.append((d, v))

    # Create SVG path
    path_d = ' '.join([f"{'M' if i == 0 else 'L'} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points)])

    # Build SVG with line and points
    svg = f'''<svg width="{width}" height="{height}" style="vertical-align: middle;">
        <path d="{path_d}" fill="none" stroke="#4A90D9" stroke-width="2"/>'''

    # Add circles with KM labels and tooltips for each point
    for i, ((x, y), (date, km)) in enumerate(zip(points, point_data)):
        date_str = date.strftime('%d-%b') if hasattr(date, 'strftime') else str(date)

        # Add circle
        svg += f'''
        <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#4A90D9" stroke="#fff" stroke-width="1.5">
            <title>{date_str}: {km:.0f} km</title>
        </circle>'''

        # Add KM label above the point
        label_y = y - 8  # Position above the circle
        svg += f'''
        <text x="{x:.1f}" y="{label_y:.1f}" font-size="9" fill="#FFD700" text-anchor="middle" font-weight="bold">{km:.0f}</text>'''

    # Add total KM text at bottom right
    total_km = sum(v for v in values if v is not None)
    svg += f'''
        <text x="{width - 5}" y="{height - 2}" font-size="10" fill="#4FC3F7" text-anchor="end" font-weight="bold">Total: {total_km:.0f}km</text>
    </svg>'''

    return svg

@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes, silent refresh
def load_vehicle_load_details():
    """Load ALL vehicles and map with their load details from swift_trip_log"""
    connection = None
    try:
        connection = get_database_connection()

        # Query to get ALL vehicles from fvts_vehicles and LEFT JOIN with latest load details
        # Normalize vehicle number format: "0167 NL01AH" -> "NL01AH0167"
        query = """
            WITH vehicle_today_km AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    COALESCE(MAX(odometer) - MIN(odometer), 0) as today_km_traveled
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND DATE(date_time) = CURRENT_DATE
                    AND odometer IS NOT NULL
                GROUP BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
            ),
            vehicle_yesterday_km AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    COALESCE(MAX(odometer) - MIN(odometer), 0) as yesterday_km_traveled
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND DATE(date_time) = CURRENT_DATE - INTERVAL '1 day'
                    AND odometer IS NOT NULL
                GROUP BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
            ),
            vehicle_month_km AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    COALESCE(MAX(odometer) - MIN(odometer), 0) as month_km_traveled
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND DATE_TRUNC('month', date_time) = DATE_TRUNC('month', CURRENT_DATE)
                    AND odometer IS NOT NULL
                GROUP BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
            ),
            all_vehicles AS (
                SELECT DISTINCT ON (
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
                )
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    location as current_location,
                    speed as current_speed,
                    ignition as current_ignition
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                ORDER BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END),
                    date_time DESC
            ),
            latest_trip_per_vehicle AS (
                SELECT DISTINCT ON (normalized_vehicle_no)
                    normalized_vehicle_no as vehicle_no,
                    trip_status,
                    route,
                    onward_route,
                    party,
                    loading_date,
                    unloading_date,
                    distance,
                    driver_name,
                    driver_code,
                    driver_phone_no,
                    created_at
                FROM (
                    SELECT
                        vehicle_no as original_vehicle_no,
                        UPPER(CASE
                            -- Format: "0167 NL01AH" -> "NL01AH0167"
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            -- Format: "2081NL01AJ" or "2081NL01AJ-S" (starts with 4 digits)
                            WHEN LENGTH(vehicle_no) >= 9
                                AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                                AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                                CASE
                                    WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                        SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                    ELSE
                                        SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                END
                            ELSE vehicle_no
                        END) as normalized_vehicle_no,
                        CASE
                            WHEN trip_status IS NULL OR trip_status = '' THEN 'Loaded'
                            ELSE trip_status
                        END as trip_status,
                        route,
                        onward_route,
                        party,
                        loading_date,
                        unloading_date,
                        COALESCE(distance, 0) as distance,
                        driver_name,
                        driver_code,
                        driver_phone_no,
                        created_at
                    FROM swift_trip_log
                    WHERE vehicle_no IS NOT NULL
                ) normalized
                ORDER BY normalized_vehicle_no, loading_date DESC NULLS LAST, created_at DESC
            )
            SELECT
                av.vehicle_no,
                ltpv.trip_status,
                ltpv.route,
                ltpv.onward_route,
                ltpv.party,
                ltpv.loading_date,
                ltpv.unloading_date,
                ltpv.distance,
                ltpv.driver_name,
                ltpv.driver_code,
                ltpv.driver_phone_no,
                COALESCE(vtk.today_km_traveled, 0) as gps_today_km,
                COALESCE(vyk.yesterday_km_traveled, 0) as gps_yesterday_km,
                COALESCE(vmk.month_km_traveled, 0) as gps_month_km,
                CASE
                    WHEN ltpv.vehicle_no IS NULL THEN true
                    ELSE false
                END as missing_load_data
            FROM all_vehicles av
            LEFT JOIN latest_trip_per_vehicle ltpv ON av.vehicle_no = ltpv.vehicle_no
            LEFT JOIN vehicle_today_km vtk ON av.vehicle_no = vtk.vehicle_no
            LEFT JOIN vehicle_yesterday_km vyk ON av.vehicle_no = vyk.vehicle_no
            LEFT JOIN vehicle_month_km vmk ON av.vehicle_no = vmk.vehicle_no
            ORDER BY av.vehicle_no;
        """

        df = pd.read_sql_query(query, connection)

        # Filter out excluded vehicles
        if EXCLUDED_VEHICLES:
            df['normalized_vehicle'] = df['vehicle_no'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            df = df[~df['normalized_vehicle'].isin(EXCLUDED_VEHICLES)]
            df = df.drop(columns=['normalized_vehicle'])

        # Convert timestamps
        for col in ['lr_date', 'trip_start_date', 'trip_end_date', 'loading_date', 'unloading_date', 'created_at']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')

        # Fetch owner name from party name map Excel file
        try:
            excel_path = os.path.join(os.path.dirname(__file__), 'party name map.xlsx')
            owner_df = pd.read_excel(excel_path)
            # Normalize RegistrationNo for matching
            owner_df['normalized_vehicle_no'] = owner_df['RegistrationNo'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            # Apply owner name mapping logic
            owner_df['owner_name'] = owner_df['PartyName'].apply(
                lambda x: 'Own Vehicle' if x in ['Swift Road Link Pvt. Ltd.', 'Nishant Saini Associates'] else x
            )
            owner_df = owner_df[['normalized_vehicle_no', 'owner_name']]
            # Normalize vehicle_no in df for matching
            df['normalized_vehicle_no_temp'] = df['vehicle_no'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            df = df.merge(owner_df, left_on='normalized_vehicle_no_temp', right_on='normalized_vehicle_no', how='left')
            # Apply custom owner mappings
            df['owner_name'] = df.apply(
                lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no_temp'], row['owner_name']),
                axis=1
            )
            df = df.drop(columns=['normalized_vehicle_no', 'normalized_vehicle_no_temp'], errors='ignore')
        except Exception as e:
            # If Excel file doesn't exist or read fails, just add empty owner_name column
            df['owner_name'] = None

        # Fetch daily GPS KM data for sparkline trends
        # Using LAG to calculate daily KM as difference between consecutive days' end odometer
        daily_km_query = """
            WITH daily_odometer AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    DATE(date_time) as km_date,
                    MAX(odometer) as end_odometer,
                    MIN(odometer) as start_odometer
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND odometer IS NOT NULL
                    AND odometer > 0
                    AND date_time >= CURRENT_DATE - INTERVAL '30 days'
                GROUP BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END),
                    DATE(date_time)
            ),
            daily_with_prev AS (
                SELECT
                    vehicle_no,
                    km_date,
                    end_odometer,
                    start_odometer,
                    LAG(end_odometer) OVER (PARTITION BY vehicle_no ORDER BY km_date) as prev_end_odometer
                FROM daily_odometer
            )
            SELECT
                vehicle_no,
                km_date,
                LEAST(
                    CASE
                        WHEN prev_end_odometer IS NOT NULL AND start_odometer >= prev_end_odometer THEN
                            GREATEST(end_odometer - prev_end_odometer, 0)
                        WHEN prev_end_odometer IS NULL THEN
                            GREATEST(end_odometer - start_odometer, 0)
                        ELSE
                            GREATEST(end_odometer - start_odometer, 0)
                    END,
                    1000
                ) as daily_km
            FROM daily_with_prev
            WHERE end_odometer IS NOT NULL
            ORDER BY vehicle_no, km_date;
        """
        daily_km_df = pd.read_sql_query(daily_km_query, connection)
        daily_km_df['km_date'] = pd.to_datetime(daily_km_df['km_date'])

        # Create a dictionary for quick lookup: {vehicle_no: {date: km}}
        km_lookup = {}
        for _, row in daily_km_df.iterrows():
            vno = row['vehicle_no']
            if vno not in km_lookup:
                km_lookup[vno] = {}
            km_lookup[vno][row['km_date'].date()] = row['daily_km']

        # Create km_trend sparkline for each vehicle from loading_date to today
        today = datetime.now().date()

        def get_km_trend(row):
            vehicle_no = row['vehicle_no']
            loading_date = row.get('loading_date')

            if pd.isna(loading_date) or vehicle_no not in km_lookup:
                return '-', 0

            start_date = loading_date.date()
            # Limit to last 10 days from loading date for display
            num_days = min((today - start_date).days + 1, 10)

            if num_days <= 0:
                return '-', 0

            # Get daily KM values and dates from loading_date to today (or max 10 days)
            km_values = []
            km_dates = []
            for i in range(num_days):
                target_date = start_date + timedelta(days=i)
                if target_date > today:
                    break
                km = km_lookup[vehicle_no].get(target_date, 0)
                km_values.append(km)
                km_dates.append(target_date)

            if not km_values:
                return '-', 0

            # Create SVG sparkline with points and tooltips
            total_km = sum(km_values)
            return create_sparkline_svg(km_values, km_dates), total_km

        # Apply function and split results into two columns
        df[['km_trend', 'trend_total_km']] = df.apply(
            lambda row: pd.Series(get_km_trend(row)), axis=1
        )

        # Handle cases where get_km_trend returned just '-'
        df['trend_total_km'] = pd.to_numeric(df['trend_total_km'], errors='coerce').fillna(0)

        # Sort by trend_total_km in descending order
        df = df.sort_values('trend_total_km', ascending=False).reset_index(drop=True)

        # Calculate days with >350 km in current month
        current_month_start = datetime.now().replace(day=1).date()

        def count_days_above_350(vehicle_no):
            if vehicle_no not in km_lookup:
                return 0
            count = 0
            for date, km in km_lookup[vehicle_no].items():
                if date >= current_month_start and km > 350:
                    count += 1
            return count

        df['days_above_350'] = df['vehicle_no'].apply(count_days_above_350)

        # Convert distance to numeric
        if 'distance' in df.columns:
            df['distance'] = pd.to_numeric(df['distance'], errors='coerce').fillna(0)

        # Calculate Current Trip Status
        now = datetime.now()

        def calculate_current_trip_status(row):
            # If no loading date, return empty
            if pd.isna(row.get('loading_date')):
                return '-'

            # Check if unloading_date exists and is in the past
            if pd.notna(row.get('unloading_date')) and row['unloading_date'] < now:
                return 'Trip End'

            # Calculate Transit Time (TT) based on distance
            distance = row.get('distance', 0) or 0
            if distance <= 0:
                return '-'

            # TT = distance / 400 km per day
            tt_days = distance / 400

            # Calculate expected arrival date (TT date from loading_date)
            loading_date = row['loading_date']
            tt_date = loading_date + timedelta(days=tt_days)

            # Compare current date with TT date
            if now < tt_date:
                return 'Early'
            else:
                return 'Delay'

        df['current_trip_status'] = df.apply(calculate_current_trip_status, axis=1)

        # Calculate trip duration for completed trips
        if 'trip_start_date' in df.columns and 'trip_end_date' in df.columns:
            df['trip_duration_hours'] = (df['trip_end_date'] - df['trip_start_date']).dt.total_seconds() / 3600
            df['trip_duration_hours'] = df['trip_duration_hours'].apply(lambda x: round(x, 1) if pd.notna(x) else None)

        return df

    except Exception as e:
        st.error(f"Error loading load details: {str(e)}")
        return pd.DataFrame()
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

def show_load_details():
    """Display load details table"""
    st.subheader("🚚 Load Details")

    load_df = load_vehicle_load_details()

    if len(load_df) == 0:
        st.warning("No load details available")
        return

    # Add search and filter options
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        vehicle_filter = st.text_input("🔍 Search Vehicle No", "", key="load_vehicle_search")

    with col2:
        party_options = ['All'] + list(load_df['party'].dropna().unique()) if 'party' in load_df.columns else ['All']
        party_filter = st.selectbox("Filter by Party", party_options, key="load_party_filter")

    with col3:
        status_options = ['All'] + list(load_df['trip_status'].dropna().unique()) if 'trip_status' in load_df.columns else ['All']
        status_filter = st.selectbox("Filter by Trip Status", status_options, key="load_status_filter")

    with col4:
        current_trip_status_options = ['All'] + list(load_df['current_trip_status'].dropna().unique()) if 'current_trip_status' in load_df.columns else ['All']
        current_trip_status_filter = st.selectbox("Filter by Current Trip Status", current_trip_status_options, key="load_current_trip_status_filter")

    # Apply filters
    filtered_load_df = load_df.copy()

    if vehicle_filter:
        filtered_load_df = filtered_load_df[
            filtered_load_df['vehicle_no'].str.contains(vehicle_filter, case=False, na=False)
        ]

    if party_filter != 'All':
        filtered_load_df = filtered_load_df[filtered_load_df['party'] == party_filter]

    if status_filter != 'All':
        filtered_load_df = filtered_load_df[filtered_load_df['trip_status'] == status_filter]

    if current_trip_status_filter != 'All':
        filtered_load_df = filtered_load_df[filtered_load_df['current_trip_status'] == current_trip_status_filter]

    # Merge driver_name and driver_code into single column: "Name (Code)"
    if 'driver_name' in filtered_load_df.columns and 'driver_code' in filtered_load_df.columns:
        filtered_load_df['driver'] = filtered_load_df.apply(
            lambda row: f"{row['driver_name']} ({row['driver_code']})"
            if pd.notna(row['driver_name']) and pd.notna(row['driver_code'])
            and str(row['driver_name']) not in ['-', 'None', 'nan', '']
            and str(row['driver_code']) not in ['-', 'None', 'nan', '']
            else (str(row['driver_name']) if pd.notna(row['driver_name']) and str(row['driver_name']) not in ['-', 'None', 'nan', ''] else '-'),
            axis=1
        )

    # Prepare display dataframe with available columns
    display_cols = [
        'vehicle_no', 'trip_status', 'current_trip_status', 'route', 'onward_route', 'party', 'loading_date',
        'driver', 'driver_phone_no',
        'gps_today_km', 'gps_yesterday_km', 'gps_month_km', 'days_above_350', 'km_trend', 'owner_name'
    ]

    # Filter columns that exist
    display_cols = [col for col in display_cols if col in filtered_load_df.columns]
    display_df = filtered_load_df[display_cols].copy()

    # Format columns
    column_mapping = {
        'vehicle_no': 'Vehicle No',
        'trip_status': 'Trip Status',
        'current_trip_status': 'Current Trip Status',
        'route': 'Route',
        'onward_route': 'Onward Route',
        'party': 'Party',
        'loading_date': 'Loading Date',
        'driver': 'Driver',
        'driver_phone_no': 'Driver Phone',
        'gps_today_km': 'GPS Today KM',
        'gps_yesterday_km': 'GPS Yesterday KM',
        'gps_month_km': 'GPS Month KM',
        'days_above_350': 'Days >350 KM',
        'km_trend': 'KM Trend (Loading to Today)',
        'owner_name': 'Owner Name'
    }

    display_df = display_df.rename(columns=column_mapping)

    # Replace all None/NaN values with '-' for better display
    display_df = display_df.fillna('-')

    # Replace string 'None' with '-'
    display_df = display_df.replace('None', '-')
    display_df = display_df.replace('', '-')

    # Format date columns (only if they have valid dates)
    for col in display_df.columns:
        if 'Date' in col and col in filtered_load_df.columns:
            # Get the original column name before mapping
            orig_col = [k for k, v in column_mapping.items() if v == col][0]
            if orig_col in filtered_load_df.columns:
                display_df[col] = filtered_load_df[orig_col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) and x != '' and x != 'None' else '-'
                )

    # Format GPS KM columns
    if 'GPS Today KM' in display_df.columns and 'gps_today_km' in filtered_load_df.columns:
        display_df['GPS Today KM'] = filtered_load_df['gps_today_km'].apply(
            lambda x: f"{round(x, 2)}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    if 'GPS Yesterday KM' in display_df.columns and 'gps_yesterday_km' in filtered_load_df.columns:
        display_df['GPS Yesterday KM'] = filtered_load_df['gps_yesterday_km'].apply(
            lambda x: f"{round(x, 2)}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    if 'GPS Month KM' in display_df.columns and 'gps_month_km' in filtered_load_df.columns:
        display_df['GPS Month KM'] = filtered_load_df['gps_month_km'].apply(
            lambda x: f"{round(x, 2)}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    # Apply styling with dark mode support
    def style_row(row):
        # Dark mode friendly styling with white/light text
        base_style = 'font-size: 13px; border: 1px solid rgba(100, 100, 100, 0.3); padding: 8px; color: #FFFFFF;'

        # Color code by trip status with darker backgrounds for dark mode
        if 'Trip Status' in row.index:
            if row['Trip Status'] == 'Completed':
                bg_color = 'background-color: rgba(34, 139, 34, 0.25);'  # Dark green
            elif row['Trip Status'] == 'In Transit':
                bg_color = 'background-color: rgba(30, 144, 255, 0.25);'  # Dark blue
            elif row['Trip Status'] == 'Scheduled':
                bg_color = 'background-color: rgba(255, 140, 0, 0.25);'  # Dark orange
            else:
                bg_color = 'background-color: rgba(40, 40, 40, 0.3);'
        else:
            bg_color = 'background-color: rgba(40, 40, 40, 0.3);'

        styles = []
        for col in display_df.columns:
            if col == 'Current Trip Status':
                # Special coloring for Current Trip Status
                current_status = row.get('Current Trip Status', '-')
                if current_status == 'Trip End':
                    styles.append(base_style + 'background-color: rgba(128, 128, 128, 0.4); text-align: center; font-weight: 600;')
                elif current_status == 'Early':
                    styles.append(base_style + 'background-color: rgba(34, 139, 34, 0.4); text-align: center; font-weight: 600; color: #90EE90;')
                elif current_status == 'Delay':
                    styles.append(base_style + 'background-color: rgba(220, 20, 60, 0.4); text-align: center; font-weight: 600; color: #FF6B6B;')
                else:
                    styles.append(base_style + bg_color + 'text-align: center;')
            elif col in ['Trip Status', 'Current Speed (km/h)', 'Ignition Status']:
                styles.append(base_style + bg_color + 'text-align: center;')
            elif col == 'Vehicle No':
                styles.append(base_style + bg_color + 'text-align: center; font-weight: 700; color: #4FC3F7;')
            elif col in ['Driver Phone', 'Record Created', 'Loading Date']:
                styles.append(base_style + bg_color + 'text-align: center;')
            elif col in ['Route', 'Party', 'Driver', 'Owner Name']:
                styles.append(base_style + bg_color + 'text-align: left;')
            elif col == 'KM Trend (Loading to Today)':
                styles.append(base_style + bg_color + 'text-align: center; font-family: monospace; font-size: 14px;')
            else:
                styles.append(base_style + bg_color + 'text-align: left;')

        return styles

    # Check if KM Trend column has SVG content
    has_svg = 'KM Trend (Loading to Today)' in display_df.columns and display_df['KM Trend (Loading to Today)'].str.contains('<svg', na=False).any()

    if has_svg:
        import streamlit.components.v1 as components

        # Build HTML table to render SVG sparklines
        num_rows = len(display_df)
        table_height = min(600, 60 + num_rows * 45)  # Dynamic height based on rows

        html_table = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .load-details-table {{
                width: max-content;
                min-width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                table-layout: auto;
            }}
            .load-details-table th {{
                background-color: #1e1e1e;
                color: #FFFFFF;
                font-weight: bold;
                text-align: center;
                border: 1px solid #444;
                padding: 10px 6px;
                font-size: 12px;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .load-details-table td {{
                border: 1px solid #333;
                padding: 6px;
                color: #FFFFFF;
                background-color: #262730;
                overflow: visible;
                text-overflow: clip;
            }}
            .load-details-table tr:hover td {{
                background-color: rgba(66, 165, 245, 0.2);
            }}
            .load-details-table .vehicle-no {{
                text-align: center;
                font-weight: 700;
                color: #4FC3F7;
                min-width: 100px;
                white-space: nowrap;
            }}
            .load-details-table .center {{
                text-align: center;
                white-space: nowrap;
            }}
            .load-details-table .left {{
                text-align: left;
                min-width: 120px;
            }}
            .load-details-table .route-cell {{
                text-align: left;
                min-width: 250px;
                white-space: nowrap;
            }}
            .load-details-table .party-cell {{
                text-align: left;
                min-width: 220px;
                white-space: nowrap;
            }}
            .load-details-table .driver-cell {{
                text-align: left;
                min-width: 140px;
                white-space: nowrap;
            }}
            .load-details-table .owner-cell {{
                text-align: left;
                min-width: 150px;
                white-space: nowrap;
            }}
            .load-details-table .trend-cell {{
                text-align: center;
                padding: 4px;
                min-width: 320px;
                max-width: 400px;
            }}
            .load-details-table .early {{
                background-color: rgba(34, 139, 34, 0.4) !important;
                color: #90EE90;
                font-weight: 600;
            }}
            .load-details-table .delay {{
                background-color: rgba(220, 20, 60, 0.4) !important;
                color: #FF6B6B;
                font-weight: 600;
            }}
            .load-details-table .trip-end {{
                background-color: rgba(128, 128, 128, 0.4) !important;
                font-weight: 600;
            }}
            .table-container {{
                max-height: {table_height - 20}px;
                overflow-y: auto;
                overflow-x: auto;
            }}
        </style>
        </head>
        <body>
        <div class="table-container">
        <table class="load-details-table">
        <thead><tr>
        '''

        # Add headers
        for col in display_df.columns:
            html_table += f'<th>{col}</th>'
        html_table += '</tr></thead><tbody>'

        # Add rows
        for idx, row in display_df.iterrows():
            html_table += '<tr>'
            for col in display_df.columns:
                val = row[col]
                if col == 'Vehicle No':
                    html_table += f'<td class="vehicle-no">{val}</td>'
                elif col == 'KM Trend (Loading to Today)':
                    html_table += f'<td class="trend-cell">{val}</td>'
                elif col == 'Current Trip Status':
                    status_class = ''
                    if val == 'Early':
                        status_class = 'early'
                    elif val == 'Delay':
                        status_class = 'delay'
                    elif val == 'Trip End':
                        status_class = 'trip-end'
                    html_table += f'<td class="center {status_class}">{val}</td>'
                elif col == 'Route':
                    html_table += f'<td class="route-cell">{val}</td>'
                elif col == 'Onward Route':
                    html_table += f'<td class="route-cell">{val}</td>'
                elif col == 'Party':
                    html_table += f'<td class="party-cell">{val}</td>'
                elif col == 'Driver':
                    html_table += f'<td class="driver-cell">{val}</td>'
                elif col == 'Owner Name':
                    html_table += f'<td class="owner-cell">{val}</td>'
                elif col in ['Trip Status', 'Loading Date', 'Driver Phone', 'GPS Today KM', 'GPS Yesterday KM', 'GPS Month KM', 'Days >350 KM']:
                    html_table += f'<td class="center">{val}</td>'
                else:
                    html_table += f'<td class="left">{val}</td>'
            html_table += '</tr>'

        html_table += '</tbody></table></div></body></html>'

        components.html(html_table, height=table_height, scrolling=True)
    else:
        # Use regular dataframe display if no SVG
        styled_df = display_df.style.apply(style_row, axis=1)

        # Dark mode table styling
        styled_df = styled_df.set_table_styles([
            {'selector': 'thead th',
             'props': [('background-color', 'rgba(30, 30, 30, 0.95)'),
                       ('color', '#FFFFFF'),
                       ('font-weight', 'bold'),
                       ('text-align', 'center'),
                       ('border', '1px solid rgba(100, 100, 100, 0.5)'),
                       ('padding', '12px'),
                       ('font-size', '14px')]},
            {'selector': 'tbody tr:hover',
             'props': [('background-color', 'rgba(66, 165, 245, 0.2)')]},
            {'selector': 'tbody td',
             'props': [('color', '#FFFFFF')]},
        ])

        st.dataframe(
            styled_df,
            use_container_width=True,
            height=500
        )

    # Legend for Current Trip Status
    st.markdown("""
    **Current Trip Status Legend:**
    - 🟢 **Early**: Vehicle is ahead of schedule (current date < expected arrival date based on Transit Time)
    - 🔴 **Delay**: Vehicle is behind schedule (current date > expected arrival date based on Transit Time)
    - ⚫ **Trip End**: Trip has been completed (unloading date has passed)
    - Transit Time (TT) = Distance / 400 km per day
    """)

    # Download option
    csv = filtered_load_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Load Details (CSV)",
        data=csv,
        file_name=f"load_details_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False
    )

@st.cache_data(ttl=600, show_spinner=False)  # Cache for 10 minutes, silent refresh
def load_trip_km_by_days_data():
    """Load trip data with daily GPS KM from loading date"""
    connection = None
    try:
        connection = get_database_connection()

        # First get the base trip data with loading dates
        base_query = """
            WITH all_vehicles AS (
                SELECT DISTINCT ON (
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
                )
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as vehicle_no,
                    vehicle_no as original_vehicle_no
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                ORDER BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END),
                    date_time DESC
            ),
            latest_trip_per_vehicle AS (
                SELECT DISTINCT ON (normalized_vehicle_no)
                    normalized_vehicle_no as vehicle_no,
                    original_vehicle_no,
                    trip_status,
                    route,
                    party,
                    loading_date,
                    driver_name,
                    driver_code,
                    driver_phone_no
                FROM (
                    SELECT
                        vehicle_no as original_vehicle_no,
                        UPPER(CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            WHEN LENGTH(vehicle_no) >= 9
                                AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                                AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                                CASE
                                    WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                        SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                    ELSE
                                        SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                END
                            ELSE vehicle_no
                        END) as normalized_vehicle_no,
                        trip_status,
                        route,
                        party,
                        loading_date,
                        driver_name,
                        driver_code,
                        driver_phone_no
                    FROM swift_trip_log
                    WHERE vehicle_no IS NOT NULL
                ) normalized
                ORDER BY normalized_vehicle_no, loading_date DESC NULLS LAST
            )
            SELECT
                av.vehicle_no,
                av.original_vehicle_no as fvts_vehicle_no,
                ltpv.original_vehicle_no as trip_vehicle_no,
                ltpv.trip_status,
                ltpv.route,
                ltpv.loading_date,
                ltpv.driver_name,
                ltpv.driver_code,
                ltpv.driver_phone_no
            FROM all_vehicles av
            LEFT JOIN latest_trip_per_vehicle ltpv ON av.vehicle_no = ltpv.vehicle_no
            ORDER BY av.vehicle_no;
        """

        df = pd.read_sql_query(base_query, connection)

        # Convert loading_date to datetime
        if 'loading_date' in df.columns:
            df['loading_date'] = pd.to_datetime(df['loading_date'], errors='coerce')

        # Now get daily GPS KM data for all vehicles
        daily_km_query = """
            SELECT
                UPPER(CASE
                    WHEN vehicle_no LIKE '% %' THEN
                        SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                    ELSE vehicle_no
                END) as vehicle_no,
                DATE(date_time) as km_date,
                COALESCE(MAX(odometer) - MIN(odometer), 0) as daily_km
            FROM fvts_vehicles
            WHERE vehicle_no IS NOT NULL
                AND odometer IS NOT NULL
                AND date_time >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY
                UPPER(CASE
                    WHEN vehicle_no LIKE '% %' THEN
                        SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                    ELSE vehicle_no
                END),
                DATE(date_time)
            ORDER BY vehicle_no, km_date;
        """

        daily_km_df = pd.read_sql_query(daily_km_query, connection)
        daily_km_df['km_date'] = pd.to_datetime(daily_km_df['km_date'])

        # Create a dictionary for quick lookup: {vehicle_no: {date: km}}
        km_lookup = {}
        for _, row in daily_km_df.iterrows():
            vno = row['vehicle_no']
            if vno not in km_lookup:
                km_lookup[vno] = {}
            km_lookup[vno][row['km_date'].date()] = row['daily_km']

        # Add columns for Loading, +1, +2, +3, +4, +5 days
        for day_offset in range(6):
            col_name = 'Loading' if day_offset == 0 else f'+{day_offset}'
            df[col_name] = df.apply(
                lambda row: get_km_for_day(row, km_lookup, day_offset),
                axis=1
            )

        return df

    except Exception as e:
        st.error(f"Error loading trip km by days: {str(e)}")
        return pd.DataFrame()
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

def get_km_for_day(row, km_lookup, day_offset):
    """Get GPS KM for a specific day offset from loading date"""
    if pd.isna(row.get('loading_date')):
        return 0

    vehicle_no = row['vehicle_no']
    loading_date = row['loading_date']

    target_date = (loading_date + timedelta(days=day_offset)).date()

    if vehicle_no in km_lookup and target_date in km_lookup[vehicle_no]:
        return round(km_lookup[vehicle_no][target_date], 2)
    return 0

def show_trip_km_by_days():
    """Display Trip Km By Days table"""
    st.subheader("📅 Trip Km By Days")

    load_df = load_trip_km_by_days_data()

    if len(load_df) == 0:
        st.warning("No data available")
        return

    # Create combined driver name with code column
    if 'driver_name' in load_df.columns and 'driver_code' in load_df.columns:
        load_df['driver_name_code'] = load_df.apply(
            lambda row: f"{row['driver_name']} ({row['driver_code']})" if pd.notna(row['driver_name']) and pd.notna(row['driver_code']) and str(row['driver_name']) != '-' and str(row['driver_code']) != '-' and str(row['driver_name']) != 'nan' and str(row['driver_code']) != 'nan'
            else (str(row['driver_name']) if pd.notna(row['driver_name']) and str(row['driver_name']) != '-' and str(row['driver_name']) != 'nan' else '-'),
            axis=1
        )

    # Prepare display dataframe
    display_cols = ['vehicle_no', 'trip_status', 'route', 'driver_name_code', 'driver_phone_no', 'loading_date', 'Loading', '+1', '+2', '+3', '+4', '+5']
    display_cols = [col for col in display_cols if col in load_df.columns]
    display_df = load_df[display_cols].copy()

    # Format columns
    column_mapping = {
        'vehicle_no': 'Vehicle No',
        'trip_status': 'Trip Status',
        'route': 'Route',
        'driver_name_code': 'Driver Name',
        'driver_phone_no': 'Driver Phone',
        'loading_date': 'Loading Date'
    }

    display_df = display_df.rename(columns=column_mapping)

    # Format loading date
    if 'Loading Date' in display_df.columns:
        display_df['Loading Date'] = load_df['loading_date'].apply(
            lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) else '-'
        )

    # Replace NaN with '-' for text columns only
    for col in ['Vehicle No', 'Trip Status', 'Route', 'Driver Name', 'Driver Phone', 'Loading Date']:
        if col in display_df.columns:
            display_df[col] = display_df[col].fillna('-').replace('None', '-').replace('nan', '-')

    # Format KM columns
    km_cols = ['Loading', '+1', '+2', '+3', '+4', '+5']
    for col in km_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) and x > 0 else '-')

    # Apply styling with dark mode support
    def style_row(row):
        base_style = 'font-size: 13px; border: 1px solid rgba(100, 100, 100, 0.3); padding: 8px; color: #FFFFFF;'

        # Color code by trip status
        trip_status = row.get('Trip Status', '')
        if trip_status == 'Completed':
            bg_color = 'background-color: rgba(34, 139, 34, 0.25);'
        elif trip_status == 'In Transit':
            bg_color = 'background-color: rgba(30, 144, 255, 0.25);'
        elif trip_status == 'Loaded':
            bg_color = 'background-color: rgba(255, 140, 0, 0.25);'
        else:
            bg_color = 'background-color: rgba(40, 40, 40, 0.3);'

        styles = []
        for col in display_df.columns:
            if col == 'Vehicle No':
                styles.append(base_style + bg_color + 'text-align: center; font-weight: 700; color: #4FC3F7;')
            elif col in ['Trip Status', 'Driver Phone', 'Loading Date']:
                styles.append(base_style + bg_color + 'text-align: center;')
            elif col in ['Route', 'Driver Name']:
                styles.append(base_style + bg_color + 'text-align: left;')
            elif col in km_cols:
                styles.append(base_style + bg_color + 'text-align: center; font-weight: 600;')
            else:
                styles.append(base_style + bg_color + 'text-align: left;')

        return styles

    styled_df = display_df.style.apply(style_row, axis=1)

    # Dark mode table styling
    styled_df = styled_df.set_table_styles([
        {'selector': 'thead th',
         'props': [('background-color', 'rgba(30, 30, 30, 0.95)'),
                   ('color', '#FFFFFF'),
                   ('font-weight', 'bold'),
                   ('text-align', 'center'),
                   ('border', '1px solid rgba(100, 100, 100, 0.5)'),
                   ('padding', '12px'),
                   ('font-size', '14px')]},
        {'selector': 'tbody tr:hover',
         'props': [('background-color', 'rgba(66, 165, 245, 0.2)')]},
        {'selector': 'tbody td',
         'props': [('color', '#FFFFFF')]},
    ])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=500
    )

    # Download option
    download_df = load_df[['vehicle_no', 'trip_status', 'route', 'driver_name', 'driver_code', 'driver_phone_no', 'loading_date', 'Loading', '+1', '+2', '+3', '+4', '+5']].copy()
    csv = download_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Trip Km By Days (CSV)",
        data=csv,
        file_name=f"trip_km_by_days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=False
    )

def show_status_summary(df):
    """Show night driving analysis - vehicles driven between 11 PM yesterday and 6 AM today"""
    import streamlit.components.v1 as components
    from datetime import datetime
    import pytz

    st.subheader("🌙 Night Driving Alerts")

    # Check current time for live alert (using Indian Standard Time)
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist)
    current_hour = current_time_ist.hour
    is_night_time = current_hour >= 23 or current_hour < 6

    # Get night driving data from database
    connection = None
    try:
        connection = get_database_connection()

        # First, show LIVE ALERT for vehicles currently running at night
        if is_night_time:
            live_query = """
                SELECT DISTINCT ON (normalized_vehicle_no)
                    vehicle_no,
                    speed,
                    location,
                    date_time,
                    normalized_vehicle_no
                FROM (
                    SELECT
                        f.vehicle_no,
                        f.speed,
                        f.location,
                        f.date_time,
                        UPPER(CASE
                            WHEN f.vehicle_no LIKE '% %' THEN
                                SPLIT_PART(f.vehicle_no, ' ', 2) || SPLIT_PART(f.vehicle_no, ' ', 1)
                            ELSE f.vehicle_no
                        END) as normalized_vehicle_no
                    FROM fvts_vehicles f
                    WHERE f.speed > 0
                        AND f.ignition = 1
                        AND f.date_time >= NOW() - INTERVAL '10 minutes'
                ) sub
                ORDER BY normalized_vehicle_no, date_time DESC
            """
            live_df = pd.read_sql_query(live_query, connection)

            if len(live_df) > 0:
                st.error(f"🚨 **LIVE ALERT: {len(live_df)} vehicles currently running at night!**")

                # Get driver info for live vehicles
                driver_query = """
                    SELECT DISTINCT ON (normalized_vehicle_no)
                        UPPER(CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            WHEN LENGTH(vehicle_no) >= 9
                                AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                                AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                                CASE
                                    WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                        SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                    ELSE
                                        SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                END
                            ELSE vehicle_no
                        END) as normalized_vehicle_no,
                        driver_name,
                        driver_code,
                        driver_phone_no
                    FROM swift_trip_log
                    WHERE vehicle_no IS NOT NULL
                        AND driver_name IS NOT NULL
                        AND driver_name != ''
                    ORDER BY
                        UPPER(CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            WHEN LENGTH(vehicle_no) >= 9
                                AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                                AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                                CASE
                                    WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                        SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                    ELSE
                                        SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                END
                            ELSE vehicle_no
                        END),
                        loading_date DESC NULLS LAST
                """
                driver_df = pd.read_sql_query(driver_query, connection)

                # Get monthly night driving stats
                monthly_night_query = """
                    SELECT
                        UPPER(CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            ELSE vehicle_no
                        END) as normalized_vehicle_no,
                        COUNT(DISTINCT DATE(date_time)) as month_night_days
                    FROM fvts_vehicles
                    WHERE speed > 0
                        AND ignition = 1
                        AND date_time >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Kolkata')
                        AND (EXTRACT(HOUR FROM date_time) >= 23 OR EXTRACT(HOUR FROM date_time) < 6)
                    GROUP BY UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
                """
                monthly_night_df = pd.read_sql_query(monthly_night_query, connection)

                # Merge driver info
                live_df = live_df.merge(driver_df, on='normalized_vehicle_no', how='left')
                live_df['driver_name'] = live_df['driver_name'].fillna('-')
                live_df['driver_code'] = live_df['driver_code'].fillna('-')
                live_df['driver_phone_no'] = live_df['driver_phone_no'].fillna('-')

                # Merge monthly night stats
                live_df = live_df.merge(monthly_night_df, on='normalized_vehicle_no', how='left')
                live_df['month_night_days'] = live_df['month_night_days'].fillna(0).astype(int)

                # Merge owner info
                owner_df = load_owner_mapping()
                live_df = live_df.merge(owner_df, on='normalized_vehicle_no', how='left')
                # Apply custom owner mappings
                live_df['owner_name'] = live_df.apply(
                    lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no'], row['owner_name']),
                    axis=1
                )
                live_df['owner_name'] = live_df['owner_name'].fillna('-')

                # Display live alert table
                alert_html = '''
                <style>
                    .alert-table { width: 100%; border-collapse: collapse; font-size: 13px; animation: blink 1s infinite; }
                    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
                    .alert-table th { background-color: #d32f2f; color: white; padding: 10px; text-align: center; }
                    .alert-table td { background-color: #ffcdd2; color: #000; padding: 8px; border: 1px solid #ef9a9a; }
                    .alert-table td.center { text-align: center; }
                    .alert-table td.month-stats { text-align: center; font-weight: bold; color: #6a1b9a; }
                </style>
                <table class="alert-table">
                <tr><th>Vehicle No</th><th>Driver (Code)</th><th>Phone</th><th>Speed</th><th>Location</th><th>Month Night Days</th><th>Owner Name</th></tr>
                '''
                for _, row in live_df.iterrows():
                    driver = f"{row['driver_name']} ({row['driver_code']})" if row['driver_name'] != '-' else '-'
                    alert_html += f'''<tr>
                        <td class="center"><b>{row['vehicle_no']}</b></td>
                        <td>{driver}</td>
                        <td class="center">{row['driver_phone_no']}</td>
                        <td class="center"><b>{row['speed']}</b> km/h</td>
                        <td>{row['location'] if row['location'] else '-'}</td>
                        <td class="month-stats">{row['month_night_days']}</td>
                        <td>{row['owner_name']}</td>
                    </tr>'''
                alert_html += '</table>'

                components.html(alert_html, height=min(500, 60 + len(live_df) * 40), scrolling=True)
                st.markdown("---")
            else:
                st.info("✅ No vehicles currently running at night.")
                st.markdown("---")
        else:
            st.info(f"ℹ️ Current time (IST): {current_time_ist.strftime('%H:%M')} - Night driving alerts active between 11 PM and 6 AM")
            st.markdown("---")

        st.subheader("📋 Last Night's Driving Summary (11 PM - 6 AM)")

        # Query to find vehicles that moved between 11 PM yesterday and 6 AM today
        # Join with swift_trip_log to get driver details
        query = """
            WITH night_data AS (
                SELECT
                    f.vehicle_no,
                    CASE
                        WHEN f.vehicle_no LIKE '% %' THEN
                            SPLIT_PART(f.vehicle_no, ' ', 2) || SPLIT_PART(f.vehicle_no, ' ', 1)
                        ELSE f.vehicle_no
                    END as normalized_vehicle_no,
                    f.date_time,
                    f.speed,
                    f.odometer,
                    f.location
                FROM fvts_vehicles f
                WHERE f.speed > 0
                    AND f.ignition = 1
                    AND (
                        (f.date_time >= (CURRENT_DATE - INTERVAL '1 day') + INTERVAL '23 hours'
                         AND f.date_time < CURRENT_DATE)
                        OR
                        (f.date_time >= CURRENT_DATE
                         AND f.date_time < CURRENT_DATE + INTERVAL '6 hours')
                    )
            ),
            night_driving AS (
                SELECT
                    vehicle_no,
                    normalized_vehicle_no,
                    MIN(date_time) as first_seen,
                    MAX(date_time) as last_seen,
                    MAX(speed) as max_speed,
                    EXTRACT(EPOCH FROM (MAX(date_time) - MIN(date_time)))/3600 as duration_hours,
                    COALESCE(MAX(odometer) - MIN(odometer), 0) as night_km,
                    COUNT(*) as data_points
                FROM night_data
                GROUP BY vehicle_no, normalized_vehicle_no
            ),
            start_locations AS (
                SELECT DISTINCT ON (vehicle_no)
                    vehicle_no,
                    location as start_location
                FROM night_data
                ORDER BY vehicle_no, date_time ASC
            ),
            end_locations AS (
                SELECT DISTINCT ON (vehicle_no)
                    vehicle_no,
                    location as end_location
                FROM night_data
                ORDER BY vehicle_no, date_time DESC
            ),
            driver_info AS (
                SELECT DISTINCT ON (normalized_vehicle_no)
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    driver_name,
                    driver_code,
                    driver_phone_no,
                    route
                FROM swift_trip_log
                WHERE vehicle_no IS NOT NULL
                    AND driver_name IS NOT NULL
                    AND driver_name != ''
                ORDER BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END),
                    loading_date DESC NULLS LAST
            ),
            monthly_night_driving AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    COUNT(DISTINCT DATE(date_time)) as month_night_days
                FROM fvts_vehicles
                WHERE speed > 0
                    AND ignition = 1
                    AND date_time >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Kolkata')
                    AND (EXTRACT(HOUR FROM date_time) >= 23 OR EXTRACT(HOUR FROM date_time) < 6)
                GROUP BY UPPER(CASE
                    WHEN vehicle_no LIKE '% %' THEN
                        SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                    ELSE vehicle_no
                END)
            )
            SELECT
                nd.vehicle_no,
                COALESCE(di.driver_name, '-') as driver_name,
                COALESCE(di.driver_code, '-') as driver_code,
                COALESCE(di.driver_phone_no, '-') as driver_phone,
                COALESCE(di.route, '-') as route,
                nd.first_seen,
                nd.last_seen,
                ROUND(nd.duration_hours::numeric, 2) as duration_hours,
                ROUND(nd.night_km::numeric, 1) as night_km,
                ROUND(nd.max_speed::numeric, 0) as max_speed,
                nd.data_points,
                COALESCE(sl.start_location, '-') as start_location,
                COALESCE(el.end_location, '-') as end_location,
                COALESCE(mnd.month_night_days, 0) as month_night_days
            FROM night_driving nd
            LEFT JOIN driver_info di ON nd.normalized_vehicle_no = di.normalized_vehicle_no
            LEFT JOIN start_locations sl ON nd.vehicle_no = sl.vehicle_no
            LEFT JOIN end_locations el ON nd.vehicle_no = el.vehicle_no
            LEFT JOIN monthly_night_driving mnd ON nd.normalized_vehicle_no = mnd.normalized_vehicle_no
            WHERE nd.duration_hours >= 0.0333
            ORDER BY nd.night_km DESC, nd.duration_hours DESC;
        """

        night_df = pd.read_sql_query(query, connection)

        if len(night_df) == 0:
            st.info("No vehicles were driven between 11 PM yesterday and 6 AM today.")
            return

        # Merge owner info
        owner_df = load_owner_mapping()
        night_df['normalized_vehicle_no'] = night_df['vehicle_no'].apply(
            lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
        )
        night_df = night_df.merge(owner_df, on='normalized_vehicle_no', how='left')
        # Apply custom owner mappings
        night_df['owner_name'] = night_df.apply(
            lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no'], row['owner_name']),
            axis=1
        )
        night_df['owner_name'] = night_df['owner_name'].fillna('-')

        # Display count
        st.success(f"🚛 **{len(night_df)} vehicles** were driven during night hours (11 PM - 6 AM)")

        # Create display dataframe
        display_df = night_df.copy()
        display_df['Driver'] = display_df.apply(
            lambda row: f"{row['driver_name']} ({row['driver_code']})"
            if row['driver_name'] != '-' and row['driver_code'] != '-'
            else row['driver_name'],
            axis=1
        )
        display_df['First Seen'] = pd.to_datetime(display_df['first_seen']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['Last Seen'] = pd.to_datetime(display_df['last_seen']).dt.strftime('%Y-%m-%d %H:%M')

        # Format duration as HH:MM
        def format_duration(hours):
            if pd.isna(hours) or hours == 0:
                return '-'
            h = int(hours)
            m = int((hours - h) * 60)
            return f"{h}h {m}m"

        display_df['Duration'] = display_df['duration_hours'].apply(format_duration)
        display_df['Night KM'] = display_df['night_km'].apply(lambda x: f"{x:.1f}" if pd.notna(x) and x > 0 else '0.0')

        # Select columns for display
        display_df = display_df[['vehicle_no', 'Driver', 'driver_phone', 'route', 'start_location', 'end_location', 'First Seen', 'Last Seen', 'Duration', 'Night KM', 'max_speed', 'month_night_days', 'owner_name']]
        display_df.columns = ['Vehicle No', 'Driver (Code)', 'Driver Phone', 'Route', 'Start Location', 'End Location', 'First Seen', 'Last Seen', 'Duration', 'Night KM', 'Max Speed', 'Month Night Days', 'Owner Name']

        # Build HTML table
        num_rows = len(display_df)
        table_height = min(500, 60 + num_rows * 40)

        html_table = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .night-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .night-table th {{
                background-color: #1a1a2e;
                color: #FFFFFF;
                font-weight: bold;
                text-align: center;
                border: 1px solid #444;
                padding: 10px;
                font-size: 13px;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .night-table td {{
                border: 1px solid #333;
                padding: 8px;
                color: #FFFFFF;
                background-color: #16213e;
            }}
            .night-table tr:hover td {{
                background-color: rgba(66, 165, 245, 0.2);
            }}
            .night-table .vehicle-no {{
                text-align: center;
                font-weight: 700;
                color: #4FC3F7;
            }}
            .night-table .center {{
                text-align: center;
            }}
            .night-table .left {{
                text-align: left;
            }}
            .table-container {{
                max-height: {table_height - 20}px;
                overflow-y: auto;
                overflow-x: auto;
            }}
        </style>
        </head>
        <body>
        <div class="table-container">
        <table class="night-table">
        <thead><tr>
        '''

        # Add headers
        for col in display_df.columns:
            html_table += f'<th>{col}</th>'
        html_table += '</tr></thead><tbody>'

        # Add rows
        for idx, row in display_df.iterrows():
            html_table += '<tr>'
            for col in display_df.columns:
                val = row[col]
                if col == 'Vehicle No':
                    html_table += f'<td class="vehicle-no">{val}</td>'
                elif col in ['Max Speed', 'First Seen', 'Last Seen', 'Duration', 'Night KM']:
                    html_table += f'<td class="center">{val}</td>'
                elif col == 'Month Night Days':
                    html_table += f'<td class="center" style="font-weight: bold; color: #ab47bc;">{int(val)}</td>'
                elif col in ['Start Location', 'End Location']:
                    html_table += f'<td class="left" style="max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{val}">{val}</td>'
                else:
                    html_table += f'<td class="left">{val}</td>'
            html_table += '</tr>'

        html_table += '</tbody></table></div></body></html>'

        components.html(html_table, height=table_height, scrolling=True)

        # Download option
        csv = night_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Night Driving Report (CSV)",
            data=csv,
            file_name=f"night_driving_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=False
        )

    except Exception as e:
        st.error(f"Error loading night driving data: {str(e)}")
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

@st.cache_data(ttl=120, show_spinner=False)  # Cache for 2 minutes
def get_overspeed_data():
    """Cached function to get overspeed data from database"""
    connection = None
    try:
        connection = get_database_connection()

        # Query for 24h overspeed summary
        query = """
            WITH overspeed_data AS (
                SELECT
                    f.vehicle_no,
                    UPPER(CASE
                        WHEN f.vehicle_no LIKE '% %' THEN
                            SPLIT_PART(f.vehicle_no, ' ', 2) || SPLIT_PART(f.vehicle_no, ' ', 1)
                        ELSE f.vehicle_no
                    END) as normalized_vehicle_no,
                    f.speed,
                    f.location,
                    f.date_time
                FROM fvts_vehicles f
                WHERE f.speed > 60
                    AND f.ignition = 1
                    AND f.date_time >= NOW() - INTERVAL '24 hours'
            ),
            max_speed_locations AS (
                SELECT DISTINCT ON (vehicle_no)
                    vehicle_no,
                    location as max_speed_location
                FROM overspeed_data
                ORDER BY vehicle_no, speed DESC
            ),
            overspeed_summary AS (
                SELECT
                    od.vehicle_no,
                    od.normalized_vehicle_no,
                    MAX(od.speed) as max_speed,
                    ROUND(AVG(od.speed)::numeric, 0) as avg_speed,
                    MIN(od.date_time) as first_overspeed,
                    MAX(od.date_time) as last_overspeed,
                    COUNT(*) as overspeed_times,
                    msl.max_speed_location
                FROM overspeed_data od
                LEFT JOIN max_speed_locations msl ON od.vehicle_no = msl.vehicle_no
                GROUP BY od.vehicle_no, od.normalized_vehicle_no, msl.max_speed_location
            ),
            driver_info AS (
                SELECT DISTINCT ON (normalized_vehicle_no)
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    driver_name,
                    driver_code,
                    driver_phone_no
                FROM swift_trip_log
                WHERE vehicle_no IS NOT NULL
                    AND driver_name IS NOT NULL
                    AND driver_name != ''
                ORDER BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END),
                    loading_date DESC NULLS LAST
            ),
            monthly_overspeed_stats AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    COUNT(*) as month_overspeed_count,
                    COUNT(DISTINCT DATE(date_time)) as month_overspeed_days
                FROM fvts_vehicles
                WHERE speed > 60
                    AND ignition = 1
                    AND date_time >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Kolkata')
                GROUP BY UPPER(CASE
                    WHEN vehicle_no LIKE '% %' THEN
                        SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                    ELSE vehicle_no
                END)
            )
            SELECT
                os.vehicle_no,
                os.normalized_vehicle_no,
                COALESCE(di.driver_name, '-') as driver_name,
                COALESCE(di.driver_code, '-') as driver_code,
                COALESCE(di.driver_phone_no, '-') as driver_phone,
                os.max_speed,
                ROUND(os.avg_speed::numeric, 0) as avg_speed,
                os.first_overspeed,
                os.last_overspeed,
                os.overspeed_times,
                os.max_speed_location,
                COALESCE(mos.month_overspeed_count, 0) as month_overspeed_count,
                COALESCE(mos.month_overspeed_days, 0) as month_overspeed_days
            FROM overspeed_summary os
            LEFT JOIN driver_info di ON os.normalized_vehicle_no = di.normalized_vehicle_no
            LEFT JOIN monthly_overspeed_stats mos ON os.normalized_vehicle_no = mos.normalized_vehicle_no
            ORDER BY os.overspeed_times DESC, os.max_speed DESC;
        """
        overspeed_df = pd.read_sql_query(query, connection)

        # Live overspeed query with driver info from swift_trip_log
        live_query = """
            WITH latest_records AS (
                SELECT DISTINCT ON (vehicle_no)
                    vehicle_no,
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    speed,
                    location,
                    date_time,
                    ignition
                FROM fvts_vehicles
                ORDER BY vehicle_no, date_time DESC
            ),
            overspeed_vehicles AS (
                SELECT
                    vehicle_no,
                    normalized_vehicle_no,
                    speed,
                    location,
                    EXTRACT(EPOCH FROM ((NOW() AT TIME ZONE 'Asia/Kolkata') - date_time))/60 as duration_mins,
                    date_time
                FROM latest_records
                WHERE speed > 60
                    AND ignition = 1
                    AND EXTRACT(EPOCH FROM ((NOW() AT TIME ZONE 'Asia/Kolkata') - date_time))/60 >= 1
            ),
            monthly_overspeed_stats AS (
                SELECT
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    COUNT(*) as month_overspeed_count,
                    COUNT(DISTINCT DATE(date_time)) as month_overspeed_days
                FROM fvts_vehicles
                WHERE speed > 60
                    AND ignition = 1
                    AND date_time >= DATE_TRUNC('month', NOW() AT TIME ZONE 'Asia/Kolkata')
                GROUP BY UPPER(CASE
                    WHEN vehicle_no LIKE '% %' THEN
                        SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                    ELSE vehicle_no
                END)
            ),
            driver_info AS (
                SELECT DISTINCT ON (normalized_vehicle_no)
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END) as normalized_vehicle_no,
                    driver_name,
                    driver_code,
                    driver_phone_no
                FROM swift_trip_log
                WHERE vehicle_no IS NOT NULL
                    AND driver_name IS NOT NULL
                    AND driver_name != ''
                ORDER BY
                    UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        WHEN LENGTH(vehicle_no) >= 9
                            AND SUBSTRING(vehicle_no FROM 1 FOR 4) ~ '^[0-9]+$'
                            AND SUBSTRING(vehicle_no FROM 5 FOR 2) ~ '^[A-Z]+$' THEN
                            CASE
                                WHEN POSITION('-' IN vehicle_no) > 0 THEN
                                    SUBSTRING(vehicle_no FROM 5 FOR POSITION('-' IN vehicle_no) - 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                                ELSE
                                    SUBSTRING(vehicle_no FROM 5) || SUBSTRING(vehicle_no FROM 1 FOR 4)
                            END
                        ELSE vehicle_no
                    END),
                    loading_date DESC NULLS LAST
            )
            SELECT
                ov.vehicle_no,
                ov.normalized_vehicle_no,
                ov.speed,
                ov.location,
                ov.duration_mins,
                ov.date_time,
                COALESCE(di.driver_name, '-') as driver_name,
                COALESCE(di.driver_code, '-') as driver_code,
                COALESCE(di.driver_phone_no, '-') as driver_phone_no,
                COALESCE(mos.month_overspeed_count, 0) as month_overspeed_count,
                COALESCE(mos.month_overspeed_days, 0) as month_overspeed_days
            FROM overspeed_vehicles ov
            LEFT JOIN driver_info di ON ov.normalized_vehicle_no = di.normalized_vehicle_no
            LEFT JOIN monthly_overspeed_stats mos ON ov.normalized_vehicle_no = mos.normalized_vehicle_no
            ORDER BY ov.duration_mins DESC, ov.speed DESC
        """
        live_overspeed_df = pd.read_sql_query(live_query, connection)

        return overspeed_df, live_overspeed_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame()
    finally:
        if connection:
            try:
                connection.close()
            except:
                pass

def show_overspeed_alerts(df):
    """Show overspeed alerts - vehicles with speed > 60 km/h for duration >= 1 minute"""
    import streamlit.components.v1 as components
    from datetime import datetime

    st.subheader("🚨 Overspeed Alerts (Speed > 60 km/h)")

    # Get overspeed data from CACHED function (fast!)
    overspeed_df, live_overspeed_df = get_overspeed_data()

    try:

        # Show live alert if there are currently overspeeding vehicles
        if len(live_overspeed_df) > 0:
            st.error(f"🚨 **LIVE ALERT: {len(live_overspeed_df)} vehicles CURRENTLY overspeeding (>60 km/h)!**")

            # Merge owner info (from cached Excel - fast)
            owner_df = load_owner_mapping()
            live_overspeed_df = live_overspeed_df.merge(owner_df, on='normalized_vehicle_no', how='left')
            # Apply custom owner mappings
            live_overspeed_df['owner_name'] = live_overspeed_df.apply(
                lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no'], row['owner_name']),
                axis=1
            )
            live_overspeed_df['owner_name'] = live_overspeed_df['owner_name'].fillna('-')

            # Display live alert table with driver info
            alert_html = '''
            <style>
                .overspeed-alert-table { width: 100%; border-collapse: collapse; font-size: 13px; animation: blink 1s infinite; }
                @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
                .overspeed-alert-table th { background-color: #ff5722; color: white; padding: 10px; text-align: center; }
                .overspeed-alert-table td { background-color: #ffccbc; color: #000; padding: 8px; border: 1px solid #ffab91; }
                .overspeed-alert-table td.center { text-align: center; }
                .overspeed-alert-table td.speed { text-align: center; font-weight: bold; color: #d32f2f; font-size: 15px; }
                .overspeed-alert-table td.duration { text-align: center; font-weight: bold; color: #e65100; }
                .overspeed-alert-table td.month-stats { text-align: center; font-weight: bold; color: #6a1b9a; }
            </style>
            <table class="overspeed-alert-table">
            <tr><th>Vehicle No</th><th>Driver (Code)</th><th>Phone</th><th>Speed</th><th>Duration</th><th>Location</th><th>Month Overspeed Count</th><th>Month Overspeed Days</th><th>Owner Name</th></tr>
            '''
            for _, row in live_overspeed_df.iterrows():
                # Format driver info
                driver_name = row.get('driver_name', '-') or '-'
                driver_code = row.get('driver_code', '-') or '-'
                driver_phone = row.get('driver_phone_no', '-') or '-'
                driver = f"{driver_name} ({driver_code})" if driver_name != '-' else '-'
                # Format duration
                dur_mins = row.get('duration_mins', 0) or 0
                if dur_mins < 60:
                    duration_str = f"{int(dur_mins)}m"
                else:
                    h = int(dur_mins // 60)
                    m = int(dur_mins % 60)
                    duration_str = f"{h}h {m}m"
                month_count = int(row.get('month_overspeed_count', 0) or 0)
                month_days = int(row.get('month_overspeed_days', 0) or 0)
                alert_html += f'''<tr>
                    <td class="center"><b>{row['vehicle_no']}</b></td>
                    <td>{driver}</td>
                    <td class="center">{driver_phone}</td>
                    <td class="speed">{row['speed']} km/h</td>
                    <td class="duration">{duration_str}</td>
                    <td>{row['location'] if row['location'] else '-'}</td>
                    <td class="month-stats">{month_count}</td>
                    <td class="month-stats">{month_days}</td>
                    <td>{row['owner_name']}</td>
                </tr>'''
            alert_html += '</table>'

            components.html(alert_html, height=min(250, 60 + len(live_overspeed_df) * 40), scrolling=True)
            st.markdown("---")
        else:
            st.success("✅ No vehicles currently overspeeding.")
            st.markdown("---")

        # Show 24-hour overspeed summary
        st.subheader("📋 Overspeed Summary (Last 24 Hours)")

        if len(overspeed_df) == 0:
            st.info("No overspeed incidents (>60 km/h for >=1 min) in the last 24 hours.")
            return

        # Merge owner info for overspeed summary
        owner_df = load_owner_mapping()
        overspeed_df['normalized_vehicle_no'] = overspeed_df['vehicle_no'].apply(
            lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
        )
        overspeed_df = overspeed_df.merge(owner_df, on='normalized_vehicle_no', how='left')
        # Apply custom owner mappings
        overspeed_df['owner_name'] = overspeed_df.apply(
            lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no'], row['owner_name']),
            axis=1
        )
        overspeed_df['owner_name'] = overspeed_df['owner_name'].fillna('-')

        st.warning(f"⚠️ **{len(overspeed_df)} vehicles** had overspeed incidents in the last 24 hours")

        # Create display dataframe
        display_df = overspeed_df.copy()
        display_df['Driver'] = display_df.apply(
            lambda row: f"{row['driver_name']} ({row['driver_code']})"
            if row['driver_name'] != '-' and row['driver_code'] != '-'
            else row['driver_name'],
            axis=1
        )
        display_df['First Overspeed'] = pd.to_datetime(display_df['first_overspeed']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['Last Overspeed'] = pd.to_datetime(display_df['last_overspeed']).dt.strftime('%Y-%m-%d %H:%M')

        # Select columns for display (removed Duration, added Overspeed Times)
        display_df = display_df[['vehicle_no', 'Driver', 'driver_phone', 'max_speed', 'avg_speed', 'overspeed_times', 'First Overspeed', 'Last Overspeed', 'max_speed_location', 'month_overspeed_count', 'month_overspeed_days', 'owner_name']]
        display_df.columns = ['Vehicle No', 'Driver (Code)', 'Phone', 'Max Speed', 'Avg Speed', 'Overspeed Count', 'First Overspeed', 'Last Overspeed', 'Location', 'Month Overspeed Count', 'Month Overspeed Days', 'Owner Name']

        # Build HTML table
        num_rows = len(display_df)
        table_height = min(500, 60 + num_rows * 40)

        html_table = f'''
        <!DOCTYPE html>
        <html>
        <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }}
            .overspeed-table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 13px;
            }}
            .overspeed-table th {{
                background-color: #ff5722;
                color: #FFFFFF;
                font-weight: bold;
                text-align: center;
                border: 1px solid #e64a19;
                padding: 10px;
                font-size: 13px;
                position: sticky;
                top: 0;
                z-index: 10;
            }}
            .overspeed-table td {{
                border: 1px solid #444;
                padding: 8px;
                color: #FFFFFF;
                background-color: #37474f;
            }}
            .overspeed-table tr:hover td {{
                background-color: rgba(255, 87, 34, 0.2);
            }}
            .overspeed-table .vehicle-no {{
                text-align: center;
                font-weight: 700;
                color: #ff9800;
            }}
            .overspeed-table .center {{
                text-align: center;
            }}
            .overspeed-table .speed {{
                text-align: center;
                font-weight: bold;
                color: #ff5722;
            }}
            .overspeed-table .left {{
                text-align: left;
            }}
            .table-container {{
                max-height: {table_height - 20}px;
                overflow-y: auto;
                overflow-x: auto;
            }}
        </style>
        </head>
        <body>
        <div class="table-container">
        <table class="overspeed-table">
        <thead><tr>
        '''

        # Add headers
        for col in display_df.columns:
            html_table += f'<th>{col}</th>'
        html_table += '</tr></thead><tbody>'

        # Add rows
        for idx, row in display_df.iterrows():
            html_table += '<tr>'
            for col in display_df.columns:
                val = row[col]
                if col == 'Vehicle No':
                    html_table += f'<td class="vehicle-no">{val}</td>'
                elif col in ['Max Speed', 'Avg Speed']:
                    html_table += f'<td class="speed">{val} km/h</td>'
                elif col == 'Overspeed Count':
                    html_table += f'<td class="speed">{int(val)} times</td>'
                elif col in ['First Overspeed', 'Last Overspeed']:
                    html_table += f'<td class="center">{val}</td>'
                elif col in ['Month Overspeed Count', 'Month Overspeed Days']:
                    html_table += f'<td class="center" style="font-weight: bold; color: #ab47bc;">{int(val)}</td>'
                else:
                    html_table += f'<td class="left">{val}</td>'
            html_table += '</tr>'

        html_table += '</tbody></table></div></body></html>'

        components.html(html_table, height=table_height, scrolling=True)

        # Download option
        csv = overspeed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Overspeed Report (CSV)",
            data=csv,
            file_name=f"overspeed_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=False
        )

        # Legends at bottom
        st.markdown("---")
        st.markdown("### Legend")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style="background-color: #1a1a2e; padding: 12px 15px; border-radius: 5px; border-left: 4px solid #ff5722;">
            <b style="color: #ff5722;">Live Alert Logic:</b><br>
            <span style="color: #ccc; font-size: 13px;">
            • Shows vehicles where <b>latest GPS entry</b> has speed > 60 km/h<br>
            • <b>Duration</b> = time since that entry was recorded<br>
            • Alert triggers only if duration >= 1 minute
            </span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style="background-color: #37474f; padding: 12px 15px; border-radius: 5px; border-left: 4px solid #ff9800;">
            <b style="color: #ff9800;">24-Hour Summary Logic:</b><br>
            <span style="color: #ccc; font-size: 13px;">
            • Shows all vehicles that exceeded 60 km/h in last 24 hours<br>
            • <b>Overspeed Count</b> = total GPS records with speed > 60 km/h<br>
            • <b>Location</b> = where max speed was recorded
            </span>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading overspeed data: {str(e)}")

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_driver_home_data():
    """Get driver home addresses from Swift Driver Address.xls file"""
    try:
        from pathlib import Path

        # Get the directory where this script is located (works on local and cloud)
        script_dir = Path(__file__).parent.resolve()
        excel_path = script_dir / 'Swift Driver Address.xls'

        if not excel_path.exists():
            return pd.DataFrame()

        driver_df = pd.read_excel(excel_path)

        # Rename columns to match expected format
        driver_df = driver_df.rename(columns={
            'Code': 'driver_code',
            'Name': 'driver_name',
            'PrintingAddress': 'present_address',
            'State': 'state',
            'City': 'city'
        })

        # Filter out rows without address
        driver_df = driver_df[
            (driver_df['driver_code'].notna()) &
            (driver_df['present_address'].notna()) &
            (driver_df['present_address'] != '')
        ]

        return driver_df[['driver_code', 'driver_name', 'present_address', 'state', 'city']]
    except Exception as e:
        return pd.DataFrame()

def check_driver_at_home(vehicle_location, driver_address, state=None, city=None):
    """Check if vehicle location matches driver's home address or city"""
    if not vehicle_location:
        return False

    # Normalize strings for comparison
    vehicle_loc = str(vehicle_location).lower().strip()
    driver_addr = str(driver_address).lower().strip() if pd.notna(driver_address) else ''
    city_str = str(city).lower().strip() if pd.notna(city) else ''

    # Exclude state names and common generic location words
    exclude_words = {
        # State names
        'uttar', 'pradesh', 'madhya', 'bihar', 'rajasthan', 'haryana', 'punjab', 'gujarat',
        'maharashtra', 'karnataka', 'tamil', 'nadu', 'kerala', 'west', 'bengal', 'odisha',
        'andhra', 'telangana', 'assam', 'chhattisgarh', 'jharkhand', 'uttarakhand',
        'himachal', 'jammu', 'kashmir', 'goa', 'sikkim', 'tripura', 'meghalaya', 'manipur',
        'mizoram', 'nagaland', 'arunachal', 'india', 'state',
        # Common generic words
        'road', 'highway', 'national', 'unnamed', 'urban', 'rural', 'near', 'opposite',
        'vill', 'village', 'post', 'dist', 'distt', 'district', 'tehsil', 'block',
        'ward', 'code', 'pincode', 'town', 'township', 'sector', 'phase', 'plot',
        # Short words that cause false matches (e.g., "tara" matches "uttarakhand")
        'tara', 'nagar', 'pur', 'pur', 'ganj', 'pura', 'abad', 'garh', 'wala', 'wali',
        'khan', 'singh', 'kumar', 'devi', 'lal', 'ram', 'shah', 'ali', 'mohd', 'mohammad',
        'house', 'home', 'flat', 'floor', 'building', 'colony', 'society', 'apartment',
        'lane', 'gali', 'mohalla', 'para', 'tola', 'basti', 'chowk', 'main', 'market',
        'muslim', 'hindu', 'sikh', 'christian', 'temple', 'masjid', 'church', 'gurudwara'
    }

    # First: Check if any word from driver address matches current location
    # Require minimum 5 characters to avoid false matches
    if driver_addr:
        addr_words = [w.strip(',-./\n').lower() for w in driver_addr.split() if len(w.strip(',-./\n')) >= 5]
        # Filter out state names and common generic words
        addr_places = [w for w in addr_words if w not in exclude_words]

        for place in addr_places:
            # Use word boundary matching to avoid partial matches
            # Check if place appears as a complete word in location
            import re
            if re.search(r'\b' + re.escape(place) + r'\b', vehicle_loc):
                return True

    # Second: Check if city name matches (must be at least 4 chars and exact word match)
    if city_str and len(city_str) >= 4 and city_str not in exclude_words:
        import re
        if re.search(r'\b' + re.escape(city_str) + r'\b', vehicle_loc):
            return True

    return False

def show_driver_at_home(df):
    """Show drivers who are currently at their home location"""
    import streamlit.components.v1 as components

    st.subheader("🏠 Driver at Home")

    try:
        # Get driver home addresses from Excel file
        driver_home_df = get_driver_home_data()

        if len(driver_home_df) == 0:
            st.info("No driver home address data available.")
            return

        # Use the same load details data as Load Details tab (for driver info)
        load_df = load_vehicle_load_details()

        if len(load_df) == 0:
            st.info("No load details data available.")
            return

        # Merge load_df with main df to get speed, location, and date_time
        # Main df has: speed, location, date_time columns
        # load_df has: driver_name, driver_code, driver_phone_no
        merge_cols = ['vehicle_no', 'speed', 'location']
        if 'date_time' in df.columns:
            merge_cols.append('date_time')
        merged_vehicle_df = load_df.merge(
            df[merge_cols],
            on='vehicle_no',
            how='left'
        )

        # Get idle time for each vehicle (time since last movement > 5 km/h)
        try:
            connection = get_database_connection()
            idle_query = """
                WITH last_movement AS (
                    SELECT
                        vehicle_no,
                        UPPER(CASE
                            WHEN vehicle_no LIKE '% %' THEN
                                SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                            ELSE vehicle_no
                        END) as normalized_vehicle_no,
                        MAX(date_time) as last_moving_time
                    FROM fvts_vehicles
                    WHERE speed > 5
                        AND date_time >= NOW() - INTERVAL '7 days'
                    GROUP BY vehicle_no, UPPER(CASE
                        WHEN vehicle_no LIKE '% %' THEN
                            SPLIT_PART(vehicle_no, ' ', 2) || SPLIT_PART(vehicle_no, ' ', 1)
                        ELSE vehicle_no
                    END)
                )
                SELECT
                    normalized_vehicle_no,
                    last_moving_time,
                    EXTRACT(EPOCH FROM (NOW() AT TIME ZONE 'Asia/Kolkata' - last_moving_time))/3600 as idle_hours
                FROM last_movement
            """
            idle_df = pd.read_sql_query(idle_query, connection)
            connection.close()

            # Normalize vehicle_no in merged_vehicle_df for joining
            merged_vehicle_df['normalized_vehicle_no'] = merged_vehicle_df['vehicle_no'].apply(
                lambda x: str(x).upper().replace(' ', '').replace('-', '') if pd.notna(x) else ''
            )
            merged_vehicle_df = merged_vehicle_df.merge(idle_df, on='normalized_vehicle_no', how='left')
        except Exception as e:
            merged_vehicle_df['idle_hours'] = None

        # Filter for stationary/idle vehicles with drivers assigned
        vehicle_driver_df = merged_vehicle_df[
            (merged_vehicle_df['speed'].fillna(0) <= 5) &
            (merged_vehicle_df['driver_code'].notna()) &
            (merged_vehicle_df['driver_code'] != '') &
            (merged_vehicle_df['driver_code'] != '-')
        ].copy()

        if len(vehicle_driver_df) == 0:
            st.info("No stationary vehicles with assigned drivers found.")
            return

        # Merge with driver home addresses from Excel file
        vehicle_driver_df['driver_code_upper'] = vehicle_driver_df['driver_code'].astype(str).str.upper().str.strip()
        driver_home_df['driver_code_upper'] = driver_home_df['driver_code'].astype(str).str.upper().str.strip()

        merged_df = vehicle_driver_df.merge(
            driver_home_df[['driver_code_upper', 'present_address', 'state', 'city']],
            on='driver_code_upper',
            how='left'
        )

        # Check which drivers are at home
        drivers_at_home = []
        for _, row in merged_df.iterrows():
            location = row.get('location', '')
            if pd.notna(row.get('present_address')) and pd.notna(location) and location:
                if check_driver_at_home(location, row['present_address'], row.get('state'), row.get('city')):
                    drivers_at_home.append(row)

        if len(drivers_at_home) == 0:
            st.success("✅ No drivers currently at home. All drivers are on duty!")

            # Show all stationary vehicles with drivers for reference
            st.markdown("---")
            st.subheader("📋 Stationary Vehicles with Drivers")
            st.info(f"Showing {len(merged_df)} stationary vehicles (speed <= 5)")

            display_cols = ['vehicle_no', 'driver_name', 'driver_code', 'driver_phone_no', 'location', 'party']
            available_cols = [c for c in display_cols if c in merged_df.columns]
            display_df = merged_df[available_cols].copy()
            display_df.columns = ['Vehicle No', 'Driver Name', 'Driver Code', 'Phone', 'Current Location', 'Party'][:len(available_cols)]
            st.dataframe(display_df, use_container_width=True, height=400)
            return

        # Create dataframe of drivers at home
        at_home_df = pd.DataFrame(drivers_at_home)

        st.warning(f"⚠️ **{len(at_home_df)} driver(s) detected at home location!**")

        # Display table
        alert_html = '''
        <style>
            .home-alert-table { width: 100%; border-collapse: collapse; font-size: 13px; }
            .home-alert-table th { background-color: #ff9800; color: white; padding: 10px; text-align: center; }
            .home-alert-table td { background-color: #fff3e0; color: #000; padding: 8px; border: 1px solid #ffcc80; }
            .home-alert-table td.center { text-align: center; }
            .home-alert-table td.vehicle { text-align: center; font-weight: bold; color: #e65100; }
            .home-alert-table td.idle { text-align: center; font-weight: bold; color: #d32f2f; }
        </style>
        <table class="home-alert-table">
        <tr><th>Vehicle No</th><th>Driver Name</th><th>Driver Code</th><th>Phone</th><th>Current Location</th><th>Home Address</th><th>Idle Time</th><th>Party</th></tr>
        '''

        for row in drivers_at_home:
            driver_name = row.get('driver_name', '-') or '-'
            driver_code = row.get('driver_code', '-') or '-'
            phone = row.get('driver_phone_no', '-') or '-'
            location = row.get('location', '-') or '-'
            current_loc = str(location)
            # Clean home address - remove phone numbers
            home_addr = str(row.get('present_address', '-')) if row.get('present_address') else '-'
            # Remove patterns like "PH - 1234567890", "Driver Ph - 1234567890", standalone numbers
            import re
            home_addr = re.sub(r'\s*(Driver\s*)?(Ph|PH|Phone)\s*[-:.]?\s*\d{7,12}', '', home_addr)
            home_addr = re.sub(r'\s+\d{10,12}(-\w+)?', '', home_addr)  # Remove standalone phone numbers
            home_addr = re.sub(r'\s*Home\s*$', '', home_addr)  # Remove trailing "Home"
            home_addr = re.sub(r'\s*-BRO\s*$', '', home_addr)  # Remove trailing "-BRO"
            home_addr = re.sub(r'\s{2,}', ' ', home_addr).strip()  # Clean extra spaces
            party = row.get('party', '-') or '-'

            # Format idle time (same logic as Live Vehicle Details)
            idle_hours = row.get('idle_hours', None)
            if idle_hours is not None and not pd.isna(idle_hours):
                if idle_hours >= 24:
                    # Show in days and hours if >= 24 hours
                    days = int(idle_hours // 24)
                    remaining_hours = int(idle_hours % 24)
                    idle_str = f"{days}d {remaining_hours}h"
                else:
                    # Show in hours and minutes if < 24 hours
                    hours = int(idle_hours)
                    minutes = int((idle_hours % 1) * 60)
                    idle_str = f"{hours}h {minutes}m"
            else:
                # No movement data in last 7 days
                idle_str = ">7 days"

            alert_html += f'''<tr>
                <td class="vehicle">{row['vehicle_no']}</td>
                <td>{driver_name}</td>
                <td class="center">{driver_code}</td>
                <td class="center">{phone}</td>
                <td>{current_loc}</td>
                <td>{home_addr}</td>
                <td class="idle">{idle_str}</td>
                <td>{party}</td>
            </tr>'''

        alert_html += '</table>'
        components.html(alert_html, height=min(300, 60 + len(drivers_at_home) * 45), scrolling=True)

        # Legend
        st.markdown("---")
        st.markdown("""
        <div style="background-color: #1a1a2e; padding: 12px 15px; border-radius: 5px; border-left: 4px solid #ff9800;">
        <b style="color: #ff9800;">Detection Logic:</b><br>
        <span style="color: #ccc; font-size: 13px;">
        • Compares vehicle's current GPS location with driver's registered home address<br>
        • Only checks stationary vehicles (speed = 0)<br>
        • Match is based on location keywords (city, district, area names)
        </span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading driver at home data: {str(e)}")

def show_nearby_vehicles(df, search_lat, search_lon, radius):
    """Show vehicles near a specific location"""

    st.subheader(f"🔍 Vehicles within {radius} km")

    # Calculate distance for each vehicle
    df['distance_km'] = df.apply(
        lambda row: calculate_distance(search_lat, search_lon, float(row['latitude']), float(row['longitude'])),
        axis=1
    )

    # Filter vehicles within radius
    nearby_df = df[df['distance_km'] <= radius].copy()
    nearby_df = nearby_df.sort_values('distance_km')

    if len(nearby_df) == 0:
        st.warning(f"No vehicles found within {radius} km of the search location")
        return nearby_df

    # Display count
    st.success(f"Found {len(nearby_df)} vehicles within {radius} km")

    # Get load details data directly from Load Details tab
    try:
        load_df = load_vehicle_load_details()

        if len(load_df) > 0:
            # Create driver display column
            load_df['driver_display'] = load_df.apply(
                lambda row: f"{row['driver_name']} ({row['driver_code']})"
                if pd.notna(row.get('driver_name')) and pd.notna(row.get('driver_code'))
                and str(row['driver_name']) not in ['-', 'None', 'nan', '']
                and str(row['driver_code']) not in ['-', 'None', 'nan', '']
                else (str(row['driver_name']) if pd.notna(row.get('driver_name')) and str(row['driver_name']) not in ['-', 'None', 'nan', ''] else '-'),
                axis=1
            )

            # Format loading date
            load_df['loading_date_fmt'] = pd.to_datetime(load_df['loading_date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')

            # Get owner_name if available
            if 'owner_name' not in load_df.columns:
                load_df['owner_name'] = '-'

            # Rename columns to avoid any conflicts and use prefixed names
            load_df_subset = load_df[['vehicle_no', 'trip_status', 'route', 'onward_route', 'party',
                                       'loading_date_fmt', 'driver_display', 'driver_phone_no', 'owner_name']].copy()
            load_df_subset.columns = ['vehicle_no', 'ld_trip_status', 'ld_route', 'ld_onward_route', 'ld_party',
                                       'ld_loading_date', 'ld_driver', 'ld_driver_phone', 'ld_owner_name']

            # Fill NaN values before merge
            for col in ['ld_trip_status', 'ld_route', 'ld_onward_route', 'ld_party', 'ld_driver_phone', 'ld_driver', 'ld_loading_date', 'ld_owner_name']:
                load_df_subset[col] = load_df_subset[col].fillna('-').astype(str)
                load_df_subset[col] = load_df_subset[col].replace(['None', 'nan', 'NaT', ''], '-')

            nearby_df = nearby_df.merge(load_df_subset, on='vehicle_no', how='left')

            # Copy to standard column names for display
            nearby_df['trip_status'] = nearby_df['ld_trip_status'].fillna('-')
            nearby_df['route'] = nearby_df['ld_route'].fillna('-')
            nearby_df['onward_route'] = nearby_df['ld_onward_route'].fillna('-')
            nearby_df['party'] = nearby_df['ld_party'].fillna('-')
            nearby_df['loading_date_fmt'] = nearby_df['ld_loading_date'].fillna('-')
            nearby_df['driver_display'] = nearby_df['ld_driver'].fillna('-')
            nearby_df['driver_phone_no'] = nearby_df['ld_driver_phone'].fillna('-')
            nearby_df['owner_name'] = nearby_df['ld_owner_name'].fillna('-')
    except Exception as e:
        st.error(f"Error loading trip details: {str(e)}")

    # Prepare display dataframe
    base_cols = ['vehicle_no', 'status', 'speed', 'distance_km', 'location', 'ignition']
    base_names = ['Vehicle No', 'Status', 'Speed (km/h)', 'Distance (km)', 'Location', 'Ignition']

    # Add load detail columns if available
    extra_cols = []
    extra_names = []
    if 'trip_status' in nearby_df.columns:
        extra_cols.append('trip_status')
        extra_names.append('Trip Status')
    if 'route' in nearby_df.columns:
        extra_cols.append('route')
        extra_names.append('Route')
    if 'onward_route' in nearby_df.columns:
        extra_cols.append('onward_route')
        extra_names.append('Onward Route')
    if 'party' in nearby_df.columns:
        extra_cols.append('party')
        extra_names.append('Party')
    if 'loading_date_fmt' in nearby_df.columns:
        extra_cols.append('loading_date_fmt')
        extra_names.append('Loading Date')
    if 'driver_display' in nearby_df.columns:
        extra_cols.append('driver_display')
        extra_names.append('Driver')
    if 'driver_phone_no' in nearby_df.columns:
        extra_cols.append('driver_phone_no')
        extra_names.append('Driver Phone')

    # Fill NaN values for display
    nearby_df['trip_status'] = nearby_df.get('trip_status', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['route'] = nearby_df.get('route', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['onward_route'] = nearby_df.get('onward_route', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['party'] = nearby_df.get('party', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['loading_date_fmt'] = nearby_df.get('loading_date_fmt', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['driver_display'] = nearby_df.get('driver_display', pd.Series(['-']*len(nearby_df))).fillna('-')
    nearby_df['driver_phone_no'] = nearby_df.get('driver_phone_no', pd.Series(['-']*len(nearby_df))).fillna('-')

    # Build HTML table
    import streamlit.components.v1 as components
    num_rows = len(nearby_df)
    table_height = min(500, 60 + num_rows * 40)

    html_table = f'''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ margin: 0; padding: 0; background-color: transparent; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        .nearby-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .nearby-table th {{ background-color: #505050; color: #FFFFFF; font-weight: bold; text-align: center; border: 1px solid #666; padding: 8px; font-size: 12px; position: sticky; top: 0; z-index: 10; }}
        .nearby-table td {{ padding: 6px; border: 1px solid #666; color: #000; }}
        .nearby-table td.center {{ text-align: center; }}
        .nearby-table td.left {{ text-align: left; }}
        .nearby-table td.vehicle {{ text-align: center; font-weight: 600; }}
        .nearby-table tr.moving {{ background-color: rgba(144, 238, 144, 0.5); }}
        .nearby-table tr.idle {{ background-color: rgba(255, 213, 128, 0.6); }}
        .nearby-table tr.stopped {{ background-color: rgba(220, 120, 130, 0.75); }}
        .table-container {{ max-height: {table_height - 20}px; overflow-y: auto; overflow-x: auto; }}
    </style>
    </head>
    <body>
    <div class="table-container">
    <table class="nearby-table">
    <thead><tr>
        <th>Vehicle No</th><th>Status</th><th>Speed</th><th>Distance</th><th>Location</th><th>Ignition</th>
        <th>Trip Status</th><th>Route</th><th>Onward Route</th><th>Party</th><th>Loading Date</th><th>Driver</th><th>Driver Phone</th><th>Owner Name</th>
    </tr></thead><tbody>
    '''

    for _, row in nearby_df.iterrows():
        status = row.get('status', 'Unknown')
        row_class = 'moving' if status == 'Moving' else ('idle' if status == 'Idle' else ('stopped' if status == 'Stopped' else ''))
        ignition = 'ON' if row.get('ignition', 0) == 1 else 'OFF'
        distance = f"{row.get('distance_km', 0):.2f}"
        speed = row.get('speed', 0)
        location = row.get('location', '-') or '-'
        trip_status = row.get('trip_status', '-') or '-'
        route = row.get('route', '-') or '-'
        onward_route = row.get('onward_route', '-') or '-'
        party = row.get('party', '-') or '-'
        loading_date = row.get('loading_date_fmt', '-') or '-'
        driver = row.get('driver_display', '-') or '-'
        driver_phone = row.get('driver_phone_no', '-') or '-'

        owner_name = row.get('owner_name', '-') or '-'

        html_table += f'''<tr class="{row_class}">
            <td class="vehicle">{row['vehicle_no']}</td>
            <td class="center">{status}</td>
            <td class="center">{speed}</td>
            <td class="center">{distance}</td>
            <td class="left" style="max-width: 200px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{location}">{location}</td>
            <td class="center">{ignition}</td>
            <td class="center">{trip_status}</td>
            <td class="left">{route}</td>
            <td class="left">{onward_route}</td>
            <td class="left" style="max-width: 150px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{party}">{party}</td>
            <td class="center">{loading_date}</td>
            <td class="left">{driver}</td>
            <td class="center">{driver_phone}</td>
            <td class="left">{owner_name}</td>
        </tr>'''

    html_table += '</tbody></table></div></body></html>'

    components.html(html_table, height=table_height, scrolling=True)

    return nearby_df

def main():
    """Main dashboard function"""

    # Header with logo inline with title
    import os
    import base64

    # Load and encode logo
    logo_html = ""
    if os.path.exists("logo1.png"):
        with open("logo1.png", "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/png;base64,{logo_data}" style="height: 50px; vertical-align: middle; margin-right: 15px;">'

    # Title with logo
    st.markdown(f"""
        <div style="text-align: center; padding: 0.3rem 0; margin-top: -20px;">
            <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0; display: inline-block;">
                {logo_html} Swift Live Tracking Dashboard
            </h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; margin-top: -15px; margin-bottom: 0;"><strong>Real-time Vehicle Location & Status Monitoring</strong></p>', unsafe_allow_html=True)

    # Auto-refresh every 60 seconds for live alerts
    # Main vehicle data stays cached for 10 minutes to minimize screen disruption
    st_autorefresh(interval=60000, limit=None, key="vehicle_data_refresh")

    # Load data (cached for 10 minutes - won't reload on every auto-refresh)
    df = load_vehicle_data()

    if len(df) == 0:
        st.error("No vehicle data available")
        return

    # Sidebar filters
    st.sidebar.header("⚙️ Filters")

    # Status filter
    status_options = ['All'] + list(df['status'].dropna().unique())
    selected_status = st.sidebar.selectbox("Filter by Status", status_options)

    # Vehicle search
    vehicle_search = st.sidebar.text_input("Search Vehicle No", "")

    st.sidebar.markdown("---")

    # Nearby location search
    st.sidebar.header("📍 Search Nearby Vehicles")
    enable_nearby_search = st.sidebar.checkbox("Enable Nearby Search", value=False)

    search_lat = None
    search_lon = None
    search_radius = None
    search_location_name = None

    if enable_nearby_search:
        # Choose search method
        search_method = st.sidebar.radio(
            "Search by:",
            ["Place Name", "Coordinates"],
            horizontal=True
        )

        if search_method == "Place Name":
            # Place name search
            place_name = st.sidebar.text_input(
                "Enter Place Name",
                value="",
                placeholder="e.g., Mumbai, Pune, Delhi, Bangalore",
                help="Enter city, town, or any place name"
            )

            if place_name:
                with st.sidebar:
                    with st.spinner(f"🔍 Searching for '{place_name}'..."):
                        search_lat, search_lon, search_location_name = geocode_location(place_name)

                if search_lat and search_lon:
                    st.sidebar.success(f"✅ Found: {search_location_name}")
                    st.sidebar.info(f"📍 Coordinates: ({search_lat:.6f}, {search_lon:.6f})")
                else:
                    st.sidebar.error(f"❌ Could not find '{place_name}'. Please try another name or use coordinates.")
                    search_lat = None
                    search_lon = None
        else:
            # Coordinate search
            col1, col2 = st.sidebar.columns(2)
            with col1:
                search_lat = st.number_input("Latitude", value=20.5937, format="%.6f", key="search_lat")
            with col2:
                search_lon = st.number_input("Longitude", value=78.9629, format="%.6f", key="search_lon")

            st.sidebar.info(f"📍 Coordinates: ({search_lat:.6f}, {search_lon:.6f})")

        if search_lat and search_lon:
            search_radius = st.sidebar.slider("Search Radius (km)", min_value=1, max_value=500, value=50)

    # Apply filters
    filtered_df = df.copy()

    if selected_status != 'All':
        filtered_df = filtered_df[filtered_df['status'] == selected_status]

    if vehicle_search:
        filtered_df = filtered_df[
            filtered_df['vehicle_no'].str.contains(vehicle_search, case=False, na=False)
        ]

    # Show live alert notification on top-right (using cached overspeed data - same as Overspeed tab)
    _, live_overspeed_df = get_overspeed_data()
    overspeed_count = len(live_overspeed_df)
    if overspeed_count > 0:
        st.markdown(f"""
        <style>
            .alert-badge {{
                position: fixed;
                top: 70px;
                right: 20px;
                background: linear-gradient(135deg, #ff5722, #d32f2f);
                color: white;
                padding: 8px 15px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
                z-index: 9999;
                box-shadow: 0 4px 15px rgba(255, 87, 34, 0.4);
                animation: pulse-alert 2s infinite;
                cursor: pointer;
            }}
            .alert-badge:hover {{
                transform: scale(1.05);
            }}
            @keyframes pulse-alert {{
                0%, 100% {{ opacity: 1; box-shadow: 0 4px 15px rgba(255, 87, 34, 0.4); }}
                50% {{ opacity: 0.9; box-shadow: 0 4px 25px rgba(255, 87, 34, 0.7); }}
            }}
        </style>
        <div class="alert-badge">
            🚨 {overspeed_count} Overspeeding → ⚠️ Overspeed
        </div>
        """, unsafe_allow_html=True)

    # Display metrics
    show_overview_metrics(filtered_df)

    st.markdown("---")

    # Tabs
    if enable_nearby_search and search_lat and search_lon:
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🗺️ Live Map", "📍 Nearby Vehicles", "📋 Live Vehicle Details", "🚚 Load Details", "🌙 Night Driving", "⚠️ Overspeed", "🏠 Driver at Home"])
    else:
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🗺️ Live Map", "📋 Live Vehicle Details", "🚚 Load Details", "🌙 Night Driving", "⚠️ Overspeed", "🏠 Driver at Home"])

    with tab1:
        st.subheader("🗺️ Swift Live Vehicle Locations")
        show_map(filtered_df)

        # Legend
        st.markdown("""
        **Legend:**
        - 🟢 Green: Moving (Ignition ON, Speed >= 0)
        - 🟠 Orange: Idle (Ignition OFF, Speed = 0)
        - 🔴 Red: Stopped (No update for 6+ hours AND Ignition OFF)
        """)

    if enable_nearby_search and search_lat and search_lon:
        with tab2:
            # Display search location info
            if search_location_name:
                st.info(f"📍 Searching near: **{search_location_name}**")
            else:
                st.info(f"📍 Searching near: **({search_lat:.6f}, {search_lon:.6f})**")

            # Show nearby vehicles
            nearby_df = show_nearby_vehicles(df, search_lat, search_lon, search_radius)

            # Show nearby vehicles on map
            if len(nearby_df) > 0:
                st.markdown("---")
                st.subheader("🗺️ Nearby Vehicles Map")

                # Create map centered on search location
                m = folium.Map(location=[search_lat, search_lon], zoom_start=10)

                # Add search location marker
                folium.Marker(
                    location=[search_lat, search_lon],
                    popup="Search Location",
                    tooltip="Search Center",
                    icon=folium.Icon(color='blue', icon='crosshairs', prefix='fa')
                ).add_to(m)

                # Add circle showing search radius
                folium.Circle(
                    location=[search_lat, search_lon],
                    radius=search_radius * 1000,  # Convert km to meters
                    color='blue',
                    fill=True,
                    fillOpacity=0.1,
                    popup=f'{search_radius} km radius'
                ).add_to(m)

                # Add vehicle markers
                for idx, row in nearby_df.iterrows():
                    if row['status'] == 'Moving':
                        color = 'green'
                        icon = 'play'
                    elif row['status'] == 'Idle':
                        color = 'orange'
                        icon = 'pause'
                    elif row['status'] == 'Stopped':
                        color = 'red'
                        icon = 'stop'
                    else:
                        color = 'gray'
                        icon = 'question'

                    popup_html = f"""
                    <div style="font-family: Arial; width: 250px;">
                        <h4 style="margin: 0; color: {color};">🚛 {row['vehicle_no']}</h4>
                        <hr style="margin: 5px 0;">
                        <p style="margin: 3px 0;"><b>Distance:</b> {row['distance_km']:.2f} km</p>
                        <p style="margin: 3px 0;"><b>Status:</b> <span style="color: {color};">{row['status']}</span></p>
                        <p style="margin: 3px 0;"><b>Speed:</b> {row['speed']} km/h</p>
                        <p style="margin: 3px 0;"><b>Location:</b> {row['location']}</p>
                    </div>
                    """

                    folium.Marker(
                        location=[float(row['latitude']), float(row['longitude'])],
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{row['vehicle_no']} - {row['distance_km']:.2f} km",
                        icon=folium.Icon(color=color, icon=icon, prefix='fa')
                    ).add_to(m)

                st_folium(m, width=1400, height=600, returned_objects=[])

        with tab3:
            show_vehicle_list(filtered_df)

        with tab4:
            show_load_details()

        with tab5:
            show_status_summary(filtered_df)

        with tab6:
            show_overspeed_alerts(filtered_df)

        with tab7:
            show_driver_at_home(filtered_df)
    else:
        with tab2:
            show_vehicle_list(filtered_df)

        with tab3:
            show_load_details()

        with tab4:
            show_status_summary(filtered_df)

        with tab5:
            show_overspeed_alerts(filtered_df)

        with tab6:
            show_driver_at_home(filtered_df)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<p style='text-align: center; color: gray;'>Swift Live Tracking Dashboard | "
        f"Data Source: FVTS_VEHICLES Database | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: 60s (Alerts) / 10 min (Main Data)</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
