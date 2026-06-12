"""
Load Details Tab - Standalone Module
=====================================
Reusable Load Details tab for any Streamlit dashboard.
Shows vehicle load details with GPS KM tracking, sparkline trends, and filters.

Usage:
    import load_details_tab
    load_details_tab.show_load_details()

    # Or with auto-refresh (every 5 minutes):
    load_details_tab.load_details_fragment()

Requirements:
    - .env file with: Host, UserName, Password, database_name, Port
    - 'party name map.xlsx' in the same directory (optional, for owner mapping)
    - pip install streamlit psycopg2-binary pandas python-dotenv pytz openpyxl
"""

import streamlit as st
import psycopg2
import psycopg2.pool
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# Load environment variables from .env file
load_dotenv()

# ===================== CONFIGURATION =====================

# Vehicles to exclude from the dashboard (normalized format - uppercase, no spaces/dashes)
EXCLUDED_VEHICLES = [
    'NL01AB2275',
    'NL01AG8492O',
]

# Custom owner name mappings (normalized vehicle_no -> owner_name)
CUSTOM_OWNER_MAPPING = {
    # Ranjeet Singh Logistics
    'HR55AQ7919': 'Ranjeet Singh Logistics',
    'HR55AQ7627': 'Ranjeet Singh Logistics',
    'HR55AQ3350': 'Ranjeet Singh Logistics',
    'HR55AM1115': 'Ranjeet Singh Logistics',
    'HR55AQ9263': 'Ranjeet Singh Logistics',
    'HR55AM5463': 'Ranjeet Singh Logistics',
    'HR55AN4660': 'Ranjeet Singh Logistics',
    'HR55AN7527': 'Ranjeet Singh Logistics',
    # R.sai Logistics India Pvt. Ltd.
    'HR55AN5406': 'PICKALL LOGISTICS PRIVATE LIMITED',
    'HR55AM2340': 'R.sai Logistics India Pvt. Ltd.',
    'NL01Q8157': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AM9667': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AP1974': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AM8703': 'PICKALL LOGISTICS PRIVATE LIMITED',
    'HR55AM0907': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AM1370': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AM6059': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AM4278': 'R.sai Logistics India Pvt. Ltd.',
    'HR55AN5307': 'R.sai Logistics India Pvt. Ltd.',
}

# ===================== DATABASE CONNECTION =====================

def get_secret(key, default=None):
    """Get secret from Streamlit secrets (Cloud) or environment variables (local)"""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


def validate_secrets():
    """Check if all required database secrets are configured"""
    required_secrets = ["Host", "UserName", "Password", "database_name"]
    missing = []
    for key in required_secrets:
        value = get_secret(key)
        if not value:
            missing.append(key)
    return missing


@st.cache_resource(show_spinner=False)
def _get_connection_pool():
    """Create a shared connection pool (cached per Streamlit session)"""
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        host=get_secret("Host"),
        user=get_secret("UserName"),
        password=get_secret("Password"),
        database=get_secret("database_name"),
        port=int(get_secret("Port", 5432)),
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=3
    )


def get_database_connection():
    """Get a connection from the pool"""
    missing_secrets = validate_secrets()
    if missing_secrets:
        raise ValueError(
            f"Missing database configuration: {', '.join(missing_secrets)}. "
            f"Please configure these in Streamlit Cloud Secrets or local .env file."
        )

    pool = _get_connection_pool()
    conn = pool.getconn()
    # Ensure connection is alive
    try:
        conn.cursor().execute("SELECT 1")
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        pool.putconn(conn, close=True)
        conn = pool.getconn()
    return conn


# ===================== SPARKLINE SVG =====================

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


# ===================== DATA LOADING =====================

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def load_vehicle_load_details():
    """Load ALL vehicles and map with their load details from swift_trip_log.
    Calculates GPS KM from fvts_vehicles: last_odometer - first_odometer per day.
    """
    connection = None
    try:
        connection = get_database_connection()

        # Calculate KM from fvts_vehicles: last_odometer - first_odometer per day
        query = """
            WITH first_odo AS (
                SELECT DISTINCT ON (UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time))
                    UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')) as vehicle_no,
                    DATE(date_time) as km_date,
                    odometer as first_odometer
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL AND odometer IS NOT NULL AND odometer > 0
                    AND date_time >= DATE_TRUNC('month', CURRENT_DATE)
                ORDER BY UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time), date_time ASC
            ),
            last_odo AS (
                SELECT DISTINCT ON (UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time))
                    UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')) as vehicle_no,
                    DATE(date_time) as km_date,
                    odometer as last_odometer
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL AND odometer IS NOT NULL AND odometer > 0
                    AND date_time >= DATE_TRUNC('month', CURRENT_DATE)
                ORDER BY UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time), date_time DESC
            ),
            daily_km AS (
                SELECT f.vehicle_no, f.km_date,
                    GREATEST(l.last_odometer - f.first_odometer, 0) as daily_km
                FROM first_odo f
                JOIN last_odo l ON f.vehicle_no = l.vehicle_no AND f.km_date = l.km_date
            ),
            vehicle_today_km AS (
                SELECT vehicle_no, COALESCE(daily_km, 0) as today_km_traveled
                FROM daily_km WHERE km_date = CURRENT_DATE
            ),
            vehicle_yesterday_km AS (
                SELECT vehicle_no, COALESCE(daily_km, 0) as yesterday_km_traveled
                FROM daily_km WHERE km_date = CURRENT_DATE - INTERVAL '1 day'
            ),
            vehicle_month_km AS (
                SELECT vehicle_no, COALESCE(SUM(daily_km), 0) as month_km_traveled
                FROM daily_km WHERE km_date >= DATE_TRUNC('month', CURRENT_DATE)
                GROUP BY vehicle_no
            ),
            -- Get all unique vehicles from live data (uses recorded_at index)
            all_vehicles AS (
                SELECT DISTINCT
                    UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')) as vehicle_no
                FROM fvts_vehicles
                WHERE recorded_at >= NOW() - INTERVAL '24 hours'
                    AND vehicle_no IS NOT NULL
            ),
            -- Latest trip per vehicle using ROW_NUMBER based on loading_date DESC
            latest_trip_per_vehicle AS (
                SELECT vehicle_no, trip_status, route, onward_route, party,
                       loading_date, unloading_date, distance, driver_name,
                       driver_code, driver_phone_no, created_at
                FROM (
                    SELECT
                        UPPER(CASE
                            WHEN vehicle_no LIKE '%% %%' THEN
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
                        END) as vehicle_no,
                        COALESCE(NULLIF(trip_status, ''), 'Loaded') as trip_status,
                        route, onward_route, party, loading_date, unloading_date,
                        COALESCE(distance, 0) as distance,
                        driver_name, driver_code, driver_phone_no, created_at,
                        ROW_NUMBER() OVER (
                            PARTITION BY UPPER(CASE
                                WHEN vehicle_no LIKE '%% %%' THEN
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
                            END)
                            ORDER BY loading_date DESC NULLS LAST
                        ) as rn
                    FROM swift_trip_log
                    WHERE vehicle_no IS NOT NULL
                      AND is_active = true
                ) ranked
                WHERE rn = 1
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
            df['normalized_vehicle'] = df['vehicle_no'].fillna('').str.upper().str.replace(' ', '', regex=False).str.replace('-', '', regex=False)
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
            owner_df['normalized_vehicle_no'] = owner_df['RegistrationNo'].fillna('').astype(str).str.upper().str.replace(' ', '', regex=False).str.replace('-', '', regex=False)
            owner_df['owner_name'] = owner_df['PartyName'].apply(
                lambda x: 'Own Vehicle' if x in ['Swift Road Link Pvt. Ltd.', 'Nishant Saini Associates'] else x
            )
            owner_df = owner_df[['normalized_vehicle_no', 'owner_name']]
            df['normalized_vehicle_no_temp'] = df['vehicle_no'].fillna('').astype(str).str.upper().str.replace(' ', '', regex=False).str.replace('-', '', regex=False)
            df = df.merge(owner_df, left_on='normalized_vehicle_no_temp', right_on='normalized_vehicle_no', how='left')
            df['owner_name'] = df.apply(
                lambda row: CUSTOM_OWNER_MAPPING.get(row['normalized_vehicle_no_temp'], row['owner_name']),
                axis=1
            )
            df = df.drop(columns=['normalized_vehicle_no', 'normalized_vehicle_no_temp'], errors='ignore')
        except Exception:
            df['owner_name'] = None

        # Fetch daily GPS KM data for sparkline trends: last_odometer - first_odometer per day
        daily_km_query = """
            WITH first_odo AS (
                SELECT DISTINCT ON (UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time))
                    UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')) as vehicle_no,
                    DATE(date_time) as km_date,
                    odometer as first_odometer
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL AND odometer IS NOT NULL AND odometer > 0
                    AND date_time >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time), date_time ASC
            ),
            last_odo AS (
                SELECT DISTINCT ON (UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time))
                    UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')) as vehicle_no,
                    DATE(date_time) as km_date,
                    odometer as last_odometer
                FROM fvts_vehicles
                WHERE vehicle_no IS NOT NULL AND odometer IS NOT NULL AND odometer > 0
                    AND date_time >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY UPPER(REPLACE(REPLACE(vehicle_no, ' ', ''), '-', '')), DATE(date_time), date_time DESC
            )
            SELECT f.vehicle_no, f.km_date,
                GREATEST(l.last_odometer - f.first_odometer, 0) as daily_km
            FROM first_odo f
            JOIN last_odo l ON f.vehicle_no = l.vehicle_no AND f.km_date = l.km_date
            ORDER BY f.vehicle_no, f.km_date;
        """
        daily_km_df = pd.read_sql_query(daily_km_query, connection)
        daily_km_df['km_date'] = pd.to_datetime(daily_km_df['km_date'])

        # Create a dictionary for quick lookup: {vehicle_no: {date: km}}
        km_lookup = {}
        for vno, km_date, daily_km in zip(daily_km_df['vehicle_no'], daily_km_df['km_date'], daily_km_df['daily_km']):
            if vno not in km_lookup:
                km_lookup[vno] = {}
            km_lookup[vno][km_date.date()] = daily_km

        # Create km_trend sparkline for each vehicle from loading_date to today
        ist = pytz.timezone('Asia/Kolkata')
        today = datetime.now(ist).date()

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
        current_month_start = datetime.now(ist).replace(day=1).date()

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
        now = datetime.now(ist).replace(tzinfo=None)

        # Calculate Current Trip Status (vectorized)
        loading_date = pd.to_datetime(df['loading_date'], errors='coerce')
        unloading_date = pd.to_datetime(df.get('unloading_date'), errors='coerce') if 'unloading_date' in df.columns else pd.Series([pd.NaT] * len(df))
        distance = pd.to_numeric(df.get('distance', 0), errors='coerce').fillna(0)
        tt_days = distance / 400
        tt_date = loading_date + pd.to_timedelta(tt_days, unit='D')

        df['current_trip_status'] = '-'
        df.loc[loading_date.notna() & (distance > 0) & (now < tt_date), 'current_trip_status'] = 'Early'
        df.loc[loading_date.notna() & (distance > 0) & (now >= tt_date), 'current_trip_status'] = 'Delay'
        df.loc[unloading_date.notna() & (unloading_date < now), 'current_trip_status'] = 'Trip End'

        # Calculate trip duration for completed trips
        if 'trip_start_date' in df.columns and 'trip_end_date' in df.columns:
            df['trip_duration_hours'] = ((df['trip_end_date'] - df['trip_start_date']).dt.total_seconds() / 3600).round(1)

        return df

    except Exception as e:
        st.error(f"Error loading load details: {str(e)}")
        return pd.DataFrame()
    finally:
        if connection:
            try:
                _get_connection_pool().putconn(connection)
            except:
                pass


# ===================== DISPLAY =====================

def show_load_details():
    """Display load details table with filters, sparklines, and CSV export"""
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
    display_df = display_df.replace('None', '-')
    display_df = display_df.replace('', '-')

    # Format date columns
    for col in display_df.columns:
        if 'Date' in col and col in filtered_load_df.columns:
            orig_col = [k for k, v in column_mapping.items() if v == col][0]
            if orig_col in filtered_load_df.columns:
                display_df[col] = filtered_load_df[orig_col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) and x != '' and x != 'None' else '-'
                )

    # Format GPS KM columns
    if 'GPS Today KM' in display_df.columns and 'gps_today_km' in filtered_load_df.columns:
        display_df['GPS Today KM'] = filtered_load_df['gps_today_km'].apply(
            lambda x: f"{int(round(x))}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    if 'GPS Yesterday KM' in display_df.columns and 'gps_yesterday_km' in filtered_load_df.columns:
        display_df['GPS Yesterday KM'] = filtered_load_df['gps_yesterday_km'].apply(
            lambda x: f"{int(round(x))}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    if 'GPS Month KM' in display_df.columns and 'gps_month_km' in filtered_load_df.columns:
        display_df['GPS Month KM'] = filtered_load_df['gps_month_km'].apply(
            lambda x: f"{int(round(x))}" if pd.notna(x) and x != '' and x != 'None' and str(x) != '-' else '-'
        )

    # Apply styling with dark mode support
    def style_row(row):
        base_style = 'font-size: 13px; border: 1px solid rgba(100, 100, 100, 0.3); padding: 8px; color: #FFFFFF;'

        if 'Trip Status' in row.index:
            if row['Trip Status'] == 'Completed':
                bg_color = 'background-color: rgba(34, 139, 34, 0.25);'
            elif row['Trip Status'] == 'In Transit':
                bg_color = 'background-color: rgba(30, 144, 255, 0.25);'
            elif row['Trip Status'] == 'Scheduled':
                bg_color = 'background-color: rgba(255, 140, 0, 0.25);'
            else:
                bg_color = 'background-color: rgba(40, 40, 40, 0.3);'
        else:
            bg_color = 'background-color: rgba(40, 40, 40, 0.3);'

        styles = []
        for col in display_df.columns:
            if col == 'Current Trip Status':
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
        table_height = 500

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
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                table-layout: fixed;
            }}
            .load-details-table th {{
                background-color: #1e1e1e;
                color: #FFFFFF;
                font-weight: bold;
                text-align: center;
                border: 1px solid #444;
                padding: 10px 8px;
                font-size: 12px;
                position: sticky;
                top: 0;
                z-index: 10;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }}
            .load-details-table td {{
                border: 1px solid #333;
                padding: 6px 8px;
                color: #FFFFFF;
                background-color: #262730;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }}
            .load-details-table th:nth-child(1) {{ width: 90px; }}
            .load-details-table th:nth-child(2) {{ width: 70px; }}
            .load-details-table th:nth-child(3) {{ width: 100px; }}
            .load-details-table th:nth-child(4) {{ width: 200px; }}
            .load-details-table th:nth-child(5) {{ width: 150px; }}
            .load-details-table th:nth-child(6) {{ width: 220px; }}
            .load-details-table th:nth-child(7) {{ width: 130px; }}
            .load-details-table th:nth-child(8) {{ width: 180px; }}
            .load-details-table th:nth-child(9) {{ width: 100px; }}
            .load-details-table th:nth-child(10) {{ width: 90px; }}
            .load-details-table th:nth-child(11) {{ width: 100px; }}
            .load-details-table th:nth-child(12) {{ width: 100px; }}
            .load-details-table th:nth-child(13) {{ width: 80px; }}
            .load-details-table th:nth-child(14) {{ width: 280px; }}
            .load-details-table th:nth-child(15) {{ width: 120px; }}
            .load-details-table tr:hover td {{
                background-color: rgba(66, 165, 245, 0.2);
            }}
            .load-details-table .vehicle-no {{
                text-align: center;
                font-weight: 700;
                color: #4FC3F7;
            }}
            .load-details-table .center {{
                text-align: center;
            }}
            .load-details-table .left {{
                text-align: left;
            }}
            .load-details-table .route-cell {{
                text-align: left;
            }}
            .load-details-table .party-cell {{
                text-align: left;
            }}
            .load-details-table .driver-cell {{
                text-align: left;
            }}
            .load-details-table .owner-cell {{
                text-align: left;
            }}
            .load-details-table .trend-cell {{
                text-align: center;
                padding: 4px;
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
                width: 100%;
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
        status_cls_map = {'Early': 'early', 'Delay': 'delay', 'Trip End': 'trip-end'}
        center_cols_ld = {'Trip Status', 'Loading Date', 'Driver Phone', 'GPS Today KM', 'GPS Yesterday KM', 'GPS Month KM', 'Days >350 KM'}
        all_cols = list(display_df.columns)
        rows_html_ld = []
        for row in display_df.to_dict('records'):
            cells = []
            for col in all_cols:
                val = row[col]
                if col == 'Vehicle No':
                    cells.append(f'<td class="vehicle-no">{val}</td>')
                elif col == 'KM Trend (Loading to Today)':
                    cells.append(f'<td class="trend-cell">{val}</td>')
                elif col == 'Current Trip Status':
                    status_class = status_cls_map.get(val, '')
                    cells.append(f'<td class="center {status_class}">{val}</td>')
                elif col in ('Route', 'Onward Route'):
                    cells.append(f'<td class="route-cell">{val}</td>')
                elif col == 'Party':
                    cells.append(f'<td class="party-cell">{val}</td>')
                elif col == 'Driver':
                    cells.append(f'<td class="driver-cell">{val}</td>')
                elif col == 'Owner Name':
                    cells.append(f'<td class="owner-cell">{val}</td>')
                elif col in center_cols_ld:
                    cells.append(f'<td class="center">{val}</td>')
                else:
                    cells.append(f'<td class="left">{val}</td>')
            rows_html_ld.append(f'<tr>{"".join(cells)}</tr>')
        html_table += ''.join(rows_html_ld)

        html_table += '</tbody></table></div></body></html>'

        components.html(html_table, height=table_height, scrolling=True)
    else:
        # Use regular dataframe display if no SVG
        styled_df = display_df.style.apply(style_row, axis=1)

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


@st.fragment(run_every=300)  # Refresh every 5 minutes
def load_details_fragment():
    """Load details section with 5-minute auto-refresh"""
    show_load_details()


# ===================== STANDALONE MODE =====================

if __name__ == "__main__":
    st.set_page_config(page_title="Load Details", page_icon="🚚", layout="wide")
    load_details_fragment()
