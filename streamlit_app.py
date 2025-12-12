import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import pydeck as pdk
import json
import geojson
from etrace.load_data import load_from_bq, load_from_bucket
from google.cloud import storage
from shapely.geometry import shape

import streamlit as st
from google.oauth2 import service_account
from google.cloud import bigquery

# Create API client.
credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)

# ---------------------------------------------------------
# E-TRACE: European Tourism Regional Analysis & Climate Effects
# Basic Streamlit Frontend Starter Template
# ---------------------------------------------------------

# Köppen climate classification labels
koppen_labels = {
    "Af": "Tropical rainforest",
    "Am": "Tropical monsoon",
    "Aw": "Tropical savanna",
    "BWh": "Hot desert",
    "BWk": "Cold desert",
    "BSh": "Hot semi-arid",
    "BSk": "Cold semi-arid",
    "Csa": "Hot-summer Mediterranean",
    "Csb": "Warm-summer Mediterranean",
    "Csc": "Cold-summer Mediterranean",
    "Cfa": "Humid subtropical",
    "Cfb": "Oceanic",
    "Cfc": "Subpolar oceanic",
    "Dfa": "Hot-summer continental",
    "Dfb": "Warm-summer continental",
    "Dfc": "Subarctic",
    "Dsa": "Dry-summer continental",
    "Dsb": "Warm-summer continental",
    "Dsc": "Cold-summer continental",
    "ET": "Tundra",
    "EF": "Ice cap",
}

SSP_SCENARIOS = {
    "SSP1 – Sustainability (Taking the Green Road)": {
        "description": "Low emissions, strong environmental policies, shifting towards greener climates",
        "ssp_code": 1,
        "co2": "low",
        "econ_growth": "high",
    },
    "SSP2 – Middle of the Road": {
        "description": "Most likely scenario: moderate emissions, moderate warming",
        "ssp_code": 2,
        "co2": "medium",
        "econ_growth": "medium",
    },
    "SSP3 – Regional Rivalry (A Rocky Road)": {
        "description": "High barriers, slow economic development, climate stress",
        "ssp_code": 3,
        "co2": "high",
        "econ_growth": "low",
    },
    "SSP4 – Inequality (A Road Divided)": {
        "description": "High inequality, limited global cooperation",
        "ssp_code": 4,
        "co2": "high",
        "econ_growth": "imbalanced",
    },
    "SSP5 – Fossil-Fueled Development (Taking the Highway)": {
        "description": "High emissions, strong economic growth, strong warming",
        "ssp_code": 5,
        "co2": "very_high",
        "econ_growth": "very_high",
    },
    "SSP1-2.6 (Low Warming Pathway)": {
        "description": "Low radiative forcing (2.6 W/m2), strong mitigation",
        "rf": 2.6,
    },
    "SSP3-7.0 (High Warming Pathway)": {
        "description": "High radiative forcing (7.0 W/m2), minimal mitigation",
        "rf": 7.0,
    },
}

# Important useful functions


@st.cache_data(ttl=900)  # Cache for 15 minutes
def colormap(v):
    if v is None or pd.isna(v):
        return [200, 200, 200]

    # Turbo colormap implementation
    turbo = [
        [48, 18, 59],
        [53, 41, 133],
        [37, 66, 167],
        [20, 92, 157],
        [16, 120, 130],
        [32, 144, 92],
        [68, 164, 54],
        [112, 181, 25],
        [160, 194, 9],
        [210, 203, 8],
        [255, 209, 28],
        [255, 189, 51],
        [255, 158, 73],
        [255, 116, 95],
        [255, 64, 112],
        [237, 5, 121],
        [203, 0, 122],
        [155, 0, 112],
        [102, 0, 92],
        [56, 0, 63],
    ]

    idx = min(int(v * (len(turbo) - 1)), len(turbo) - 1)
    return turbo[idx]


@st.cache_data(ttl=900)  # Cache for 15 minutes
def highlight_selected_column(df, column_name):
    """
    Highlights selected column with a special color
    """
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    if column_name in df.columns:
        styles[column_name] = (
            "background-color: #FFD700; color: black; font-weight: bold"
        )
    return styles


@st.cache_data(ttl=900)  # Cache for 15 minutes
def compute_centroid(feature):
    geom = shape(feature["geometry"])
    c = geom.centroid
    return c.y, c.x  # lat, lon order for pydeck


@st.cache_data(ttl=900)  # Cache for 15 minutes
def extract_all_coords(geometry):
    coords = geometry["coordinates"]
    geom_type = geometry["type"]

    all_points = []

    if geom_type == "Polygon":
        # coords = [ring1, ring2, ...]
        for ring in coords:
            all_points.extend(ring)

    elif geom_type == "MultiPolygon":
        # coords = [poly1, poly2, ...]
        for poly in coords:
            for ring in poly:
                all_points.extend(ring)

    return all_points


@st.cache_data
def load_predictions():
    bucket = "etrace-data"
    blob = "data/raw_data/FINAL_DATAFRAME_PREDICTIONS_V1.csv"
    local_path = "/tmp/FINAL_DATAFRAME_PREDICTIONS_V1.csv"

    csv_path = load_from_bucket(bucket, blob, local_path)
    df = pd.read_csv(csv_path)

    # Normalise column names (optional but handy)
    df.columns = [c.strip() for c in df.columns]

    return df


# Page configuration
st.set_page_config(page_title="E-TRACE Dashboard", page_icon="🌍", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 4rem;'>
        🌍 <span style='color: #1f77b4;'>E</span>-TRACE Dashboard
    </h1>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    """
    <p style='text-align: center; font-size: 1.2rem;'>
    Welcome to <span style='color: #1f77b4; font-weight: bold;'>E</span>-TRACE —
    <span style='color: #1f77b4; font-weight: bold;'>E</span>uropean
    <span style='color: #1f77b4; font-weight: bold;'>T</span>ourism
    <span style='color: #1f77b4; font-weight: bold;'>R</span>egional
    <span style='color: #1f77b4; font-weight: bold;'>A</span>nalysis &
    <span style='color: #1f77b4; font-weight: bold;'>C</span>limate
    <span style='color: #1f77b4; font-weight: bold;'>E</span>ffects.
    </p>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Köppen-Geiger Climate Classification 101
# ---------------------------------------------------------
with st.expander("🌡️ What is the Köppen-Geiger Climate Classification?", expanded=False):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        The **Köppen-Geiger climate classification** is one of the most widely used systems for
        categorizing the world's climates. Developed by climatologist Wladimir Köppen in 1884
        and later refined by Rudolf Geiger, it divides climates into five main groups based on
        temperature and precipitation patterns.
        """
        )

        st.markdown("#### 🌍 Five Main Climate Groups")

        st.markdown(
            """
        🌴 **A - Tropical**
        Hot and humid year-round with abundant rainfall

        🏜️ **B - Dry**
        Arid and semi-arid regions with low precipitation

        🌤️ **C - Temperate**
        Moderate temperatures with distinct seasons (like most of Europe)

        ❄️ **D - Continental**
        Cold winters and warm summers with significant seasonal variation

        🧊 **E - Polar**
        Extremely cold climates with little vegetation (tundra and ice caps)
        """
        )

        st.markdown(
            """
        Each main group is further subdivided with additional letters indicating specific characteristics
        like precipitation patterns (**f** = fully humid, **s** = dry summer, **w** = dry winter) and
        temperature ranges (**a** = hot summer, **b** = warm summer, **c** = cool summer, etc.).
        """
        )

    with col2:
        st.success(
            """
        **🎯 Why it matters for E-TRACE**

        Climate zones directly influence tourism patterns, seasonal demand, and visitor preferences.

        Understanding how climate distributions change over time helps us:

        - 📈 Predict shifts in regional tourism attractiveness
        - 🏖️ Identify emerging seasonal patterns
        - 🌡️ Track climate change impacts on tourism
        - 🎿 Plan climate adaptation strategies
        - 💡 Forecast future visitor preferences
        """
        )


st.divider()

# ---------------------------------------------------------
# Dataset Api Call to Load Data
# ---------------------------------------------------------


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_main_dataset():
    """Load the main dataset from BigQuery with caching."""
    return load_from_bq(
        "SELECT * FROM `aklewagonproject.etrace.cleaned_final_jaume_dataset_newnames`"
    )


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_nuts2_geo():
    """Load NUTS2 GeoJSON from Google Cloud Storage with caching."""
    client = storage.Client(credentials=credentials)
    bucket = client.bucket("etrace-data")
    blob = bucket.blob("data/raw_data/nuts2_geo.geojson")
    geojson_bytes = blob.download_as_bytes()
    return geojson.loads(geojson_bytes.decode("utf-8"))


@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_predictions_v5():
    """Load and parse predictions V5 CSV with complex formatting."""
    bucket = "etrace-data"
    blob = "data/raw_data/FINAL_DATAFRAME_PREDICTIONS_V5.csv"
    local_path = "/tmp/FINAL_DATAFRAME_PREDICTIONS_V5.csv"

    csv_path = load_from_bucket(bucket, blob, local_path)

    # ---- READ RAW LINES ----
    with open(csv_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f.readlines()]

    # Remove outer quotes on each row (important!)
    clean_lines = [line.strip('"') for line in raw_lines]

    # Split each row by comma
    rows = [line.split(",") for line in clean_lines]

    # Convert to DataFrame
    pred_df = pd.DataFrame(rows)

    # First row is the header → promote it
    pred_df.columns = pred_df.iloc[0]  # header row
    pred_df = pred_df.iloc[1:].reset_index(drop=True)

    # Clean column names
    pred_df.columns = pred_df.columns.str.strip().str.replace("\ufeff", "")

    # Clean NUTS_ID and year types
    pred_df["NUTS_ID"] = pred_df["NUTS_ID"].str.strip()
    pred_df["Scenario"] = pred_df["Scenario"].str.strip()
    pred_df["Year"] = pd.to_numeric(pred_df["Year"], errors="coerce").astype(int)

    # Clean column names again
    pred_df.columns = pred_df.columns.str.strip().str.replace("\ufeff", "")

    # Fix leading commas if any
    pred_df.columns = pred_df.columns.str.lstrip(",")

    # Define column names
    SCENARIO_COL = "Scenario"
    GEO_COL = "NUTS_ID"
    YEAR_COL = "Year"
    NIGHTS_COL = "Overnight_Stays"
    NAME_COL = "NUTS_NAME"

    # Clean column types
    pred_df[SCENARIO_COL] = pred_df[SCENARIO_COL].str.strip()
    pred_df[GEO_COL] = pred_df[GEO_COL].str.strip()
    pred_df[NAME_COL] = pred_df[NAME_COL].str.strip()

    # Convert to numeric
    pred_df[YEAR_COL] = pd.to_numeric(pred_df[YEAR_COL], errors="coerce").astype(int)
    pred_df[NIGHTS_COL] = pd.to_numeric(pred_df[NIGHTS_COL], errors="coerce")

    return pred_df


df = load_main_dataset()

st.session_state["df"] = df

# ---------------------------------------------------------
# Sidebar Navigation (for future pages)
# ---------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Exploration", "Mapping", "Model"])

# ---------------------------------------------------------
# Exploration Page (Merged with Welcome Text)
# ---------------------------------------------------------

if page == "Exploration":

    st.markdown(
        """
    Welcome to the E-TRACE Dashboard! Use the sidebar to navigate between different sections:
    - **Exploration**: Dive into regional data and visualize time-series indicators.
    - **Mapping**: Explore interactive maps of NUTS-2 regions with various socioeconomic and climate variables.
    - **Model**: Experiment with an XGBoost-model based on future climate and socioeconomic scenarios.

    Get started by exploring historical figures below and the select the other pages from the sidebar!
    """
    )

    st.divider()

    st.header("🔎 Region Explorer")
    st.markdown("Select a NUTS-2 region to explore its time-series indicators.")

    if df is None:
        st.warning(
            "Data could not be loaded. Please check your connection or BigQuery permissions."
        )
    else:
        # ------------------------
        # Region Selector
        # ------------------------
        regions = sorted(df["NUTS_NAME"].dropna().unique())
        region = st.selectbox("Select a NUTS-2 region:", regions)

        # Using capitalized 'Year' as per new schema
        df_region = df[df["NUTS_NAME"] == region].sort_values("Year")

        # Build a display-only copy: hide metadata + all-zero climate columns
        drop_meta_cols = ["is_coastal", "NUTS_NAME_completo", "CNTR_CODE"]

        zero_climate_cols = [
            c
            for c in df_region.columns
            if c in koppen_labels and df_region[c].abs().sum() == 0
        ]

        df_region_display = df_region.drop(
            columns=drop_meta_cols + zero_climate_cols,
            errors="ignore",
        )

        st.subheader(f"📍 Region: **{region}**")
        st.write(df_region_display)

        st.divider()

        # ------------------------
        # Time-series plots
        # ------------------------

        # Updated Section Header with Region Name
        st.header(f"{region} Historical Metrics")

        numeric_columns = df_region.select_dtypes(include=["float64", "int64"]).columns

        # 1. Define Core Variables (Order matters: Tourism, GDP, Pop, Employment)
        core_vars_map = {
            "Tourism (Nights Spent)": "Overnight_Stays",
            "GDP per Capita": "GDP_per_Capita",
            "Population": "Population",
            "Employment Rate": "Employment_Rate",
            "Precipitation (rr)": "rr",
            "Temperature (Mean - tg)": "tg",
            "Temperature (Max - tx)": "tx",
            "Temperature (Min - tn)": "tn",
        }

        # 2. Identify Climate Variables (Double-letter codes like Dfb, Cfa)
        # We only include those present in the data and koppen_labels
        climate_vars_list = [col for col in numeric_columns if col in koppen_labels]

        # 3. Build the final ordered dictionary of variables to plot
        #    Order: Core Vars -> Climate Vars
        final_plot_vars = {}

        # Add Core Vars first
        for label, col in core_vars_map.items():
            if col in df_region.columns:
                final_plot_vars[label] = col

        # Add Climate Vars second
        for c in climate_vars_list:
            final_plot_vars[f"Climate: {c}"] = c

        # 4. Plotting Loop
        for label, col in final_plot_vars.items():
            # Skip if column not in df (safety check)
            if col not in df_region.columns:
                continue

            # Skip variables that are zero for all years in this region
            if df_region[col].abs().sum() == 0:
                continue

            # Determine pretty label for the chart title
            if col in koppen_labels:
                # e.g., "Climate: Oceanic (Cfb)"
                pretty_label = f"Climate: {koppen_labels.get(col, col)} ({col})"
            else:
                pretty_label = label

            # Create figure with simplified title
            fig = px.line(
                df_region,
                x="Year",
                y=col,
                markers=True,
                title=pretty_label,  # Title is just the variable name now
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# Mapping Page
# ---------------------------------------------------------
elif page == "Mapping":

    st.header("🗺️ NUTS-2 Regional Map Visualization")

    df_clean = st.session_state.get("df")

    if df_clean is None:
        st.warning("Something went wrong uploading the data.")
        st.stop()

    # Load NUTS2 GeoJSON from google cloud (cached)
    nuts2_geo = load_nuts2_geo()

    # Ensure 'NUTS_ID' column exists
    if "NUTS_ID" not in df_clean.columns:
        st.error("The dataset does not contain a 'NUTS_ID' column.")
        st.stop()

    # Variables available for mapping
    map_numeric_cols = [
        c for c in df_clean.columns if df_clean[c].dtype in ["float64", "int64"]
    ]
    map_numeric_cols = df_clean.select_dtypes(include=["float64", "int64"]).columns
    selected_var = st.selectbox("Variable to visualize:", map_numeric_cols)
    st.write(f"### Visualizing: **{selected_var}**")

    years = sorted(df_clean["Year"].unique())
    selected_year = st.slider("Select Year", min(years), max(years), min(years))

    df_year = df_clean[df_clean["Year"] == selected_year]

    # map style
    map_style_choice = st.radio("Map Style", ["Flat Map"], horizontal=True)

    all_geo2 = []
    for each in nuts2_geo["features"]:
        if each.properties["LEVL_CODE"] == 2:
            all_geo2.append(each)
    nuts2_geo["features"] = all_geo2

    # Attach values to GeoJSON
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]

        if not match.empty:
            try:
                feature["properties"][selected_var] = float(
                    match[selected_var].values[0]
                )
            except Exception as e:
                feature["properties"][selected_var] = None
        else:
            feature["properties"][selected_var] = None

    # -------------------------------------------
    # Compute normalized color values
    # -------------------------------------------

    vmin = df_year[selected_var].min()
    vmax = df_year[selected_var].max()

    df_year["scaled_value"] = (df_year[selected_var] - vmin) / (vmax - vmin)

    # -------------------------------------------
    # SAFE Normalize values
    # -------------------------------------------

    vals = df_year[selected_var].astype(float)
    vmin = vals.min()
    vmax = vals.max()

    # Avoid division by zero
    if vmin == vmax:
        df_year["scaled_value"] = 0
    else:
        df_year["scaled_value"] = (vals - vmin) / (vmax - vmin)

    # Replace NaN with 0.5 (neutral mid-value)
    df_year["scaled_value"] = df_year["scaled_value"].fillna(0.5)
    df_year["color"] = df_year["scaled_value"].apply(colormap)

    # -------------------------------------------
    # Attach COLOR to GeoJSON features
    # -------------------------------------------
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]

        if not match.empty:
            feature["properties"]["color"] = match["color"].values[0]
        else:
            feature["properties"]["color"] = [180, 180, 180]  # grey fallback

    # -------------------------------------------
    # Data Preview Section
    # -------------------------------------------
    st.write("### Data Preview")

    # 1. Identify all-zero columns in the current year slice
    zero_cols = [
        c
        for c in df_year.columns
        if df_year[c].dtype in ["float64", "int64"] and df_year[c].abs().sum() == 0
    ]

    # 2. Define specific columns to drop from the screenshot instructions
    # (Screenshot 1: NUTS_NAME_completo, CNTR_CODE, is_coastal, Overnight_Stays, scaled_value, color)
    manual_drop_cols = [
        "NUTS_NAME_completo",
        "CNTR_CODE",
        "is_coastal",
        "Overnight_Stays",
        "scaled_value",
        "color",
    ]

    # 3. Combine both lists
    cols_to_drop = list(set(zero_cols + manual_drop_cols))

    # 4. Create display dataframe
    df_display = df_year.drop(columns=cols_to_drop, errors="ignore")

    st.dataframe(
        df_display.style.apply(
            highlight_selected_column, column_name=selected_var, axis=None
        ),
        use_container_width=True,
        height=400,
    )
    st.write(df_display.shape)

    # -------------------------------------------
    # Legend Section (Renamed & Cleaned)
    # -------------------------------------------

    # Changed "Color Legend" to the variable name itself
    st.markdown(f"### 📊 {selected_var}")

    # Statistics with comma formatting (e.g., 1,234.56)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min value", f"{vmin:,.2f}")
    with col2:
        st.metric("Average value", f"{df_year[selected_var].mean():,.2f}")
    with col3:
        st.metric("Max value", f"{vmax:,.2f}")

    # Heat bar with formatted min/max labels below
    st.markdown(
        f"""
    <div style="background: linear-gradient(to right,
        rgb(48,18,59), rgb(37,66,167), rgb(16,120,130),
        rgb(68,164,54), rgb(160,194,9), rgb(255,209,28),
        rgb(255,158,73), rgb(255,64,112), rgb(203,0,122));
        height: 30px; border-radius: 5px; margin: 10px 0;">
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{vmin:,.2f}</span>
        <span>{vmax:,.2f}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    height_scale = 5000
    df_year["height"] = df_year["scaled_value"] * height_scale

    # Attaching height to geojson
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]

        if not match.empty:
            feature["properties"]["color"] = match["color"].values[0]
            feature["properties"]["height"] = float(match["height"].values[0])
        else:
            feature["properties"]["color"] = [180, 180, 180]
            feature["properties"]["height"] = 0

    # PyDeck layer
    layer = pdk.Layer(
        "GeoJsonLayer",
        nuts2_geo,
        opacity=0.75,
        stroked=True,
        filled=True,
        get_fill_color="color",
        pickable=True,
    )

    data_layer = pdk.Layer("DataLayer", data=df_year)

    # -------------------------------------------
    # Build DATA FOR 3D COLUMN LAYER
    # -------------------------------------------

    columns_data = []

    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]

        lat, lon = compute_centroid(feature)

        if not match.empty:
            value = match[selected_var].values[0]
            scaled = float(match["scaled_value"].values[0])
            height = scaled * 75000
        else:
            value = None
            height = 0

        columns_data.append(
            {
                "NUTS_ID": geo_id,
                "value": value,
                "height": height,
                "lat": lat,
                "lon": lon,
            }
        )

    column_layer = pdk.Layer(
        "ColumnLayer",
        data=columns_data,
        get_position=["lon", "lat"],
        get_elevation="height",
        elevation_scale=1,
        radius=20000,
        get_fill_color=[255, 140, 0],
        pickable=True,
        auto_highlight=True,
    )

    extruded_layer = pdk.Layer(
        "GeoJsonLayer",
        nuts2_geo,
        opacity=0.9,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color="color",
        get_elevation="height",
        elevation_scale=1,
        pickable=True,
    )

    if map_style_choice == "3D Stacked Map":
        view_state = pdk.ViewState(
            latitude=50, longitude=10, zoom=3.4, pitch=45, bearing=0
        )
    else:
        view_state = pdk.ViewState(
            latitude=50,
            longitude=10,
            zoom=3.3,
            bearing=0,
            pitch=35,
        )

    if map_style_choice == "Flat Map":
        layer_to_show = layer
    else:
        layer_to_show = extruded_layer

    if map_style_choice == "Flat Map":
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer, data_layer],
                tooltip={
                    "text": f"NUTS: {{NUTS_ID}}\n{selected_var}: {{{selected_var}}}"
                },
            )
        )
    else:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                layers=[column_layer, layer_to_show],
                initial_view_state=view_state,
                tooltip={
                    "text": f"NUTS: {{NUTS_ID}}\n{selected_var}: {{{selected_var}}}\n"
                    f"{selected_var}: {{value}}"
                },
            )
        )

# --------------------------
# Model Page
# --------------------------

elif page == "Model":

    st.header("🤖 Predictive Model")

    # --------------------------
    # Load predictions dataframe
    # --------------------------

    # Load NUTS2 GeoJSON from google cloud (cached)
    nuts2_geo = load_nuts2_geo()

    # Load predictions data (cached)
    pred_df = load_predictions_v5()

    # Column definitions for compatibility
    SCENARIO_COL = "Scenario"
    GEO_COL = "NUTS_ID"
    YEAR_COL = "Year"
    NIGHTS_COL = "Overnight_Stays"
    NAME_COL = "NUTS_NAME"

    # Safety checks
    missing_cols = [
        c
        for c in [SCENARIO_COL, GEO_COL, YEAR_COL, NIGHTS_COL]
        if c not in pred_df.columns
    ]
    if missing_cols:
        st.error(f"These columns are missing in prediction table: {missing_cols}")
        st.stop()

    # ---------------------------------
    # User controls: SSP, region, year
    # ---------------------------------
    st.subheader("Select scenario and region")

    scenario_list = sorted(pred_df[SCENARIO_COL].unique())
    selected_ssp = st.selectbox("Choose SSP scenario:", scenario_list)

    df_ssp = pred_df[pred_df[SCENARIO_COL] == selected_ssp]

    region_options = df_ssp[[GEO_COL]].drop_duplicates().sort_values(GEO_COL)

    # Show nice name but keep the code
    region_label_to_code = {
        f"{row[GEO_COL]} ({row[GEO_COL]})": row[GEO_COL]
        for _, row in region_options.iterrows()
    }

    selected_label = st.selectbox(
        "NUTS-2 region:",
        list(region_label_to_code.keys()),
    )

    selected_geo = region_label_to_code[selected_label]

    df_region = df_ssp[df_ssp[GEO_COL] == selected_geo]

    year_min = int(df_region[YEAR_COL].min())
    year_max = int(df_region[YEAR_COL].max())
    selected_year = st.slider(
        "Prediction year:",
        min_value=year_min,
        max_value=year_max,
        value=year_min,
        step=1,
    )

    df_map = pred_df[
        (pred_df[SCENARIO_COL] == selected_ssp) & (pred_df[YEAR_COL] == selected_year)
    ][[GEO_COL, NIGHTS_COL]].copy()

    # -------------------------------
    # Get the prediction for the row
    # -------------------------------
    row_mask = (
        (pred_df[SCENARIO_COL] == selected_ssp)
        & (pred_df[GEO_COL] == selected_geo)
        & (pred_df[YEAR_COL] == selected_year)
    )
    row = pred_df[row_mask]

    if row.empty:
        st.warning("No prediction found for this combination of SSP, region and year.")
        st.stop()

    row = row.iloc[0]
    nights_pred = float(row[NIGHTS_COL])
    region_name = row[GEO_COL]
    region_name_col = row[NAME_COL]

    # --- FIX START ---
    # Use the selected_geo from the dropdown, not the first row of the entire dataframe
    nuts_id = selected_geo
    # --- FIX END ---

    # Extracting only the selected nuts region
    region_feature = [
        feat
        for feat in nuts2_geo["features"]
        if feat["properties"]["NUTS_ID"] == nuts_id
    ]

    st.subheader(f"📍 Region: **{region_name_col}**")

    ### --- FULL MAP WITH SELECTED REGION HIGHLIGHTED --- ###

    # Background polygons (all regions)
    background_data = []
    for feat in nuts2_geo["features"]:
        geom = feat["geometry"]

        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
            background_data.append({"polygon": coords})
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                background_data.append({"polygon": poly[0]})

    background_layer = pdk.Layer(
        "PolygonLayer",
        data=background_data,
        get_polygon="polygon",
        get_fill_color=[200, 200, 200, 80],
        get_line_color=[80, 80, 80, 160],
        pickable=False,
    )

    # ---- Highlight selected region ----
    region_feature = next(
        (
            feat
            for feat in nuts2_geo["features"]
            if feat["properties"]["NUTS_ID"] == nuts_id
        ),
        None,
    )

    if region_feature is None:
        st.error(f"Region {nuts_id} not found in GeoJSON.")
    else:
        geom = region_feature["geometry"]
        highlight_data = []

        if geom["type"] == "Polygon":
            highlight_data.append({"polygon": geom["coordinates"][0]})
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                highlight_data.append({"polygon": poly[0]})

        highlight_layer = pdk.Layer(
            "PolygonLayer",
            data=highlight_data,
            get_polygon="polygon",
            get_fill_color=[60, 140, 230, 200],
            get_line_color=[0, 0, 0, 255],
            line_width_min_pixels=2,
            pickable=False,
        )

        # Compute center of first polygon
        poly = highlight_data[0]["polygon"]
        center_lon = sum([p[0] for p in poly]) / len(poly)
        center_lat = sum([p[1] for p in poly]) / len(poly)

        view_state = pdk.ViewState(
            longitude=center_lon,
            latitude=center_lat,
            zoom=4,
        )

        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                layers=[background_layer, highlight_layer],
                initial_view_state=view_state,
            )
        )

    st.divider()

    # Show a metric
    st.subheader("Predicted tourism")
    st.metric(
        label=f"Predicted nights stayed in {region_name} ({selected_geo})",
        value=f"{nights_pred:,.0f}",
        delta=None,
        help=f"Scenario: {selected_ssp}, year: {selected_year}",
    )

    st.markdown("---")

    df_year = pred_df[pred_df["Year"] == selected_year]

    # 3D stacked map style
    map_style_choice = st.radio(
        "Map Style", ["Flat Map", "3D Stacked Map"], horizontal=True
    )

    all_geo2 = []
    for each in nuts2_geo["features"]:
        if each.properties["LEVL_CODE"] == 2:
            all_geo2.append(each)
    nuts2_geo["features"] = all_geo2

    # Attach values to GeoJSON
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]
        match = match.loc[match[SCENARIO_COL] == selected_ssp]

        if not match.empty:
            feature["properties"][NIGHTS_COL] = float(match[NIGHTS_COL].values[0])
        else:
            feature["properties"][NIGHTS_COL] = None

    # -------------------------------------------
    # Compute normalized color values
    # -------------------------------------------

    vmin = df_year[NIGHTS_COL].min()
    vmax = df_year[NIGHTS_COL].max()

    df_year["scaled_value"] = (df_year[NIGHTS_COL] - vmin) / (vmax - vmin)

    # -------------------------------------------
    # SAFE Normalize values
    # -------------------------------------------

    vals = df_year[NIGHTS_COL].astype(float)

    vmin = vals.min()
    vmax = vals.max()

    # Avoid division by zero: if no variation, fill with zero
    if vmin == vmax:
        df_year["scaled_value"] = 0
    else:
        df_year["scaled_value"] = (vals - vmin) / (vmax - vmin)

    # Replace NaN with 0.5 (neutral mid-value)
    df_year["scaled_value"] = df_year["scaled_value"].fillna(0.5)

    df_year["color"] = df_year["scaled_value"].apply(colormap)

    # -------------------------------------------
    # Attach COLOR to GeoJSON features
    # -------------------------------------------
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_map[df_map[GEO_COL] == geo_id]
        # match = match.loc[match[SCENARIO_COL] == selected_ssp]

        if not match.empty:
            try:
                feature["properties"]["color"] = match["color"].values[0]
                feature["properties"]["overnight"] = float(match[NIGHTS_COL].values[0])
            except:
                feature["properties"]["overnight"] = None

        else:
            feature["properties"]["color"] = [200, 200, 200, 30]  # grey
            feature["properties"]["overnight"] = None

    st.markdown("### Overnight Stays Prediction")

    # statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min value", f"{vmin:,.2f}")
    with col2:
        st.metric("Average value", f"{df_year[NIGHTS_COL].mean():,.2f}")
    with col3:
        st.metric("Max value", f"{vmax:,.2f}")

    # visual grad ( nabla)
    st.markdown(
        f"""
    <div style="background: linear-gradient(to right,
        rgb(48,18,59), rgb(37,66,167), rgb(16,120,130),
        rgb(68,164,54), rgb(160,194,9), rgb(255,209,28),
        rgb(255,158,73), rgb(255,64,112), rgb(203,0,122));
        height: 30px; border-radius: 5px; margin: 10px 0;">
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{vmin:.2f}</span>
        <span style="font-weight: bold;">{NIGHTS_COL}</span>
        <span>{vmax:.2f}</span>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # height column necessary for stacked maps
    height_scale = 5000
    df_year["height"] = df_year["scaled_value"] * height_scale
    #

    # Attaching height to geojson
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]
        match = match.loc[match[SCENARIO_COL] == selected_ssp]

        if not match.empty:
            feature["properties"]["color"] = match["color"].values[0]
            feature["properties"]["height"] = float(match["height"].values[0])
        else:
            feature["properties"]["color"] = [180, 180, 180]
            feature["properties"]["height"] = 0

    # PyDeck layer
    layer = pdk.Layer(
        "GeoJsonLayer",
        nuts2_geo,
        opacity=0.75,
        stroked=True,
        filled=True,
        get_fill_color="color",
        pickable=True,
    )

    #  Nikita line
    df_year = df_year.astype(float, errors="ignore").drop(columns=[None])
    # df_year = df_year.loc[df_year[SCENARIO_COL] == selected_ssp]

    # PyDeck layer
    data_layer = pdk.Layer(
        "DataLayer",
        data=df_year,
    )

    # -------------------------------------------
    # Build DATA FOR 3D COLUMN LAYER
    # -------------------------------------------

    columns_data = []

    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["NUTS_ID"] == geo_id]
        match = match.loc[match[SCENARIO_COL] == selected_ssp]

        lat, lon = compute_centroid(feature)

        if not match.empty:
            value = match[NIGHTS_COL].values[0]
            scaled = float(match["scaled_value"].values[0])
            height = scaled * 75000  # adjust height multiplier
        else:
            value = None
            height = 0

        columns_data.append(
            {
                "NUTS_ID": geo_id,
                "value": value,
                "height": height,
                "lat": lat,
                "lon": lon,
            }
        )

    # Column layer
    column_layer = pdk.Layer(
        "ColumnLayer",
        data=columns_data,
        get_position=["lon", "lat"],
        get_elevation="height",  # height of each bar
        elevation_scale=1,
        radius=20000,  # size of the column footprint
        get_fill_color=[255, 140, 0],  # orange columns
        pickable=True,
        auto_highlight=True,
    )

    # 3D Stacked layer
    extruded_layer = pdk.Layer(
        "GeoJsonLayer",
        nuts2_geo,
        opacity=0.9,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color="color",
        get_elevation="height",
        elevation_scale=1,
        pickable=True,
    )

    # Deciding which view to use depending on the selected map
    if map_style_choice == "3D Stacked Map":
        view_state = pdk.ViewState(
            latitude=50, longitude=10, zoom=3.4, pitch=45, bearing=0
        )
    else:
        view_state = pdk.ViewState(
            latitude=50,
            longitude=10,
            zoom=3.3,
            bearing=0,
            pitch=35,
        )

    if map_style_choice == "Flat Map":
        layer_to_show = layer
    else:
        layer_to_show = extruded_layer

    if map_style_choice == "Flat Map":
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v9",
                initial_view_state=view_state,
                layers=[layer, data_layer],
                tooltip={"text": f"NUTS: {{NUTS_ID}}\n{NIGHTS_COL}: {{{NIGHTS_COL}}}"},
            )
        )

    else:
        st.pydeck_chart(
            pdk.Deck(
                map_style="mapbox://styles/mapbox/dark-v10",
                layers=[column_layer, layer_to_show],
                initial_view_state=view_state,
                tooltip={
                    "text": f"NUTS: {{NUTS_ID}}\n{NIGHTS_COL}: {{{NIGHTS_COL}}}\n"
                    f"{NIGHTS_COL}: {{value}}"
                },
            )
        )

    st.subheader("Change in Tourism Compared to 2020")

    # -----------------------------------------
    # 1) Extract baseline (2020) and selected year
    # -----------------------------------------

    baseline_year = 2020

    df_base = df[df["Year"] == baseline_year][[GEO_COL, NIGHTS_COL]].copy()
    df_base = df_base.rename(columns={NIGHTS_COL: "base_nights"})

    df_future = pred_df[pred_df[YEAR_COL] == selected_year][
        [GEO_COL, NIGHTS_COL]
    ].copy()
    df_future = df_future.rename(columns={NIGHTS_COL: "future_nights"})

    df_2020 = df[df["Year"] == 2020][[GEO_COL, NIGHTS_COL]].copy()
    difference_df = df_future.merge(df_2020, on=["NUTS_ID"], how="inner")
    difference_df["delta"] = (
        difference_df["future_nights"] - difference_df["Overnight_Stays"]
    )

    # Normalise delta for coloring (min → -1, max → +1)
    max_abs = max(abs(difference_df["delta"].min()), abs(difference_df["delta"].max()))
    difference_df = difference_df.dropna(subset=["delta"])

    difference_df["scaled"] = difference_df["delta"] / max_abs

    # Diverging color scale: red (decrease) → yellow (neutral) → green (increase)
    def diverging_color(v):
        """
        v in [-1, 1]
        """
        if pd.isna(v):
            return [200, 200, 200, 80]

        if v < 0:
            # Negative → red to yellow
            r = 255
            g = int(200 * (1 + v))  # v = -1 → g=0, v=0 → g=200
            b = 0
        else:
            # Positive → yellow to green
            r = int(255 * (1 - v))  # v=1 → 0, v=0 → 255
            g = 255
            b = 0

        return [r, g, b, 160]

    difference_df["color"] = difference_df["scaled"].apply(diverging_color)

    # Attach delta + color to geojson
    color_count = 0
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]

        match = difference_df[difference_df[GEO_COL] == geo_id]

        if not match.empty:
            feature["properties"]["delta"] = float(match["delta"].values[0])
            feature["properties"]["color"] = match["color"].values[0]
            color_count += 1
        else:
            feature["properties"]["delta"] = None
            feature["properties"]["color"] = [220, 220, 220, 80]

    st.subheader(f"Overnight Stays: {baseline_year} → {selected_year}")

    import numpy as np

    delta_layer = pdk.Layer(
        "GeoJsonLayer",
        nuts2_geo,
        opacity=0.75,
        stroked=True,
        filled=True,
        get_fill_color="color",
        get_line_color=[80, 80, 80],
        line_width_min_pixels=1,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=50,
        longitude=10,
        zoom=3.3,
        bearing=0,
        pitch=30,
    )

    st.pydeck_chart(
        pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            layers=[delta_layer],
            initial_view_state=view_state,
            tooltip={"text": "NUTS: {NUTS_ID}\nΔ Nights: {delta}"},
        )
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.write(
    "E-TRACE • European Tourism Regional Analysis & Climate Effects • Built with ❤️ using Streamlit"
)
