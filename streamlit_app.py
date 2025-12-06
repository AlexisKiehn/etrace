import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import pydeck as pdk
import json
import geojson
from etrace.load_data import load_from_bq, load_from_bucket
from google.cloud import storage

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
    "EF": "Ice cap"
}

# Page configuration
st.set_page_config(
    page_title="E-TRACE Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align: center; font-size: 4rem;'>
        🌍 <span style='color: #1f77b4;'>E</span>-TRACE Dashboard
    </h1>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown("""
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
    unsafe_allow_html=True
)
st.markdown("""
This is the first version of our interactive website, where we will:
- Upload and explore the dataset
- Visualize trends (tourism activity, population, GDP, employment, climate variables…)
- Build predictive insights
- Interactively compare NUTS-2 regions

This page is just a starting point — we will expand it into multiple tabs and visualizations.
""")

# ---------------------------------------------------------
# Köppen-Geiger Climate Classification 101
# ---------------------------------------------------------
with st.expander("🌡️ What is the Köppen-Geiger Climate Classification?", expanded=False):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        The **Köppen-Geiger climate classification** is one of the most widely used systems for
        categorizing the world's climates. Developed by climatologist Wladimir Köppen in 1884
        and later refined by Rudolf Geiger, it divides climates into five main groups based on
        temperature and precipitation patterns.
        """)

        st.markdown("#### 🌍 Five Main Climate Groups")

        st.markdown("""
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
        """)

        st.markdown("""
        Each main group is further subdivided with additional letters indicating specific characteristics
        like precipitation patterns (**f** = fully humid, **s** = dry summer, **w** = dry winter) and
        temperature ranges (**a** = hot summer, **b** = warm summer, **c** = cool summer, etc.).
        """)

    with col2:
        st.success("""
        **🎯 Why it matters for E-TRACE**

        Climate zones directly influence tourism patterns, seasonal demand, and visitor preferences.

        Understanding how climate distributions change over time helps us:

        - 📈 Predict shifts in regional tourism attractiveness
        - 🏖️ Identify emerging seasonal patterns
        - 🌡️ Track climate change impacts on tourism
        - 🎿 Plan climate adaptation strategies
        - 💡 Forecast future visitor preferences
        """)


st.divider()

# ---------------------------------------------------------
# Dataset Loader Section
# ---------------------------------------------------------
st.header("📁💻 Loading E-Trace Processed Dataset...")

df = load_from_bq("SELECT * FROM `aklewagonproject.etrace.cleaned_final_jaume_dataset`")

st.session_state["df_clean"] = df

st.success("Dataset loaded successfully!")
st.write("### Preview of the data:")
st.dataframe(df.head())

st.write("### Dataset statistics:")
st.write(df.describe(include="all"))


# ---------------------------------------------------------
# Sidebar Navigation (for future pages)
# ---------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Exploration", "Mapping", "Models"])

if page == "Exploration":

    st.header("🔎 Region Explorer")
    st.markdown("Select a NUTS-2 region to explore its time-series indicators.")

    if df is None:
        st.warning("Please upload a dataset first in the Home page.")
    else:
        # ------------------------
        # Region Selector
        # ------------------------
        regions = sorted(df["geo"].dropna().unique())
        region = st.selectbox("Select a NUTS-2 region:", regions)

        df_region = df[df["geo"] == region].sort_values("year")

        st.subheader(f"📍 Region: **{region}**")
        st.write(df_region)

        st.divider()

        # ------------------------
        # Time-series plots
        # ------------------------

        numeric_columns = df_region.select_dtypes(include=["float64", "int64"]).columns

        available_vars = {
            "Tourism (Nights Spent)": "nights_spent",
            "GDP": "gdp",
            "Population": "pop",
            "Employment Rate": "employment_rate",
        }

        # Detect climate variables if present
        climate_vars = [col for col in numeric_columns if col.startswith("pct_")]
        for c in climate_vars:
            available_vars[f"Climate: {c}"] = c

        st.header("📈 Time-Series Indicators")

        # Plot each available variable
        for label, col in available_vars.items():
            if col in df_region.columns:
                # If it is a climate variable, convert pct_Dfb → Dfb → readable label
                if col.startswith("pct_"):
                    code = col.replace("pct_", "")  # e.g. Dfb
                    pretty_label = f"Climate: {koppen_labels.get(code, code)}"
                else:
                    pretty_label = label

                st.subheader(pretty_label)

                fig = px.line(
                    df_region,
                    x="year",
                    y=col,
                    markers=True,
                    title=f"{pretty_label} over time in {region}"
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)


        # ------------------------
        # Climate Stacked Area Chart
        # ------------------------

        st.subheader("🌍 Climate Composition Over Time")

        climate_cols = [c for c in df_region.columns if c.startswith("pct_")]

        if climate_cols:
            climate_df = df_region[["year"] + climate_cols].copy()

            # Melt into long format
            climate_long = climate_df.melt(
                id_vars="year",
                var_name="climate_zone",
                value_name="fraction"
            )

            # Apply human-readable names
            climate_long["climate_zone_label"] = climate_long["climate_zone"].apply(
                lambda x: koppen_labels[x.replace("pct_", "")]
                if x.replace("pct_", "") in koppen_labels
                else x
            )

            fig_climate = px.area(
                climate_long,
                x="year",
                y="fraction",
                color="climate_zone_label",
                title=f"Climate Distribution Over Time in {region}"
            )

            st.plotly_chart(fig_climate, use_container_width=True)

elif page == "Mapping":

    st.header("🗺️ NUTS-2 Regional Map Visualization")

    # Load your merged dataset
    df_clean = st.session_state.get("df_clean")

    if df_clean is None:
        st.warning("Something went wrong uploading the data.")
        st.stop()

    # Load NUTS2 GeoJSON from google cloud
    client = storage.Client()
    bucket = client.bucket("etrace-data")
    blob = bucket.blob("data/raw_data/nuts2_geo.geojson")

    geojson_bytes = blob.download_as_bytes()
    nuts2_geo = geojson.loads(geojson_bytes.decode("utf-8"))

    # Ensure 'geo' column exists
    if "geo" not in df_clean.columns:
        st.error("The dataset does not contain a 'geo' column.")
        st.stop()

    # Variables available for mapping
    map_numeric_cols = [
        c for c in df_clean.columns
        if df_clean[c].dtype in ["float64", "int64"]
    ]

    selected_var = st.selectbox("Variable to visualize:", map_numeric_cols)
    st.write(f"### Visualizing: **{selected_var}**")

    years = sorted(df_clean["year"].unique())
    selected_year = st.slider("Select Year", min(years), max(years), min(years))

    df_year = df_clean[df_clean["year"] == selected_year]

    all_geo2= []
    for each in nuts2_geo["features"]:
        if each.properties["LEVL_CODE"] == 2:
            all_geo2.append(each)
    nuts2_geo["features"] = all_geo2

    # Attach values to GeoJSON
    for feature in nuts2_geo["features"]:
        geo_id = feature["properties"]["NUTS_ID"]
        match = df_year[df_year["geo"] == geo_id]

        if not match.empty:
            feature["properties"][selected_var] = float(match[selected_var].values[0])
        else:
            feature["properties"][selected_var] = None

    # -------------------------------------------
    # Compute normalized color values
    # -------------------------------------------

    vmin = df_year[selected_var].min()
    vmax = df_year[selected_var].max()

    df_year["scaled_value"] = (df_year[selected_var] - vmin) / (vmax - vmin)

    def colormap(v):
        if v is None or pd.isna(v):
            return [200, 200, 200]

        import math

        # Turbo colormap implementation
        turbo = [
            [48, 18, 59], [53, 41, 133], [37, 66, 167], [20, 92, 157], [16, 120, 130],
            [32, 144, 92], [68, 164, 54], [112, 181, 25], [160, 194, 9], [210, 203, 8],
            [255, 209, 28], [255, 189, 51], [255, 158, 73], [255, 116, 95], [255, 64, 112],
            [237, 5, 121], [203, 0, 122], [155, 0, 112], [102, 0, 92], [56, 0, 63]
        ]

        idx = min(int(v * (len(turbo)-1)), len(turbo)-1)
        return turbo[idx]




    # -------------------------------------------
    # SAFE Normalize values
    # -------------------------------------------

    vals = df_year[selected_var].astype(float)

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
        match = df_year[df_year["geo"] == geo_id]

        if not match.empty:
            feature["properties"]["color"] = match["color"].values[0]
        else:
            feature["properties"]["color"] = [180, 180, 180]   # grey fallback


    def highlight_selected_column(df, column_name):
        """
        Resalta la columna seleccionada con un color especial
        """
        styles = pd.DataFrame('', index=df.index, columns=df.columns)
        if column_name in df.columns:
            styles[column_name] = 'background-color: #FFD700; color: black; font-weight: bold'
        return styles

    #highlight col
    st.write("### Data Preview")
    st.dataframe(
        df_year.style.apply(highlight_selected_column, column_name=selected_var, axis=None),
        use_container_width=True,
        height=400
    )
    st.write(df_year.shape)

    st.markdown("### 📊 Color Legend")

    # statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mínimo", f"{vmin:.2f}")
    with col2:
        st.metric("Media", f"{df_year[selected_var].mean():.2f}")
    with col3:
        st.metric("Máximo", f"{vmax:.2f}")

    # visual grad ( nabla)
    st.markdown(f"""
    <div style="background: linear-gradient(to right,
        rgb(48,18,59), rgb(37,66,167), rgb(16,120,130),
        rgb(68,164,54), rgb(160,194,9), rgb(255,209,28),
        rgb(255,158,73), rgb(255,64,112), rgb(203,0,122));
        height: 30px; border-radius: 5px; margin: 10px 0;">
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{vmin:.2f}</span>
        <span style="font-weight: bold;">{selected_var}</span>
        <span>{vmax:.2f}</span>
    </div>
    """, unsafe_allow_html=True)


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

    # PyDeck layer
    data_layer = pdk.Layer(
        "DataLayer",
        data=df_year,
    )

    view_state = pdk.ViewState(
        latitude=50,
        longitude=10,
        zoom=3.3,
        bearing=0,
        pitch=35,
    )

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

elif page == "Models":
    st.header("🤖 Predictive Models")
    st.write("Coming soon: model training, forecasting, climate-tourism interactions…")


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.write(
    "E-TRACE • European Tourism Regional Analysis & Climate Effects • Built with ❤️ using Streamlit"
)
