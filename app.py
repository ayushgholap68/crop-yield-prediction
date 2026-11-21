
import streamlit as st
import pandas as pd
import joblib
import json
import math
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Crop Yield Prediction System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# LOAD MODEL AND METADATA
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(
        BASE_DIR / "crop_yield_model.pkl"
    )

@st.cache_data
def load_metadata():
    with open(
        BASE_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)

try:
    model = load_model()
    metadata = load_metadata()

except Exception as error:
    st.error(f"Failed to load project files: {error}")
    st.stop()

# ==========================================
# CUSTOM THEME
# ==========================================

st.markdown("""
<style>
:root {
    color-scheme: light;
}

.stApp {
    background-color: #f4f8f3;
    color: #172b20;
}

[data-testid="stHeader"] {
    background-color: #f4f8f3;
}

[data-testid="stMain"] {
    color: #172b20;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3 {
    color: #14532d !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: #243b2d;
}

[data-testid="stSidebar"] {
    background-color: #123c27;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #d8e9da;
}

[data-testid="stMetricValue"] {
    color: #166534 !important;
}

[data-testid="stMetricLabel"] * {
    color: #475569 !important;
}

.hero {
    background: linear-gradient(
        115deg,
        #104027,
        #23834b
    );
    padding: 38px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: 36px;
}

.hero p {
    color: #e6f7e9 !important;
    font-size: 17px;
}

.card {
    background-color: white;
    border: 1px solid #dce9dc;
    padding: 25px;
    border-radius: 14px;
    margin: 15px 0;
}

.card h3 {
    color: #166534 !important;
}

.card p {
    color: #334155 !important;
}

.result-card {
    background: #dcfce7;
    border: 2px solid #86efac;
    border-radius: 18px;
    padding: 35px;
    text-align: center;
    margin-top: 25px;
}

.result-card h2 {
    color: #14532d !important;
    font-size: 42px;
    margin: 10px 0;
}

.result-card p {
    color: #166534 !important;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #15803d;
    color: white !important;
    border-radius: 10px;
    border: none;
    min-height: 48px;
    font-weight: 650;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #166534;
    color: white !important;
}

[data-testid="stNumberInput"] input {
    color: #172b20 !important;
    background-color: white;
}

.footer {
    text-align: center;
    color: #64748b;
    margin-top: 40px;
    padding: 20px;
    border-top: 1px solid #d8e9da;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def show_hero(title, description):
    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_card(title, description):
    st.markdown(
        f"""
        <div class="card">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🌱 Crop Yield System")

    st.caption(
        "Computational Intelligence Project"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🌾 Predict Yield",
            "📊 Model Performance",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.write("**Algorithm:** Random Forest")
    st.write("**Frontend:** Streamlit")
    st.write("**Language:** Python")

# ==========================================
# HOME PAGE
# ==========================================

if page == "🏠 Home":

    show_hero(
        "🌾 Crop Yield Prediction System",
        "Machine Learning for Smart Agriculture"
    )

    st.write(
        "Predict agricultural crop yield using "
        "crop type, state, season, rainfall, "
        "cultivated area and agricultural inputs."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset Records",
            f"{metadata.get('rows', 0):,}"
        )

    with col2:
        st.metric(
            "Available Crops",
            len(metadata["crops"])
        )

    with col3:
        st.metric(
            "States / UTs",
            len(metadata["states"])
        )

    st.subheader("Project Overview")

    c1, c2 = st.columns(2)

    with c1:
        show_card(
            "🎯 Objective",
            "Estimate crop yield using machine "
            "learning to support agricultural planning."
        )

    with c2:
        show_card(
            "🧠 Machine Learning",
            "Random Forest Regression trained "
            "using agricultural records."
        )

    show_card(
        "🌿 Prediction Workflow",
        "Select crop and location, enter "
        "agricultural parameters, and generate "
        "a crop yield prediction."
    )

    st.info(
        "Open the Predict Yield page from "
        "the sidebar to begin."
    )

# ==========================================
# CROP YIELD PREDICTION
# ==========================================

elif page == "🌾 Predict Yield":

    show_hero(
        "🌱 Predict Crop Yield",
        "Enter agricultural parameters to estimate yield."
    )

    st.subheader("Agricultural Input Features")

    st.caption(
        "Enter total fertilizer and pesticide "
        "quantities for the selected cultivated area, "
        "not quantities per hectare."
    )

    with st.form("crop_prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            crop = st.selectbox(
                "🌾 Crop Name",
                sorted(metadata["crops"])
            )

            state = st.selectbox(
                "📍 State / Union Territory",
                sorted(metadata["states"])
            )

            season = st.selectbox(
                "☀️ Crop Season",
                sorted(metadata["seasons"])
            )

            year = st.number_input(
                "📅 Cultivation Year",
                min_value=int(metadata["year_min"]),
                max_value=int(metadata["year_max"]),
                value=int(metadata["year_max"]),
                step=1
            )

        with col2:

            area = st.number_input(
                "🌱 Cultivated Area (hectares)",
                min_value=0.01,
                value=100.0,
                step=1.0,
                help=(
                    "1 hectare = 10,000 square metres. "
                    "Enter the total cultivated area."
                )
            )

            rainfall = st.number_input(
                "🌧️ Annual Rainfall (mm)",
                min_value=0.0,
                value=1000.0,
                step=10.0,
                help=(
                    "Total annual rainfall "
                    "in millimetres."
                )
            )

            fertilizer = st.number_input(
                "🧪 Total Fertilizer Used (kg)",
                min_value=0.0,
                value=15000.0,
                step=100.0,
                help=(
                    "Total fertilizer quantity "
                    "for the entire selected area."
                )
            )

            pesticide = st.number_input(
                "🛡️ Total Pesticide Used (kg)",
                min_value=0.0,
                value=30.0,
                step=1.0,
                help=(
                    "Total pesticide quantity "
                    "for the entire selected area."
                )
            )

        submitted = st.form_submit_button(
            "🌾 Predict Crop Yield",
            use_container_width=True
        )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    if submitted:

        input_data = pd.DataFrame([{
            "Crop": crop,
            "Year": int(year),
            "Season": season,
            "State": state,
            "Area": float(area),
            "Annual_Rainfall": float(rainfall),
            "Fertilizer": float(fertilizer),
            "Pesticide": float(pesticide)
        }])

        try:

            prediction = float(
                model.predict(input_data)[0]
            )

            if not math.isfinite(prediction):
                st.error(
                    "The model returned an invalid value."
                )
                st.stop()

            prediction = max(0.0, prediction)

            # Assumes Yield is tonnes/hectare.
            yield_kg = prediction * 1000

            # Estimated total crop production.
            total_production = prediction * area

            st.markdown(
                f"""
                <div class="result-card">
                    <p>🌾 Estimated Crop Yield</p>
                    <h2>{prediction:,.3f} t/ha</h2>
                    <p>Tonnes per Hectare</p>
                    <p>
                        Equivalent to
                        {yield_kg:,.2f} kg/hectare
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Prediction Results")

            r1, r2, r3 = st.columns(3)

            with r1:
                st.metric(
                    "Yield (t/ha)",
                    f"{prediction:,.3f}"
                )

            with r2:
                st.metric(
                    "Yield (kg/ha)",
                    f"{yield_kg:,.2f}"
                )

            with r3:
                st.metric(
                    "Estimated Production (tonnes)",
                    f"{total_production:,.2f}"
                )

            st.subheader("Selected Agricultural Parameters")

            summary = pd.DataFrame({
                "Feature": [
                    "Crop",
                    "State",
                    "Season",
                    "Cultivation Year",
                    "Cultivated Area",
                    "Annual Rainfall",
                    "Total Fertilizer",
                    "Total Pesticide"
                ],
                "Value": [
                    crop,
                    state,
                    season,
                    year,
                    f"{area:,.2f} hectares",
                    f"{rainfall:,.2f} mm",
                    f"{fertilizer:,.2f} kg",
                    f"{pesticide:,.2f} kg"
                ]
            })

            st.dataframe(
                summary,
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"Prediction completed for {crop} "
                f"in {state}."
            )

            st.warning(
                "Yield and agricultural input units "
                "are based on assumed dataset conventions. "
                "Verify the dataset documentation before "
                "using these estimates for real decisions."
            )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

# ==========================================
# MODEL PERFORMANCE
# ==========================================

elif page == "📊 Model Performance":

    show_hero(
        "📊 Model Performance",
        "Random Forest Regression Evaluation"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mean Absolute Error",
            f"{metadata['mae']:.4f}"
        )

    with c2:
        st.metric(
            "Root Mean Squared Error",
            f"{metadata['rmse']:.4f}"
        )

    with c3:
        st.metric(
            "R² Score",
            f"{metadata['r2']:.4f}"
        )

    show_card(
        "Mean Absolute Error (MAE)",
        "Average absolute difference between "
        "actual and predicted crop yield. "
        "Lower values are better."
    )

    show_card(
        "Root Mean Squared Error (RMSE)",
        "Measures prediction error while "
        "penalizing larger errors more strongly."
    )

    show_card(
        "Coefficient of Determination (R²)",
        "Measures how well the model explains "
        "variation in crop yield on test data."
    )

    st.subheader("Model Details")

    model_details = pd.DataFrame({
        "Parameter": [
            "Algorithm",
            "Learning Type",
            "Dataset",
            "Training Split",
            "Testing Split",
            "Target Variable"
        ],
        "Value": [
            "Random Forest Regressor",
            "Supervised Learning",
            "India Crop Yield 2000-2026",
            "80%",
            "20%",
            "Yield"
        ]
    })

    st.dataframe(
        model_details,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# ABOUT PROJECT
# ==========================================

elif page == "ℹ️ About Project":

    show_hero(
        "🌍 About the Project",
        "Crop Yield Prediction Using Machine Learning"
    )

    st.subheader("Project Abstract")

    st.write(
        "The Crop Yield Prediction System uses "
        "supervised machine learning to estimate "
        "agricultural yield based on historical "
        "crop records and environmental factors. "
        "Random Forest Regression is used for "
        "prediction and Streamlit provides an "
        "interactive web application."
    )

    st.subheader("Methodology")

    st.write("""
    1. Dataset Collection
    2. Data Cleaning
    3. Categorical Encoding
    4. Train-Test Split
    5. Random Forest Training
    6. Model Evaluation
    7. Model Saving
    8. Streamlit Deployment
    """)

    st.subheader("Input Features and Units")

    features = pd.DataFrame({
        "Feature": [
            "Crop",
            "State",
            "Season",
            "Year",
            "Area",
            "Annual Rainfall",
            "Fertilizer",
            "Pesticide"
        ],
        "Unit": [
            "Category",
            "Category",
            "Category",
            "Calendar Year",
            "Hectares (ha)",
            "Millimetres (mm)",
            "Kilograms (kg)",
            "Kilograms (kg)"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Technologies")

    st.write(
        "Python, Pandas, NumPy, Scikit-learn, "
        "Matplotlib, Seaborn, Joblib and Streamlit."
    )

    st.subheader("SDG Alignment")

    st.write("""
    - SDG 2: Zero Hunger
    - SDG 12: Responsible Consumption and Production
    - SDG 13: Climate Action
    """)

    st.subheader("Team Members")

    team = pd.DataFrame({
        "Name": [
            "Yadnyesh Kumbhar",
            "Ruchita More",
            "Samarth Wakchaure",
            "Ayush Gholap"
        ],
        "PRN": [
            "202401070170",
            "202401070166",
            "202401070173",
            "202401070176"
        ]
    })

    st.dataframe(
        team,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown(
    """
    <div class="footer">
        🌱 Crop Yield Prediction System |
        Computational Intelligence Project |
        Powered by Random Forest Regression
    </div>
    """,
    unsafe_allow_html=True
)
