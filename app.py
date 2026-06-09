import streamlit as st
import joblib
import numpy as np

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("random_forest.pkl")

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0E1117;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
    margin-bottom: 10px;
}

/* Subtitle */
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #9CA3AF;
    margin-bottom: 35px;
}

/* Labels */
label {
    color: white !important;
    font-size: 16px !important;
    font-weight: 600 !important;
}

/* Input Fields */
[data-testid="stNumberInput"] input {
    background-color: #1F2937 !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #374151 !important;
    height: 45px !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00C853, #00E676);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    border: none;
    margin-top: 30px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(90deg, #00E676, #69F0AE);
}

/* Metric Card */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 2px solid #00E676;
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,255,120,0.25);
}

[data-testid="metric-container"] label {
    color: white !important;
    font-size: 20px !important;
}

[data-testid="metric-container"] div {
    color: #00E676 !important;
    font-size: 38px !important;
    font-weight: bold !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏠 AI House Price Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Machine Learning Powered Real Estate Valuation System</div>',
    unsafe_allow_html=True
)

# ==========================================
# INPUT SECTION
# ==========================================

col1, col2 = st.columns(2)

with col1:

    MedInc = st.number_input(
        "Median Income (USD)",
        min_value=0,
        step=1000,
        placeholder="Enter annual income"
    )

    HouseAge = st.number_input(
        "House Age (Years)",
        min_value=0,
        step=1,
        placeholder="Enter house age"
    )

    AveRooms = st.number_input(
        "Average Rooms",
        min_value=0,
        step=1,
        placeholder="Enter average rooms"
    )

    AveBedrms = st.number_input(
        "Average Bedrooms",
        min_value=0,
        step=1,
        placeholder="Enter average bedrooms"
    )

with col2:

    Population = st.number_input(
        "Population",
        min_value=0,
        step=1,
        placeholder="Enter population"
    )

    AveOccup = st.number_input(
        "Average Occupancy",
        min_value=0,
        step=1,
        placeholder="Enter occupancy"
    )

    Latitude = st.number_input(
        "Latitude",
        value=37.0,
        format="%.4f"
    )

    Longitude = st.number_input(
        "Longitude",
        value=-122.0,
        format="%.4f"
    )

# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict House Price"):

    # Convert income scale
    scaled_income = MedInc / 10000

    # Feature Array
    features = np.array([[
        scaled_income,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    # Prediction
    prediction = model.predict(features)

    # Convert to USD
    predicted_price_usd = prediction[0] * 100000

    st.success("Prediction Generated Successfully!")

    st.write("")

    # Price Output
    st.metric(
        label="🏠 Estimated House Price",
        value=f"${predicted_price_usd:,.2f}"
    )

    st.write("")

    # Price Category
    if predicted_price_usd < 200000:
        st.info("Affordable Housing Category")

    elif predicted_price_usd < 500000:
        st.warning("Mid-Range Housing Category")

    else:
        st.error("Luxury Housing Category")
