import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Steel Pile Corrosion Rate Predictor", layout="wide")

# =========================
# McMASTER THEME (CSS)
# =========================
st.markdown("""
    <style>
        h1, h2, h3 {
            color: #7A003C !important;  /* McMaster maroon */
        }

        .stButton>button {
            background-color:#7A003C;
            color:white;
            border-radius:8px;
            height:3em;
            width:14em;
            font-size:16px;
            font-weight:bold;
        }
        .stButton>button:hover {
            background-color:#5C002C;
            color:white;
        }

        .warning-text {
            color: red;
            font-weight: bold;
            font-size: 12px;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL + PREPROCESSOR
# =========================
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# =========================
# LOGO + TITLE
# =========================
logo_col, title_col = st.columns([1, 4])

with logo_col:
    # Make sure this file exists in repo root
    st.image("mcmaster_logo.png", width=110)

with title_col:
    st.markdown(
        "<h1 style='text-align:center;'>Prediction of Corrosion Rate for Steel Piles Embedded in Soil</h1>",
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# ACCEPTED RANGES (NICELY ROUNDED)
# =========================
ranges = {
    "age": (1, 65),
    "pH": (3.0, 10.0),
    "chloride": (10, 12000),
    "resistivity": (80, 13000),
    "sulphate": (5, 22000),
    "moisture": (5, 250)
}

# Categories (from your dataset)
soil_types = [
    "Granite", "ML+SM", "ML", "CL+SC", "CL+ML", "SM", "CL",
    "SP+SM", "SP", "CH", "SW", "OL", "GP", "GP+GM", "SC"
]
foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]
location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

maroon = "#7A003C"
gold = "#FFCC33"

# ============================================================
# INPUT SECTION — 2 COLUMNS, 5 INPUTS EACH
# ============================================================
left, right = st.columns(2)

with left:
    age = st.number_input(
        f"Age (yr) [{ranges['age'][0]}–{ranges['age'][1]}]",
        min_value=1, max_value=65, step=1
    )
    if not (ranges["age"][0] <= age <= ranges["age"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

    soil_pH = st.number_input(
        f"Soil pH [{ranges['pH'][0]}–{ranges['pH'][1]}]",
        min_value=3.0, max_value=10.0, step=0.1
    )
    if not (ranges["pH"][0] <= soil_pH <= ranges["pH"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

    chloride = st.number_input(
        f"Chloride Content (mg/kg) [{ranges['chloride'][0]}–{ranges['chloride'][1]}]",
        min_value=10.0, max_value=12000.0, step=10.0
    )
    if not (ranges["chloride"][0] <= chloride <= ranges["chloride"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

    resistivity = st.number_input(
        f"Soil Resistivity (Ω·cm) [{ranges['resistivity'][0]}–{ranges['resistivity'][1]}]",
        min_value=80.0, max_value=13000.0, step=10.0
    )
    if not (ranges["resistivity"][0] <= resistivity <= ranges["resistivity"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

    moisture = st.number_input(
        f"Moisture Content (%) [{ranges['moisture'][0]}–{ranges['moisture'][1]}]",
        min_value=5.0, max_value=250.0, step=0.1
    )
    if not (ranges["moisture"][0] <= moisture <= ranges["moisture"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

with right:
    sulphate = st.number_input(
        f"Sulphate Content (mg/kg) [{ranges['sulphate'][0]}–{ranges['sulphate'][1]}]",
        min_value=5.0, max_value=22000.0, step=1.0  # step=1, not forcing multiples of 5
    )
    if not (ranges["sulphate"][0] <= sulphate <= ranges["sulphate"][1]):
        st.markdown("<span class='warning-text'>Outside training range – prediction may be unreliable.</span>",
                    unsafe_allow_html=True)

    soil_type = st.selectbox("Soil Type", soil_types)
    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)

    is_fill_str = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill_str == "Yes" else 0

# ============================================================
# PREDICTION BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)
if st.button("Predict Corrosion Rate"):

    # Prepare input dataframe (must match training feature names)
    raw = pd.DataFrame([{
        "Age (yr)": age,
        "Soil_pH": soil_pH,
        "Chloride Content (mg/kg)": chloride,
        "Soil_Resistivity (Ω·cm)": resistivity,
        "Sulphate_Content (mg/kg)": sulphate,
        "Moisture_Content (%)": moisture,
        "Soil Type": soil_type,
        "Foreign_Inclusion_Type": foreign,
        "Location wrt Water Table": location,
        "Is_Fill_Material": is_fill
    }])

    # Transform through preprocessor
    Xp = preprocessor.transform(raw)

    # Predict NGBoost distribution
    dist = model.pred_dist(Xp)
    mu = float(dist.params["loc"])
    sigma = float(dist.params["scale"])

    st.markdown(
        f"<h3 style='text-align:center;'>Most likely corrosion rate is "
        f"<b>{mu:.4f} ± {sigma:.4f}</b> mm/yr (μ ± σ)</h3>",
        unsafe_allow_html=True
    )

    # ============================
    # TWO-COLUMN OUTPUT: PDF + CDF
    # ============================
    pdf_col, cdf_col = st.columns(2)

    # X range for plots
    x_min = max(0.0001, mu - 4 * sigma)
    x_max = mu + 4 * sigma
    x_vals = np.linspace(x_min, x_max, 500)
    pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    cdf_vals = norm.cdf(x_vals, loc=mu, scale=sigma)

    # -------- PDF PLOT (smaller size) --------
    with pdf_col:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(x_vals, pdf_vals, linewidth=2, color=maroon)
        ax1.axvline(mu, color="black", linestyle="--", linewidth=1.5, label=f"μ = {mu:.4f}")
        ax1.set_title("PDF — Probability Density", fontsize=13)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=11)
        ax1.set_ylabel("PDF", fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=9)
        st.pyplot(fig1)

    # -------- CDF PLOT (smaller size) --------
    with cdf_col:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(x_vals, cdf_vals, linewidth=2, color=gold)
        ax2.axvline(mu, color="black", linestyle="--", linewidth=1.5, label=f"μ = {mu:.4f}")
        ax2.set_title("CDF — Probability that corrosion rate ≤ X", fontsize=13)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=11)
        ax2.set_ylabel("CDF", fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        st.pyplot(fig2)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<br><h2 style='text-align:center;'>Developed by <b>Rishav Jaiswal</b><br>McMaster University</h2>",
    unsafe_allow_html=True
)
