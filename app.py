import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

# ------------------------------------------------
# Streamlit page config
# ------------------------------------------------
st.set_page_config(
    page_title="Steel Pile Corrosion Rate Predictor",
    layout="wide"
)

# McMaster-ish theme colors
MAC_MAROON = "#7A003C"
MAC_GOLD = "#FDBF57"
MU_COLOR = "#00427A"   # dark blue for μ text and line

plt.rcParams.update({
    "figure.autolayout": False
})

# ------------------------------------------------
# Load model + preprocessor
# ------------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ------------------------------------------------
# Header + (optional) logo
# ------------------------------------------------
header_col, logo_col = st.columns([4, 1])

with header_col:
    st.markdown(
        "<h1 style='text-align:center;'>Prediction of Corrosion Rate for Steel Piles Embedded in Soil</h1>",
        unsafe_allow_html=True
    )

with logo_col:
    # If you upload a file called "mcmaster_logo.png" to the repo, it will appear here.
    if os.path.exists("mcmaster_logo.png"):
        st.image("mcmaster_logo.png", use_container_width=True)

st.markdown(
    f"<h4 style='text-align:center; color:{MAC_MAROON};'>NGBoost-based probabilistic model (Normal distribution)</h4>",
    unsafe_allow_html=True
)

# ------------------------------------------------
# Accepted ranges (nice, rounded)
# ------------------------------------------------
ranges = {
    "age": (1, 65),
    "pH": (3.0, 10.0),
    "chloride": (10.0, 12000.0),
    "resistivity": (80.0, 13000.0),
    "sulphate": (5.0, 22000.0),
    "moisture": (5.0, 250.0)
}

# Categories (from your summary)
soil_types = [
    "Granite", "ML+SM", "ML", "CL+SC", "CL+ML", "SM", "CL",
    "SP+SM", "SP", "CH", "SW", "OL", "GP", "GP+GM", "SC"
]
foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]
location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

# ------------------------------------------------
# INPUT SECTION — Two Columns (5 inputs each)
# ------------------------------------------------
left, right = st.columns(2)

with left:
    age = st.number_input(
        f"Age (yr) [{ranges['age'][0]}–{ranges['age'][1]}]",
        min_value=1,
        max_value=100,
        step=1,
        value=34
    )
    if not (ranges["age"][0] <= age <= ranges["age"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

    soil_pH = st.number_input(
        f"Soil pH [{ranges['pH'][0]}–{ranges['pH'][1]}]",
        min_value=0.0,
        max_value=14.0,
        step=0.1,
        value=7.8
    )
    if not (ranges["pH"][0] <= soil_pH <= ranges["pH"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

    chloride = st.number_input(
        f"Chloride Content (mg/kg) [{ranges['chloride'][0]}–{ranges['chloride'][1]}]",
        min_value=0.0,
        max_value=20000.0,
        step=10.0,
        value=500.0
    )
    if not (ranges["chloride"][0] <= chloride <= ranges["chloride"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

    resistivity = st.number_input(
        f"Soil Resistivity (Ω·cm) [{ranges['resistivity'][0]}–{ranges['resistivity'][1]}]",
        min_value=0.0,
        max_value=20000.0,
        step=10.0,
        value=800.0
    )
    if not (ranges["resistivity"][0] <= resistivity <= ranges["resistivity"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

    sulphate = st.number_input(
        f"Sulphate Content (mg/kg) [{ranges['sulphate'][0]}–{ranges['sulphate'][1]}]",
        min_value=0.0,
        max_value=30000.0,
        step=1.0,      # ✅ no forced multiples of 5
        value=300.0
    )
    if not (ranges["sulphate"][0] <= sulphate <= ranges["sulphate"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

with right:
    moisture = st.number_input(
        f"Moisture Content (%) [{ranges['moisture'][0]}–{ranges['moisture'][1]}]",
        min_value=0.0,
        max_value=300.0,
        step=0.1,
        value=25.0
    )
    if not (ranges["moisture"][0] <= moisture <= ranges["moisture"][1]):
        st.markdown(
            "<span style='color:red; font-size:13px;'>Warning: input is outside the training range of the ML model.</span>",
            unsafe_allow_html=True
        )

    soil_type = st.selectbox("Soil Type", soil_types)
    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)
    is_fill_str = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill_str == "Yes" else 0

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if st.button("Predict Corrosion Rate"):

    # Prepare input dataframe
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

    # Transform
    Xp = preprocessor.transform(raw)

    # Predict distribution
    dist = model.pred_dist(Xp)
    mu = float(dist.params["loc"])
    sigma = float(dist.params["scale"])

    # μ ± σ – nicely rounded
    st.markdown(
        f"""
        <h3 style='text-align:center;'>
        Most likely corrosion rate is 
        <span style='color:{MU_COLOR};'><b>{mu:.4f} ± {sigma:.4f}</b></span> mm/yr (μ ± σ)
        </h3>
        """,
        unsafe_allow_html=True
    )

    # ------------------------------------------------
    # TWO-COLUMN OUTPUT (PDF + CDF) – equal size, smaller
    # ------------------------------------------------
    pdf_col, cdf_col = st.columns(2)

    x_vals = np.linspace(max(0.0001, mu - 4*sigma), mu + 4*sigma, 600)
    pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    cdf_vals = norm.cdf(x_vals, loc=mu, scale=sigma)

    # -------- PDF PLOT --------
    with pdf_col:
        fig1, ax1 = plt.subplots(figsize=(3.2, 2.5), dpi=150)
        ax1.plot(x_vals, pdf_vals, linewidth=2, color=MAC_MAROON)
        ax1.axvline(mu, color=MU_COLOR, linestyle="--", linewidth=1.6, label=f"μ = {mu:.4f}")
        ax1.set_title("PDF — Probability Density", fontsize=11)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=9)
        ax1.set_ylabel("PDF", fontsize=9)
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)
        fig1.tight_layout(pad=0.3)
        st.pyplot(fig1, clear_figure=True)

    # -------- CDF PLOT --------
    with cdf_col:
        fig2, ax2 = plt.subplots(figsize=(3.2, 2.5), dpi=150)
        ax2.plot(x_vals, cdf_vals, linewidth=2, color=MAC_GOLD)
        ax2.axvline(mu, color=MU_COLOR, linestyle="--", linewidth=1.6, label=f"μ = {mu:.4f}")
        ax2.set_title("CDF — Probability corrosion rate ≤ X", fontsize=11)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=9)
        ax2.set_ylabel("CDF", fontsize=9)
        ax2.grid(alpha=0.25)
        ax2.legend(fontsize=8)
        fig2.tight_layout(pad=0.3)
        st.pyplot(fig2, clear_figure=True)

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown(
    f"""
    <br>
    <h2 style='text-align:center;'>
    Developed by <b>Rishav Jaiswal</b><br>
    <span style='color:{MAC_MAROON};'>McMaster University</span>
    </h2>
    """,
    unsafe_allow_html=True
)
