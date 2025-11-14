import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

st.set_page_config(page_title="Steel Pile Corrosion Predictor", layout="wide")

# ============================
# Load model + preprocessor
# ============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ============================
# Header + McMaster Logo
# ============================
title_col, logo_col = st.columns([5,1])

with title_col:
    st.markdown(
        "<h1 style='text-align:center; color:#7A003C;'>"
        "Probabilistic ML Model for Predicting Corrosion of Steel Piles Embedded in Soil"
        "</h1>",
        unsafe_allow_html=True
    )

with logo_col:
    if os.path.exists("mcmaster_logo.png"):
        st.image("mcmaster_logo.png", width=70)

# ============================
# Accepted Ranges
# ============================
ranges = {
    "age": (1, 70),
    "pH": (3.0, 10.0),
    "chloride": (10, 12000),
    "resistivity": (80, 13000),
    "sulphate": (5, 22000),
    "moisture": (5, 250)
}

soil_types = [
    "Granite","ML+SM","ML","CL+SC","CL+ML","SM","CL","SP+SM","SP",
    "CH","SW","OL","GP","GP+GM","SC"
]
foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]
location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

# ============================================================
# INPUT SECTION — Two columns (clean)
# ============================================================
left, right = st.columns(2)

def warn_if_out_of_range(val, lo, hi):
    """Return red warning text (inline) without blocking prediction."""
    if not (lo <= val <= hi):
        return f" <span style='color:red;'>⚠ Out of range</span>"
    return ""

# -----------------------------------------
# LEFT SIDE
# -----------------------------------------
with left:
    age = st.number_input(f"Age (yr) [{ranges['age'][0]}–{ranges['age'][1]}]", 
                          min_value=1, max_value=70, step=1)
    st.markdown(warn_if_out_of_range(age, *ranges["age"]), unsafe_allow_html=True)

    soil_pH = st.number_input(f"Soil pH [{ranges['pH'][0]}–{ranges['pH'][1]}]",
                              min_value=3.0, max_value=10.0, step=0.1)
    st.markdown(warn_if_out_of_range(soil_pH, *ranges["pH"]), unsafe_allow_html=True)

    chloride = st.number_input(
        f"Chloride Content (mg/kg) [{ranges['chloride'][0]}–{ranges['chloride'][1]}]",
        min_value=1.0, max_value=20000.0, step=1.0
    )
    st.markdown(warn_if_out_of_range(chloride, *ranges["chloride"]), unsafe_allow_html=True)

    moisture = st.number_input(
        f"Moisture Content (%) [{ranges['moisture'][0]}–{ranges['moisture'][1]}]",
        min_value=1.0, max_value=300.0, step=0.1
    )
    st.markdown(warn_if_out_of_range(moisture, *ranges["moisture"]), unsafe_allow_html=True)

    soil_type = st.selectbox("Soil Type", soil_types)

# -----------------------------------------
# RIGHT SIDE
# -----------------------------------------
with right:
    resistivity = st.number_input(
        f"Soil Resistivity (Ω·cm) [{ranges['resistivity'][0]}–{ranges['resistivity'][1]}]",
        min_value=10.0, max_value=20000.0, step=1.0
    )
    st.markdown(warn_if_out_of_range(resistivity, *ranges["resistivity"]), unsafe_allow_html=True)

    sulphate = st.number_input(
        f"Sulphate Content (mg/kg) [{ranges['sulphate'][0]}–{ranges['sulphate'][1]}]",
        min_value=1.0, max_value=30000.0
    )
    st.markdown(warn_if_out_of_range(sulphate, *ranges["sulphate"]), unsafe_allow_html=True)

    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)
    is_fill = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill == "Yes" else 0

# ============================================================
# Prediction
# ============================================================
if st.button("Predict Corrosion Rate"):

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

    Xp = preprocessor.transform(raw)
    dist = model.pred_dist(Xp)
    mu = float(dist.params["loc"])
    sigma = float(dist.params["scale"])

    st.markdown(
        f"<h3 style='color:#7A003C;'>Most likely corrosion rate: <b>{mu:.4f} ± {sigma:.4f} mm/yr</b></h3>",
        unsafe_allow_html=True
    )

    # ============================
    # PDF + CDF — same size as earlier
    # ============================
    c1, c2 = st.columns(2)

    x_vals = np.linspace(max(0.0001, mu - 4*sigma), mu + 4*sigma, 600)
    pdf_vals = norm.pdf(x_vals, mu, sigma)
    cdf_vals = norm.cdf(x_vals, mu, sigma)

    # PDF
    with c1:
        fig1, ax1 = plt.subplots(figsize=(6, 4.5))
        ax1.plot(x_vals, pdf_vals, linewidth=2, color="#7A003C")
        ax1.axvline(mu, color="#DAA520", linestyle="--", linewidth=2)
        ax1.set_title("PDF — Probability Density", fontsize=14)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=12)
        ax1.set_ylabel("PDF", fontsize=12)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # CDF
    with c2:
        fig2, ax2 = plt.subplots(figsize=(6, 4.5))
        ax2.plot(x_vals, cdf_vals, linewidth=2, color="#7A003C")
        ax2.axvline(mu, color="#DAA520", linestyle="--", linewidth=2)
        ax2.set_title("CDF — Probability Corrosion ≤ X", fontsize=14)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=12)
        ax2.set_ylabel("CDF", fontsize=12)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ============================================================
# Footer
# ============================================================
st.markdown(
    "<br><h2 style='text-align:center; color:#7A003C;'>Developed by <b>Rishav Jaiswal</b><br>McMaster University</h2>",
    unsafe_allow_html=True
)
