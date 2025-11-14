import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Steel Pile Corrosion Rate Predictor", layout="wide")

# ============================
# Load model + preprocessor
# ============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ============================
# Header
# ============================
st.markdown("<h1 style='text-align:center;'>Prediction of Corrosion Rate for Steel Piles Embedded in Soil</h1>", unsafe_allow_html=True)

# ============================
# Accepted Ranges
# ============================
ranges = {
    "age": (1, 70),
    "pH": (3.4, 10.0),
    "chloride": (10, 12000),
    "resistivity": (80, 13000),
    "sulphate": (7, 22000),
    "moisture": (2, 260)
}

# Categories
soil_types = [
    "Granite","ML+SM","ML","CL+SC","CL+ML","SM","CL","SP+SM","SP","CH",
    "SW","OL","GP","GP+GM","SC"
]
foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]
location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

# ============================================================
# INPUT SECTION — Two Columns
# ============================================================
left, right = st.columns(2)

with left:
    age = st.number_input(f"Age (yr) [{ranges['age'][0]}–{ranges['age'][1]}]", min_value=1, max_value=70, step=1)
    if not (ranges["age"][0] <= age <= ranges["age"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

    soil_pH = st.number_input(f"Soil pH [{ranges['pH'][0]}–{ranges['pH'][1]}]", min_value=3.4, max_value=10.0, step=0.1)
    if not (ranges["pH"][0] <= soil_pH <= ranges["pH"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

    chloride = st.number_input(f"Chloride Content (mg/kg) [{ranges['chloride'][0]}–{ranges['chloride'][1]}]",
                               min_value=10.0, max_value=12000.0, step=10.0)
    if not (ranges["chloride"][0] <= chloride <= ranges["chloride"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

    moisture = st.number_input(f"Moisture Content (%) [{ranges['moisture'][0]}–{ranges['moisture'][1]}]",
                               min_value=2.0, max_value=260.0, step=0.1)
    if not (ranges["moisture"][0] <= moisture <= ranges["moisture"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

with right:
    resistivity = st.number_input(f"Soil Resistivity (Ω·cm) [{ranges['resistivity'][0]}–{ranges['resistivity'][1]}]",
                                  min_value=80.0, max_value=13000.0, step=10.0)
    if not (ranges["resistivity"][0] <= resistivity <= ranges["resistivity"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

    sulphate = st.number_input(f"Sulphate Content (mg/kg) [{ranges['sulphate'][0]}–{ranges['sulphate'][1]}]",
                               min_value=7.0, max_value=22000.0, step=10.0)
    if not (ranges["sulphate"][0] <= sulphate <= ranges["sulphate"][1]):
        st.markdown("<span style='color:red;'>Outside dataset range – prediction may be unreliable.</span>", unsafe_allow_html=True)

    soil_type = st.selectbox("Soil Type", soil_types)
    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)
    is_fill = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill == "Yes" else 0

# ============================================================
# Prediction
# ============================================================
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

    # Display μ ± σ
    st.markdown(f"<h3>Most likely corrosion rate is <b>{mu:.4f} ± {sigma:.4f}</b> mm/yr (μ ± σ)</h3>", 
                unsafe_allow_html=True)

    # ============================
    # TWO-COLUMN OUTPUT (PDF + CDF)
    # ============================
    pdf_col, cdf_col = st.columns(2)

    x_vals = np.linspace(max(0.0001, mu - 4*sigma), mu + 4*sigma, 600)
    pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    cdf_vals = norm.cdf(x_vals, loc=mu, scale=sigma)

    # -------- PDF PLOT --------
    with pdf_col:
        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(x_vals, pdf_vals, linewidth=2, color="blue")
        ax1.axvline(mu, color="red", linestyle="--", linewidth=2)
        ax1.set_title("PDF — Probability Density", fontsize=16)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=14)
        ax1.set_ylabel("PDF", fontsize=14)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # -------- CDF PLOT --------
    with cdf_col:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.plot(x_vals, cdf_vals, linewidth=2, color="green")
        ax2.axvline(mu, color="red", linestyle="--", linewidth=2)
        ax2.set_title("CDF — Probability Corrosion Rate ≤ X", fontsize=16)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=14)
        ax2.set_ylabel("CDF", fontsize=14)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ============================================================
# Footer
# ============================================================
st.markdown("<br><h2 style='text-align:center;'>Developed by <b>Rishav Jaiswal</b><br>McMaster University</h2>", unsafe_allow_html=True)
