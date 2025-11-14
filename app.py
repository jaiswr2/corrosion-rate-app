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
st.markdown("<h1 style='text-align:center;'>Prediction of Corrosion Rate for Steel Piles Embedded in Soil</h1>", 
            unsafe_allow_html=True)

# ============================
# ACCEPTED RANGES
# ============================
ranges = {
    "age": (1, 65),
    "pH": (3, 10),
    "chloride": (10, 12000),
    "resistivity": (80, 13000),
    "sulphate": (5, 22000),
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
# INPUT SECTION — Two Columns (5 left + 5 right)
# ============================================================
left, right = st.columns(2)

with left:
    age = st.number_input("Age (yr) [1–65]", 1, 65, step=1)
    if not (ranges["age"][0] <= age <= ranges["age"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    soil_pH = st.number_input("Soil pH [3–10]", 3.0, 10.0, step=0.1)
    if not (ranges["pH"][0] <= soil_pH <= ranges["pH"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    chloride = st.number_input("Chloride Content (mg/kg) [10–12000]", 10.0, 12000.0, step=10.0)
    if not (ranges["chloride"][0] <= chloride <= ranges["chloride"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    moisture = st.number_input("Moisture Content (%) [2–260]", 2.0, 260.0, step=0.1)
    if not (ranges["moisture"][0] <= moisture <= ranges["moisture"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    soil_type = st.selectbox("Soil Type", soil_types)

with right:
    resistivity = st.number_input("Soil Resistivity (Ω·cm) [80–13000]", 80.0, 13000.0, step=10.0)
    if not (ranges["resistivity"][0] <= resistivity <= ranges["resistivity"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    sulphate = st.number_input("Sulphate Content (mg/kg) [5–22000]", 5.0, 22000.0, step=10.0)
    if not (ranges["sulphate"][0] <= sulphate <= ranges["sulphate"][1]):
        st.markdown("<span style='color:red;'>Out of range – may reduce reliability.</span>", unsafe_allow_html=True)

    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)

    is_fill = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill == "Yes" else 0

# ============================================================
# PREDICTION
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
        f"<h3>Most likely corrosion rate is <b>{mu:.4f} ± {sigma:.4f}</b> mm/yr (μ ± σ)</h3>",
        unsafe_allow_html=True
    )

    # ============================================================
    # OUTPUT PLOTS — PDF and CDF (smaller size)
    # ============================================================
    pdf_col, cdf_col = st.columns(2)

    x_vals = np.linspace(max(0.0001, mu - 4*sigma), mu + 4*sigma, 600)
    pdf_vals = norm.pdf(x_vals, mu, sigma)
    cdf_vals = norm.cdf(x_vals, mu, sigma)

    # -------- PDF --------
    with pdf_col:
        fig1, ax1 = plt.subplots(figsize=(4,3))
        ax1.plot(x_vals, pdf_vals, color="blue", linewidth=2)
        ax1.axvline(mu, color="red", linestyle="--", linewidth=2)
        ax1.set_title("PDF (Probability Density)", fontsize=12)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=10)
        ax1.set_ylabel("PDF", fontsize=10)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # -------- CDF --------
    with cdf_col:
        fig2, ax2 = plt.subplots(figsize=(4,3))
        ax2.plot(x_vals, cdf_vals, color="green", linewidth=2)
        ax2.axvline(mu, color="red", linestyle="--", linewidth=2)
        ax2.set_title("CDF (Probability corrosion ≤ X)", fontsize=12)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=10)
        ax2.set_ylabel("CDF", fontsize=10)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    "<br><h2 style='text-align:center;'>Developed by <b>Rishav Jaiswal</b><br>McMaster University</h2>",
    unsafe_allow_html=True
)
