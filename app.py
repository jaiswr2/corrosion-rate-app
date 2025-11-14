import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Steel Pile Corrosion Predictor", layout="wide")

# ===============================
# Load model and preprocessor
# ===============================
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# ===============================
# INPUT RANGES from your dataset
# ===============================
RANGES = {
    "Age (yr)": (1, 64),
    "Soil_pH": (3.4, 10),
    "Chloride Content (mg/kg)": (14, 11400),
    "Soil_Resistivity (Ω·cm)": (80, 13106),
    "Sulphate_Content (mg/kg)": (6.9, 21800),
    "Moisture_Content (%)": (1.7, 261.4),
}

categorical_options = {
    "Soil Type": ['CL', 'ML', 'CL-ML', 'SM', 'SC', 'SW', 'SP'],
    "Location wrt Water Table": ['Above', 'Below'],
    "Foreign_Inclusion_Type": ['None', 'Organic', 'Industrial', 'Construction'],
}

binary_options = {
    "Is_Fill_Material": [0, 1]
}

# ===============================
# Title
# ===============================
st.markdown("<h1 style='text-align:center;'>Steel Pile Corrosion Rate Predictor</h1>", unsafe_allow_html=True)
st.markdown("### Enter soil & environmental parameters below:")

# ===============================
# Two-column INPUT layout
# ===============================
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (yr)", min_value=0.0, step=1.0)
    ph = st.number_input("Soil_pH", min_value=0.0, step=0.1)
    chloride = st.number_input("Chloride Content (mg/kg)", min_value=0.0, step=1.0)
    soil_type = st.selectbox("Soil Type", categorical_options["Soil Type"])

with col2:
    resistivity = st.number_input("Soil_Resistivity (Ω·cm)", min_value=0.0, step=1.0)
    sulphate = st.number_input("Sulphate_Content (mg/kg)", min_value=0.0, step=1.0)
    moisture = st.number_input("Moisture_Content (%)", min_value=0.0, step=0.1)
    location = st.selectbox("Location wrt Water Table", categorical_options["Location wrt Water Table"])
    fill = st.selectbox("Is_Fill_Material", binary_options["Is_Fill_Material"])
    foreign = st.selectbox("Foreign_Inclusion_Type", categorical_options["Foreign_Inclusion_Type"])

# Collect in dict
input_dict = {
    "Age (yr)": age,
    "Soil_pH": ph,
    "Chloride Content (mg/kg)": chloride,
    "Soil_Resistivity (Ω·cm)": resistivity,
    "Sulphate_Content (mg/kg)": sulphate,
    "Moisture_Content (%)": moisture,
    "Soil Type": soil_type,
    "Location wrt Water Table": location,
    "Is_Fill_Material": fill,
    "Foreign_Inclusion_Type": foreign,
}

# ===============================
# Range Warning Messages
# ===============================
st.markdown("---")
st.markdown("### ⚠️ Input Validation")

for key, val in input_dict.items():
    if key in RANGES:
        min_v, max_v = RANGES[key]
        if val < min_v or val > max_v:
            st.warning(f"{key} = {val} is outside dataset range [{min_v}, {max_v}]")

# ===============================
# Prediction Button
# ===============================
st.markdown("---")
if st.button("Predict Corrosion Rate"):
    
    # Convert to dataframe
    import pandas as pd
    df_input = pd.DataFrame([input_dict])

    # Transform using your preprocessor
    X_trans = preprocessor.transform(df_input)

    # Predict distribution
    y_dist = model.pred_dist(X_trans)[0]
    mu = y_dist.params["loc"]
    sigma = y_dist.params["scale"]

    st.success(f"### Most likely corrosion rate: **{mu:.4f} ± {sigma:.4f} mm/yr**")

    # Threshold input for CDF
    st.markdown("### Enter threshold to compute probability:")
    threshold = st.number_input("Threshold corrosion rate (mm/yr)", min_value=0.0, step=0.001)

    if threshold > 0:
        cdf_val = norm.cdf(threshold, mu, sigma)
        exceed_val = 1 - cdf_val

        st.info(
            f"**Probability corrosion ≤ {threshold:.3f} mm/yr:** {cdf_val*100:.2f}%\n\n"
            f"**Probability corrosion > {threshold:.3f} mm/yr:** {exceed_val*100:.2f}%"
        )

    # ===============================
    # 2-Column OUTPUT: PDF & CDF Plots
    # ===============================
    out1, out2 = st.columns(2)

    # x-values for PDF & CDF
    x = np.linspace(max(0, mu - 4*sigma), mu + 6*sigma, 400)
    pdf_vals = norm.pdf(x, mu, sigma)
    cdf_vals = norm.cdf(x, mu, sigma)

    # ---------------- PDF plot ----------------
    with out1:
        fig1, ax1 = plt.subplots(figsize=(6, 5), dpi=140)
        ax1.plot(x, pdf_vals, linewidth=3)
        ax1.set_title("Probability Density Function (PDF)", fontsize=18)
        ax1.set_xlabel("Corrosion Rate (mm/yr)", fontsize=16)
        ax1.set_ylabel("PDF", fontsize=16)
        ax1.grid(True)
        st.pyplot(fig1)

    # ---------------- CDF plot ----------------
    with out2:
        fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=140)
        ax2.plot(x, cdf_vals, linewidth=3, color="green")
        ax2.set_title("Cumulative Distribution Function (CDF)", fontsize=18)
        ax2.set_xlabel("Corrosion Rate (mm/yr)", fontsize=16)
        ax2.set_ylabel("CDF", fontsize=16)
        ax2.grid(True)
        st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("<h4 style='text-align:center;'>Developed by Rishav Jaiswal (2025)</h4>", unsafe_allow_html=True)
