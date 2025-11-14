import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="Steel Pile Corrosion Rate Predictor", layout="centered")

# Load model + transformer
model = joblib.load("model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.title("🛠️ Steel Pile Corrosion Rate Predictor (NGBoost – Normal)")
st.write("Enter soil and environmental parameters to predict **mean corrosion rate (μ)**, **uncertainty (σ)**, and **CDF plot**.")

# ----------- Input Widgets -----------

soil_types = [
    "Granite","ML+SM","ML","CL+SC","CL+ML","SM","CL","SP+SM","SP","CH",
    "SW","OL","GP","GP+GM","SC"
]

foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]

location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

age = st.number_input("Age (yr)", min_value=0.0, max_value=200.0, step=0.1)
soil_pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1)
chloride = st.number_input("Chloride Content (mg/kg)", min_value=0.0, max_value=50000.0, step=10.0)
resistivity = st.number_input("Soil Resistivity (Ω·cm)", min_value=0.0, max_value=100000.0, step=10.0)
sulphate = st.number_input("Sulphate Content (mg/kg)", min_value=0.0, max_value=50000.0, step=10.0)
moisture = st.number_input("Moisture Content (%)", min_value=0.0, max_value=100.0, step=0.1)

soil_type = st.selectbox("Soil Type", soil_types)
location = st.selectbox("Location wrt Water Table", location_types)
foreign = st.selectbox("Foreign Inclusion Type", foreign_types)

is_fill = st.selectbox("Is Fill Material?", ["No", "Yes"])
is_fill = 1 if is_fill == "Yes" else 0

# ----------- Prediction Button -----------

if st.button("Predict Corrosion Rate"):

    # Build dataframe for model
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

    st.success(f"### Predicted Mean Corrosion Rate (μ): **{mu:.4f} mm/yr**")
    st.info(f"### Predicted Uncertainty (σ): **{sigma:.4f} mm/yr**")

    # ----------- CDF Plot -----------
    x = np.linspace(max(0.0001, mu - 4*sigma), mu + 4*sigma, 500)
    cdf_vals = norm.cdf(x, loc=mu, scale=sigma)

    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, cdf_vals, label="CDF", color="blue", linewidth=2)
    ax.axvline(mu, color="red", linestyle="--", label=f"μ = {mu:.4f}")
    ax.set_xlabel("Corrosion Rate (mm/yr)")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution Function")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)
