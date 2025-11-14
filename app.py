import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="Corrosion Rate Predictor", layout="wide")

# ============================
# LOAD MODEL & PREPROCESSOR
# ============================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# ============================
# HEADER + LOGO
# ============================
logo_url = "https://upload.wikimedia.org/wikipedia/en/thumb/1/1d/McMaster_University_Coat_of_Arms.svg/256px-McMaster_University_Coat_of_Arms.svg.png"

header_col, logo_col = st.columns([8, 1])
with header_col:
    st.markdown(
        "<h1 style='text-align:center; color:#7A003C;'>"
        "Probabilistic Model for Predicting Corrosion of Steel Piles Embedded in Soil"
        "</h1>",
        unsafe_allow_html=True
    )

with logo_col:
    st.image(logo_url, width=80)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================
# VARIABLE RANGES
# ============================
ranges = {
    "age": (1, 70),
    "pH": (3.4, 10.0),
    "chloride": (10, 12000),
    "resistivity": (80, 13000),
    "sulphate": (7, 22000),
    "moisture": (2, 260),
}

soil_types = [
    "Granite","ML+SM","ML","CL+SC","CL+ML","SM","CL","SP+SM","SP","CH",
    "SW","OL","GP","GP+GM","SC"
]
foreign_types = ["Type_None", "Type_Shreded wood", "Type_Flyash", "Type_Cinder"]
location_types = ["Above WaterTable", "Fluctuation Zone", "Permanent Immersion"]

# ============================
# INPUT SECTION (TWO COLUMNS)
# ============================
left, right = st.columns(2)

warnings = {}  # store warnings to show beside fields

with left:
    age = st.number_input(f"Age (yr) [{ranges['age'][0]}–{ranges['age'][1]}]",
                          min_value=1, max_value=70, step=1)
    if not (ranges["age"][0] <= age <= ranges["age"][1]):
        warnings["age"] = "Warning: Out of trained range."

    soil_pH = st.number_input(f"Soil pH [{ranges['pH'][0]}–{ranges['pH'][1]}]",
                              min_value=3.4, max_value=10.0, step=0.1)
    if not (ranges["pH"][0] <= soil_pH <= ranges["pH"][1]):
        warnings["pH"] = "Warning: Out of trained range."

    chloride = st.number_input(
        f"Chloride Content (mg/kg) [{ranges['chloride'][0]}–{ranges['chloride'][1]}]",
        min_value=10.0, max_value=12000.0, step=1.0
    )
    if not (ranges["chloride"][0] <= chloride <= ranges["chloride"][1]):
        warnings["chloride"] = "Warning: Out of trained range."

    moisture = st.number_input(
        f"Moisture Content (%) [{ranges['moisture'][0]}–{ranges['moisture'][1]}]",
        min_value=2.0, max_value=260.0, step=0.1
    )
    if not (ranges["moisture"][0] <= moisture <= ranges["moisture"][1]):
        warnings["moisture"] = "Warning: Out of trained range."

with right:
    resistivity = st.number_input(
        f"Soil Resistivity (Ω·cm) [{ranges['resistivity'][0]}–{ranges['resistivity'][1]}]",
        min_value=80.0, max_value=13000.0, step=1.0
    )
    if not (ranges["resistivity"][0] <= resistivity <= ranges["resistivity"][1]):
        warnings["resistivity"] = "Warning: Out of trained range."

    sulphate = st.number_input(
        f"Sulphate Content (mg/kg) [{ranges['sulphate'][0]}–{ranges['sulphate'][1]}]",
        min_value=7.0, max_value=22000.0, step=1.0
    )
    if not (ranges["sulphate"][0] <= sulphate <= ranges["sulphate"][1]):
        warnings["sulphate"] = "Warning: Out of trained range."

    soil_type = st.selectbox("Soil Type", soil_types)
    foreign = st.selectbox("Foreign Inclusion Type", foreign_types)
    location = st.selectbox("Location wrt Water Table", location_types)

    is_fill = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill == "Yes" else 0

# Show warnings beside inputs
for key, msg in warnings.items():
    st.markdown(f"<p style='color:red;'>{msg}</p>", unsafe_allow_html=True)

# ============================
# PREDICT
# ============================
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
        "Is_Fill_Material": is_fill,
    }])

    Xp = preprocessor.transform(raw)
    dist = model.pred_dist(Xp)

    mu = float(dist.params["loc"])
    sigma = float(dist.params["scale"])

    st.markdown(
        f"<h3 style='color:#7A003C;'>Most likely corrosion rate: "
        f"<b>{mu:.4f} ± {sigma:.4f}</b> mm/yr (μ ± σ)</h3>",
        unsafe_allow_html=True
    )

    # ============================
    # SMALL EQUAL-SIZE PDF & CDF
    # ============================
    pdf_col, cdf_col = st.columns(2)

    x_vals = np.linspace(max(0.0001, mu - 4 * sigma), mu + 4 * sigma, 500)
    pdf_vals = norm.pdf(x_vals, loc=mu, scale=sigma)
    cdf_vals = norm.cdf(x_vals, loc=mu, scale=sigma)

    # ---- PDF ----
    with pdf_col:
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.plot(x_vals, pdf_vals, color="#7A003C", linewidth=2)
        ax1.axvline(mu, color="black", linestyle="--")
        ax1.set_title("PDF – Probability Density", fontsize=12)
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

    # ---- CDF ----
    with cdf_col:
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        ax2.plot(x_vals, cdf_vals, color="#C5A900", linewidth=2)
        ax2.axvline(mu, color="black", linestyle="--")
        ax2.set_title("CDF – Probability (Corrosion Rate ≤ X)", fontsize=12)
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

# ============================
# FOOTER
# ============================
st.markdown(
    "<h3 style='text-align:center; color:#7A003C;'>Developed by <b>Rishav Jaiswal</b><br>McMaster University</h3>",
    unsafe_allow_html=True
)
