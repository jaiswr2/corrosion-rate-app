import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import joblib

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Prediction of Corrosion Rate of Steel Piles Embedded in Soil",
    layout="centered"
)

# -------------------------------------------------------
# Load model and preprocessor
# -------------------------------------------------------
@st.cache_resource
def load_model_and_preprocessor():
    model = joblib.load("model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# -------------------------------------------------------
# Title & description
# -------------------------------------------------------
st.title("Prediction of Corrosion Rate of Steel Piles Embedded in Soil")

st.markdown(
    "This tool predicts the **mean corrosion rate (μ)** and **uncertainty (σ)** "
    "for steel piles embedded in soil using a probabilistic NGBoost model."
)

st.markdown("### Input Parameters")

# -------------------------------------------------------
# Training ranges (from your dataset summary)
# -------------------------------------------------------
RANGES = {
    "age": (1.0, 64.0),
    "pH": (3.4, 10.0),
    "chloride": (14.0, 11400.0),
    "resistivity": (80.0, 13106.04),
    "sulphate": (6.9, 21800.0),
    "moisture": (1.7, 261.4),
}

# Categorical options (exactly as provided)
SOIL_TYPES = [
    "Granite",
    "ML+SM",
    "ML",
    "CL+SC",
    "CL+ML",
    "SM",
    "CL",
    "SP+SM",
    "SP",
    "CH",
    "SW",
    "OL",
    "GP",
    "GP+GM",
    "SC"
]

FOREIGN_TYPES = [
    "Type_None",
    "Type_Shreded wood",
    "Type_Flyash",
    "Type_Cinder"
]

LOCATION_TYPES = [
    "Above WaterTable",
    "Fluctuation Zone",
    "Permanent Immersion"
]

# -------------------------------------------------------
# Inputs in two columns
# Left col: 5 numeric; Right col: remaining
# -------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        f"Age (yr) [{RANGES['age'][0]}–{RANGES['age'][1]}]",
        min_value=0.0, max_value=200.0, value=34.0, step=1.0
    )
    soil_pH = st.number_input(
        f"Soil pH [{RANGES['pH'][0]}–{RANGES['pH'][1]}]",
        min_value=0.0, max_value=14.0, value=7.8, step=0.1
    )
    chloride = st.number_input(
        f"Chloride Content (mg/kg) [{RANGES['chloride'][0]}–{RANGES['chloride'][1]}]",
        min_value=0.0, max_value=20000.0, value=500.0, step=10.0
    )
    resistivity = st.number_input(
        f"Soil Resistivity (Ω·cm) [{int(RANGES['resistivity'][0])}–{int(RANGES['resistivity'][1])}]",
        min_value=0.0, max_value=20000.0, value=800.0, step=10.0
    )
    sulphate = st.number_input(
        f"Sulphate Content (mg/kg) [{RANGES['sulphate'][0]}–{RANGES['sulphate'][1]}]",
        min_value=0.0, max_value=30000.0, value=300.0, step=10.0
    )

with col2:
    moisture = st.number_input(
        f"Moisture Content (%) [{RANGES['moisture'][0]}–{RANGES['moisture'][1]}]",
        min_value=0.0, max_value=300.0, value=30.0, step=0.1
    )

    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    location = st.selectbox("Location wrt Water Table", LOCATION_TYPES)
    foreign = st.selectbox("Foreign Inclusion Type", FOREIGN_TYPES)

    is_fill_str = st.selectbox("Is Fill Material?", ["No", "Yes"])
    is_fill = 1 if is_fill_str == "Yes" else 0

    threshold = st.number_input(
        "Threshold corrosion rate X for probability queries (mm/yr)",
        min_value=0.0, max_value=1.0, value=0.05, step=0.005,
        help="We will compute P(CR ≤ X) and P(CR > X)."
    )

# -------------------------------------------------------
# Range checks (inline warnings near inputs)
# -------------------------------------------------------
# We only *block prediction* inside the button click, but show
# warnings right under each input here.
def in_range(val, key):
    lo, hi = RANGES[key]
    return (val >= lo) and (val <= hi)

with col1:
    if age > 0 and not in_range(age, "age"):
        st.warning(f"Age is outside training range [{RANGES['age'][0]}, {RANGES['age'][1]}].",
                   icon="⚠️")
    if soil_pH > 0 and not in_range(soil_pH, "pH"):
        st.warning(f"Soil pH is outside training range [{RANGES['pH'][0]}, {RANGES['pH'][1]}].",
                   icon="⚠️")
    if chloride > 0 and not in_range(chloride, "chloride"):
        st.warning(
            f"Chloride is outside training range [{RANGES['chloride'][0]}, {RANGES['chloride'][1]}].",
            icon="⚠️"
        )
    if resistivity > 0 and not in_range(resistivity, "resistivity"):
        st.warning(
            f"Soil resistivity is outside training range "
            f"[{int(RANGES['resistivity'][0])}, {int(RANGES['resistivity'][1])}].",
            icon="⚠️"
        )
    if sulphate > 0 and not in_range(sulphate, "sulphate"):
        st.warning(
            f"Sulphate content is outside training range "
            f"[{RANGES['sulphate'][0]}, {RANGES['sulphate'][1]}].",
            icon="⚠️"
        )

with col2:
    if moisture > 0 and not in_range(moisture, "moisture"):
        st.warning(
            f"Moisture content is outside training range "
            f"[{RANGES['moisture'][0]}, {RANGES['moisture'][1]}].",
            icon="⚠️"
        )

# -------------------------------------------------------
# Prediction button
# -------------------------------------------------------
st.markdown("---")
if st.button("Predict Corrosion Rate"):

    # Hard check ranges before prediction
    out_of_range_msgs = []

    if not in_range(age, "age"):
        out_of_range_msgs.append("Age (yr)")
    if not in_range(soil_pH, "pH"):
        out_of_range_msgs.append("Soil pH")
    if not in_range(chloride, "chloride"):
        out_of_range_msgs.append("Chloride Content (mg/kg)")
    if not in_range(resistivity, "resistivity"):
        out_of_range_msgs.append("Soil Resistivity (Ω·cm)")
    if not in_range(sulphate, "sulphate"):
        out_of_range_msgs.append("Sulphate Content (mg/kg)")
    if not in_range(moisture, "moisture"):
        out_of_range_msgs.append("Moisture Content (%)")

    if out_of_range_msgs:
        st.error(
            "Prediction not performed because the following inputs are "
            "outside the training range:\n- " + "\n- ".join(out_of_range_msgs)
        )
    else:
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

        st.success(
            f"Most likely corrosion rate is **{mu:.4f} ± {sigma:.44f} mm/yr** "
            f"(μ ± σ)."
        )

        # Probability queries: P(CR ≤ X) and P(CR > X)
        if sigma > 0:
            p_le = norm.cdf(threshold, loc=mu, scale=sigma)
            p_gt = 1.0 - p_le

            st.info(
                f"**Probability statements for X = {threshold:.4f} mm/yr**\n\n"
                f"- P(CR ≤ X) = **{100 * p_le:.1f}%**\n"
                f"- P(CR > X) = **{100 * p_gt:.1f}%**"
            )
        else:
            st.warning("Predicted σ is zero or negative; probability "
                       "interpretation is not meaningful.")

        # ---------------------------------------------------
        # PDF & CDF side-by-side
        # ---------------------------------------------------
        st.markdown("### Predicted Distribution (Normal)")

        # Choose x-range
        x_min = max(0.0, mu - 4 * sigma)
        x_max = mu + 4 * sigma
        if x_max <= x_min:
            x_min = max(0.0, mu * 0.5)
            x_max = mu * 1.5 if mu > 0 else 1.0

        x = np.linspace(x_min, x_max, 600)
        pdf_vals = norm.pdf(x, loc=mu, scale=sigma) if sigma > 0 else np.zeros_like(x)
        cdf_vals = norm.cdf(x, loc=mu, scale=sigma) if sigma > 0 else np.zeros_like(x)

        out_col1, out_col2 = st.columns(2)

        # Common styling for larger fonts
        plt.rcParams.update({
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        })

        # -------- PDF in left column --------
        with out_col1:
            fig_pdf, ax_pdf = plt.subplots(figsize=(5.5, 4.5))
            ax_pdf.plot(x, pdf_vals, linewidth=2)
            ax_pdf.axvline(mu, color="red", linestyle="--", linewidth=1.5,
                           label=f"μ = {mu:.4f}")
            ax_pdf.axvline(threshold, color="green", linestyle=":", linewidth=1.5,
                           label=f"X = {threshold:.4f}")
            ax_pdf.set_xlabel("Corrosion Rate (mm/yr)")
            ax_pdf.set_ylabel("Probability Density")
            ax_pdf.set_title("PDF of Corrosion Rate")
            ax_pdf.grid(True, alpha=0.3)
            ax_pdf.legend()
            st.pyplot(fig_pdf)

        # -------- CDF in right column --------
        with out_col2:
            fig_cdf, ax_cdf = plt.subplots(figsize=(5.5, 4.5))
            ax_cdf.plot(x, cdf_vals, linewidth=2)
            ax_cdf.axvline(mu, color="red", linestyle="--", linewidth=1.5,
                           label=f"μ = {mu:.4f}")
            ax_cdf.axvline(threshold, color="green", linestyle=":", linewidth=1.5,
                           label=f"X = {threshold:.4f}")
            ax_cdf.set_xlabel("Corrosion Rate (mm/yr)")
            ax_cdf.set_ylabel("CDF  =  P(CR ≤ x)")
            ax_cdf.set_title("Cumulative Distribution Function")
            ax_cdf.grid(True, alpha=0.3)
            ax_cdf.legend()
            st.pyplot(fig_cdf)

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:14px;'>"
    "Developed by <b>Rishav Jaiswal</b> (McMaster University)"
    "</div>",
    unsafe_allow_html=True
)
