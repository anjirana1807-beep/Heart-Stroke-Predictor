import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Stroke Predictor", page_icon="❤️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0f1a;
    color: #e8eaf0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header {visibility: hidden;}

/* Hero section */
.hero {
    text-align: center;
    padding: 1.5rem 0 2rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}
.hero h1 span { color: #e74c6e; }
.hero p {
    color: #8892a4;
    font-size: 0.95rem;
    font-weight: 300;
}
.pulse-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #1a0811;
    border: 1px solid #3a1020;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.78rem;
    color: #e74c6e;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 0.8rem;
}
.pulse-dot {
    width: 8px; height: 8px;
    background: #e74c6e;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.6); opacity: 0.5; }
}

/* ECG animation */
.ecg-container {
    width: 100%;
    overflow: hidden;
    height: 50px;
    margin-bottom: 1.5rem;
    opacity: 0.7;
}

/* Section labels */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #3a5070;
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1a2535;
}

/* Cards */
.info-card {
    background: #111827;
    border: 1px solid #1e2d42;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s;
}
.info-card:hover { border-color: #2e4060; }

/* Result boxes */
.result-high {
    background: #1a0811;
    border: 1.5px solid #c0304e;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
}
.result-low {
    background: #071812;
    border: 1.5px solid #0d9e6e;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-high h2 { color: #e74c6e; font-family: 'Syne', sans-serif; font-size: 1.4rem; }
.result-low  h2 { color: #0fbe87; font-family: 'Syne', sans-serif; font-size: 1.4rem; }
.result-sub { color: #8892a4; font-size: 0.9rem; margin-top: 0.4rem; }

/* Streamlit widget overrides */
div[data-testid="stSlider"] > div { padding: 0; }

div[data-testid="stSelectbox"] > div > div {
    background: #0d1520 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

div[data-testid="stNumberInput"] input {
    background: #0d1520 !important;
    border: 1px solid #1e2d42 !important;
    border-radius: 8px !important;
    color: #e8eaf0 !important;
}

/* Button */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #c0304e, #e74c6e) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.8rem 1rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s !important;
    margin-top: 1rem;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(231,76,110,0.35) !important;
}

/* Slider label color */
label { color: #8892a4 !important; font-size: 0.85rem !important; }

/* Metric cards */
.metric-card {
    background: #111827;
    border: 1px solid #1e2d42;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-label { font-size: 0.72rem; color: #5a7a9e; text-transform: uppercase; letter-spacing: 0.8px; }
.metric-value { font-size: 1.3rem; font-weight: 600; color: #e74c6e; font-family: 'Syne', sans-serif; }
</style>

<!-- Hero -->
<div class="hero">
    <svg width="100%" height="50" viewBox="0 0 780 50" xmlns="http://www.w3.org/2000/svg" style="margin-bottom:1rem;opacity:.7">
        <polyline points="0,25 80,25 100,25 110,5 120,45 130,2 145,48 155,25 250,25 330,25 340,5 350,45 360,2 375,48 385,25 480,25 560,25 570,5 580,45 590,2 605,48 615,25 710,25 780,25"
            stroke="#e74c6e" stroke-width="1.5" fill="none">
            <animateTransform attributeName="transform" type="translate" from="0,0" to="-260,0" dur="2.4s" repeatCount="indefinite"/>
        </polyline>
    </svg>
    <div class="pulse-badge"><span class="pulse-dot"></span> Cardiac Risk Analyzer</div>
    <h1>Heart <span>Stroke</span> Predictor</h1>
    <p>Enter your vitals below — get an instant risk assessment powered by KNN</p>
</div>
""", unsafe_allow_html=True)

# Load model assets
@st.cache_resource
def load_model():
    model = joblib.load("knn_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    expected_columns = joblib.load("columns.pkl")
    return model, scaler, expected_columns

try:
    model, scaler, expected_columns = load_model()
    model_loaded = True
except Exception:
    model_loaded = False

# ── Personal Info ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Personal Info</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 40)
with col2:
    sex = st.selectbox("Sex", ["M", "F"], format_func=lambda x: "Male" if x == "M" else "Female")

col3, col4 = st.columns(2)
with col3:
    max_hr = st.slider("Max Heart Rate (bpm)", 60, 220, 150)
with col4:
    oldpeak = st.slider("Oldpeak — ST Depression", 0.0, 6.0, 1.0, step=0.1)

# ── Clinical Measurements ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">Clinical Measurements</div>', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
with col6:
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

col7, col8 = st.columns(2)
with col7:
    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1],
        format_func=lambda x: "Yes (1)" if x == 1 else "No (0)"
    )
with col8:
    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["N", "Y"],
        format_func=lambda x: "Yes" if x == "Y" else "No"
    )

# ── ECG & Pain Indicators ─────────────────────────────────────────────────────
st.markdown('<div class="section-label">ECG & Pain Indicators</div>', unsafe_allow_html=True)

col9, col10 = st.columns(2)
with col9:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"],
        help="ATA=Atypical Angina · NAP=Non-Anginal · TA=Typical Angina · ASY=Asymptomatic"
    )
with col10:
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"],
    format_func=lambda x: f"{'↑ ' if x=='Up' else '→ ' if x=='Flat' else '↓ '}{x}"
)

# ── Quick Summary Cards ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">Your Summary</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-card"><div class="metric-label">Age</div><div class="metric-value">{age}</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="metric-label">Max HR</div><div class="metric-value">{max_hr}</div></div>', unsafe_allow_html=True)
m3.markdown(f'<div class="metric-card"><div class="metric-label">BP</div><div class="metric-value">{resting_bp}</div></div>', unsafe_allow_html=True)
m4.markdown(f'<div class="metric-card"><div class="metric-label">Chol.</div><div class="metric-value">{cholesterol}</div></div>', unsafe_allow_html=True)

st.write("")

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("❤️  Analyze Heart Risk"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        f'Sex_{sex}': 1,
        f'ChestPainType_{chest_pain}': 1,
        f'RestingECG_{resting_ecg}': 1,
        f'ExerciseAngina_{exercise_angina}': 1,
        f'ST_Slope_{st_slope}': 1,
    }

    input_df = pd.DataFrame([raw_input])

    if model_loaded:
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
    else:
        # Demo fallback scoring when model files are not present
        score = 0
        if age > 55: score += 2
        elif age > 45: score += 1
        if sex == 'M': score += 1
        if chest_pain == 'ASY': score += 3
        elif chest_pain == 'TA': score += 2
        if resting_bp > 140: score += 2
        if cholesterol > 240: score += 2
        if fasting_bs == 1: score += 1
        if resting_ecg == 'LVH': score += 2
        elif resting_ecg == 'ST': score += 1
        if max_hr < 120: score += 2
        if exercise_angina == 'Y': score += 2
        if oldpeak > 2: score += 2
        if st_slope == 'Down': score += 2
        elif st_slope == 'Flat': score += 1
        prediction = 1 if score >= 10 else 0

    if prediction == 1:
        st.markdown("""
        <div class="result-high">
            <div style="font-size:2.2rem;margin-bottom:.5rem">⚠️</div>
            <h2>High Risk of Heart Disease</h2>
            <p class="result-sub">Several risk factors detected. Please consult a cardiologist for a complete evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-low">
            <div style="font-size:2.2rem;margin-bottom:.5rem">✅</div>
            <h2>Low Risk of Heart Disease</h2>
            <p class="result-sub">Your indicators look favorable. Keep maintaining a healthy lifestyle and schedule regular checkups.</p>
        </div>
        """, unsafe_allow_html=True)

    if not model_loaded:
        st.info("💡 Model files not found — showing demo scoring. Place `knn_heart_model.pkl`, `heart_scaler.pkl`, and `heart_columns.pkl` in the same folder to use the real KNN model.")