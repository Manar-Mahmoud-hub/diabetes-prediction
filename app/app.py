from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_diabetes_model.joblib"

st.set_page_config(page_title="Test du modèle diabète", page_icon="🩺")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# Exemples prêts à l'emploi (valeurs réelles du dataset)
EXAMPLES = {
    "Saisie libre": None,
    "Exemple 1 : patient à risque": dict(Age=58, Gender="F", BMI=36.4, Glucose_Level=151,
                                         Blood_Pressure=102, Insulin=98.5,
                                         Physical_Activity=86, Family_History="Yes"),
    "Exemple 2 : patient sain": dict(Age=34, Gender="F", BMI=24.0, Glucose_Level=86,
                                     Blood_Pressure=139, Insulin=181.0,
                                     Physical_Activity=46, Family_History="No"),
}

st.title("🩺 Test du modèle de prédiction du diabète")
st.caption("Projet pédagogique. Ne remplace pas un avis médical.")

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}\n\nExécutez d'abord le notebook 03_modeling.")
    st.stop()

model = load_model()

choice = st.selectbox("Charger un exemple", list(EXAMPLES))
ex = EXAMPLES[choice] or dict(Age=50, Gender="M", BMI=28.0, Glucose_Level=120,
                              Blood_Pressure=100, Insulin=150.0,
                              Physical_Activity=60, Family_History="No")

with st.form("patient"):
    c1, c2 = st.columns(2)
    age = c1.number_input("Âge", 18, 100, ex["Age"])
    gender = c2.selectbox("Genre", ["F", "M"], index=["F", "M"].index(ex["Gender"]))
    bmi = c1.number_input("IMC (BMI)", 10.0, 60.0, float(ex["BMI"]), step=0.1)
    glucose = c2.number_input("Glucose", 50, 300, ex["Glucose_Level"])
    bp = c1.number_input("Tension artérielle", 40, 220, ex["Blood_Pressure"])
    insulin = c2.number_input("Insuline", 0.0, 400.0, float(ex["Insulin"]), step=0.5)
    activity = c1.number_input("Activité physique", 0, 200, ex["Physical_Activity"])
    family = c2.selectbox("Antécédents familiaux", ["No", "Yes"],
                          index=["No", "Yes"].index(ex["Family_History"]))
    submitted = st.form_submit_button("Prédire")

if submitted:
    patient = pd.DataFrame([{
        "Age": age, "Gender": gender, "BMI": bmi, "Glucose_Level": glucose,
        "Blood_Pressure": bp, "Insulin": insulin, "Physical_Activity": activity,
        "Family_History": family,
    }])
    proba = float(model.predict_proba(patient)[0, 1])
    pred = int(model.predict(patient)[0])

    if pred == 1:
        st.error(f"Prédiction : **diabétique** (probabilité {proba:.1%})")
    else:
        st.success(f"Prédiction : **non diabétique** (probabilité de diabète {proba:.1%})")
    st.progress(proba)
