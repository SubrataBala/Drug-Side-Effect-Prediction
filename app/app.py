import streamlit as st
import pickle
import pandas as pd
import os
import sys
import numpy as np

# Add the project root to the path to import backend utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.utils import clean_text
from backend.predict import get_interaction, get_side_effects

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
# Must be the first Streamlit command
st.set_page_config(page_title="Drug Recommendation & Interaction Check", layout="centered")

@st.cache_resource
def load_all_models():
    """Loads all models using the robust loader from the backend."""
    from backend.predict import load_models
    return load_models()

model, vectorizer, side_effects_map, interaction_map = load_all_models()

if not all([model, vectorizer, side_effects_map, interaction_map]):
    st.stop()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💊 Drug Recommendation & Interaction Check")
st.write("AI-based assistance system (Not a medical prescription)")

st.markdown("---")

# User Inputs
st.subheader("Step 1: Your Current Medication")
regular_medicine = st.text_input("💊 Enter the medicine you take regularly (e.g., Metformin, Escitalopram)")

st.subheader("Step 2: Your New Health Issue")
issue = st.text_area("🤒 Describe your new symptoms or health condition")

# Predict Button
if st.button("🔍 Predict & Check Interactions"):
    if not regular_medicine.strip() or not issue.strip():
        st.warning("Please enter both your regular medicine and your new health issue.")
    else:
        input_text = clean_text(issue)
        input_vector = vectorizer.transform([input_text])

        # Get prediction probabilities to show a confidence score
        probabilities = model.predict_proba(input_vector)[0]
        confidence = np.max(probabilities)
        predicted_medicine_index = np.argmax(probabilities)
        predicted_medicine = model.classes_[predicted_medicine_index]

        # Get side effects for both medicines
        predicted_side_effects = get_side_effects(predicted_medicine, side_effects_map)
        regular_side_effects = get_side_effects(regular_medicine, side_effects_map)
        
        # Check for interaction
        interaction = get_interaction(predicted_medicine, regular_medicine, interaction_map)

        st.markdown("---")
        st.header("Results")

        # Display interaction first as it's most critical
        st.subheader(f"🚨 Interaction Check: `{regular_medicine}` + `{predicted_medicine}`")
        st.info(str(interaction))

        # Display prediction and its side effects
        st.subheader(f"✅ Recommended Medicine for New Issue: **{predicted_medicine}**")
        st.metric(label="Prediction Confidence", value=f"{confidence:.1%}")
        st.caption("This score represents how confident the AI model is in its recommendation. A higher score is better.")
        with st.expander(f"See possible side effects of {predicted_medicine}"):
            for effect in predicted_side_effects:
                st.markdown(f"- {effect.get('effect')} (`Severity: {effect.get('severity', 'N/A')}`)")
        
        # Display info for regular medicine
        st.subheader(f"ℹ️ Information on Your Regular Medicine: **{regular_medicine}**")
        with st.expander(f"See known side effects of {regular_medicine}"):
            for effect in regular_side_effects:
                st.markdown(f"- {effect.get('effect')} (`Severity: {effect.get('severity', 'N/A')}`)")

        st.markdown("---")
        st.caption("⚠️ Disclaimer: This system provides AI-based suggestions only. Please consult a doctor before taking any medicine.")
