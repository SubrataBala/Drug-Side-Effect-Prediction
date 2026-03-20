import streamlit as st
import pickle
import pandas as pd
import os
import sys

# Add the project root to the path to import backend utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.utils import clean_text

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
# Must be the first Streamlit command
st.set_page_config(page_title="Drug Recommendation & Interaction Check", layout="centered")

# -----------------------------
# Load saved ML model & vectorizer
# -----------------------------
@st.cache_resource
def load_models():
    """Loads the model, vectorizer, and data maps from disk."""
    try:
        model_path = "../backend/saved_models/drug_model.pkl"
        vectorizer_path = "../backend/saved_models/hashing_vectorizer.pkl"
        side_effects_path = "../backend/saved_models/side_effects_map.pkl"
        interaction_path = "../backend/saved_models/interaction_map.pkl"

        model = pickle.load(open(model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
        side_effects_map = pickle.load(open(side_effects_path, "rb"))
        interaction_map = pickle.load(open(interaction_path, "rb"))
        
        return model, vectorizer, side_effects_map, interaction_map
    except Exception as e:
        st.error(f"Error loading models: {e}. Please run 'train_model.py' to generate them.")
        return None, None, None, None

model, vectorizer, side_effects_map, interaction_map = load_models()

if not all([model, vectorizer, side_effects_map, interaction_map]):
    st.stop()

# -----------------------------
# Helper functions
# -----------------------------
def get_side_effects(medicine_name):
    """Looks up side effects from the pre-loaded map."""
    return side_effects_map.get(medicine_name, "Side effect data not available in the trained model.")

def get_interaction(med1, med2):
    """Checks for interactions between two medicines using the pre-loaded map."""
    if not interaction_map or not med1 or not med2:
        return "Interaction data not available."
    
    med1_lower, med2_lower = med1.lower(), med2.lower()

    # Check interaction from med1's perspective
    if med1 in interaction_map:
        med1_info = interaction_map[med1]
        interacts_with_lower = [d.lower() for d in med1_info.get('interacts_with', [])]
        if med2_lower in interacts_with_lower:
            return med1_info.get('effect', f"Interaction found between {med1} and {med2}, but no description is available.")

    # Check interaction from med2's perspective
    if med2 in interaction_map:
        med2_info = interaction_map[med2]
        interacts_with_lower = [d.lower() for d in med2_info.get('interacts_with', [])]
        if med1_lower in interacts_with_lower:
            return med2_info.get('effect', f"Interaction found between {med2} and {med1}, but no description is available.")
            
    return f"No specific interaction found between **{med1}** and **{med2}** in the dataset."

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

        predicted_medicine = model.predict(input_vector)[0]

        # Get side effects for both medicines
        predicted_side_effects = get_side_effects(predicted_medicine)
        regular_side_effects = get_side_effects(regular_medicine)
        
        # Check for interaction
        interaction = get_interaction(predicted_medicine, regular_medicine)

        st.markdown("---")
        st.header("Results")

        # Display interaction first as it's most critical
        st.subheader(f"🚨 Interaction Check: `{regular_medicine}` + `{predicted_medicine}`")
        st.info(str(interaction))

        # Display prediction and its side effects
        st.subheader(f"✅ Recommended Medicine for New Issue: **{predicted_medicine}**")
        with st.expander(f"See possible side effects of {predicted_medicine}"):
            st.write(predicted_side_effects)
        
        # Display info for regular medicine
        st.subheader(f"ℹ️ Information on Your Regular Medicine: **{regular_medicine}**")
        with st.expander(f"See known side effects of {regular_medicine}"):
            st.write(regular_side_effects)

        st.markdown("---")
        st.caption("⚠️ Disclaimer: This system provides AI-based suggestions only. Please consult a doctor before taking any medicine.")
