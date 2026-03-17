import streamlit as st
import pickle
import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords

# -----------------------------
# Streamlit UI Configuration
# -----------------------------
# Must be the first Streamlit command
st.set_page_config(page_title="Drug Recommendation System", layout="centered")

nltk.download('stopwords')

# -----------------------------
# Load saved ML model & vectorizer
# -----------------------------
@st.cache_resource
def load_models():
    try:
        model = pickle.load(open("../backend/saved_models/drug_model.pkl", "rb"))
        vectorizer = pickle.load(open("../backend/saved_models/tfidf_vectorizer.pkl", "rb"))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure the pickle files are valid.")
        return None, None

model, vectorizer = load_models()

if model is None or vectorizer is None:
    st.stop()

# -----------------------------
# Load datasets
# -----------------------------
data_dir = "../data"
dfs = []

# Load all CSVs to ensure we have side effect data for all medicines used in training
if os.path.exists(data_dir):
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".csv"):
                try:
                    temp_df = pd.read_csv(os.path.join(root, file), low_memory=False)
                    temp_df.columns = temp_df.columns.str.strip()
                    # Normalize column names
                    for col in ['Drug', 'Drug Name', 'drugName', 'Medicine', 'drug', 'Drug_Name', 'medicine', 'drug_name', 'name', 'Name']:
                        if col in temp_df.columns:
                            temp_df.rename(columns={col: 'Medicine Name'}, inplace=True)
                    
                    # Normalize Side Effects column
                    for col in ['Side Effects', 'SideEffects', 'sideEffects', 'side_effects', 'sideEffect', 'SideEffect']:
                        if col in temp_df.columns:
                            temp_df.rename(columns={col: 'Side Effects'}, inplace=True)
                    
                    # Normalize Substitute column
                    for col in ['Substitute', 'substitute', 'Alternative', 'alternative', 'substitutes']:
                        if col in temp_df.columns:
                            temp_df.rename(columns={col: 'Substitute'}, inplace=True)
                            
                    if 'Medicine Name' in temp_df.columns:
                        dfs.append(temp_df)
                except:
                    pass

medicine_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

food_path = "../data/Drug to Food interactions Dataset.csv"
food_df = pd.read_csv(food_path) if os.path.exists(food_path) else pd.DataFrame(columns=['Drug', 'Food Interaction'])

# -----------------------------
# Text cleaning function
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# -----------------------------
# Helper functions
# -----------------------------
def get_side_effects(medicine_name):
    rows = medicine_df[medicine_df['Medicine Name'].str.lower() == medicine_name.lower()]
    if not rows.empty:
        # Return the first non-empty side effect found
        if 'Side Effects' in rows.columns:
            for effect in rows['Side Effects'].values:
                if isinstance(effect, str) and len(effect) > 3 and effect.lower() != 'not available':
                    return effect
    return "Side effect data not available"

def get_substitute(medicine_name):
    rows = medicine_df[medicine_df['Medicine Name'].str.lower() == medicine_name.lower()]
    if not rows.empty and 'Substitute' in rows.columns:
        for sub in rows['Substitute'].values:
            if isinstance(sub, str) and len(sub) > 3 and sub.lower() != 'not available':
                return sub
    return "No substitute information available"

def get_food_interaction(medicine_name):
    row = food_df[food_df['Drug'].str.lower() == medicine_name.lower()]
    if len(row) > 0:
        return row['Food Interaction'].values[0]
    else:
        return "No major food interaction found"

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("💊 Drug Recommendation & Side-Effect Prediction")
st.write("AI-based assistance system (Not a medical prescription)")

st.markdown("---")

# User Inputs
disease = st.text_input("🩺 Enter Major Disease (e.g. Diabetes, BP, Asthma)")
issue = st.text_area("🤒 Enter Current Issue / Symptoms")

# Predict Button
if st.button("🔍 Predict Medicine"):
    if disease.strip() == "" or issue.strip() == "":
        st.warning("Please enter both disease and current issue.")
    else:
        input_text = clean_text(disease + " " + issue)
        input_vector = vectorizer.transform([input_text])

        predicted_medicine = model.predict(input_vector)[0]

        side_effects = get_side_effects(predicted_medicine)
        substitute = get_substitute(predicted_medicine)
        food_warning = get_food_interaction(predicted_medicine)

        st.success(f"✅ Recommended Medicine: **{predicted_medicine}**")

        st.subheader("⚠️ Possible Side Effects")
        st.write(side_effects)
        
        st.subheader("🔄 Substitute Medicine")
        st.write(substitute)

        st.subheader("🍽️ Food Interaction Warning")
        st.write(food_warning)

        st.markdown("---")
        st.caption("⚠️ Disclaimer: This system provides AI-based suggestions only. Please consult a doctor before taking any medicine.")
