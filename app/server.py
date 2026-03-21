from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import numpy as np
import json, re, time

# Add new imports for Gemini
import google.generativeai as genai
from dotenv import load_dotenv

# Add the project root to the path to import backend utils and predict functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.predict import load_models, get_side_effects, get_interaction
from backend.utils import clean_text

# --- NEW: Gemini API Configuration ---
# Load environment variables from a .env file if it exists.
load_dotenv()

# Global variable to hold the name of the dynamically selected Gemini model
SELECTED_GEMINI_MODEL = None

def find_available_gemini_model():
    """
    Checks available Gemini models and selects a suitable one that supports 'generateContent'.
    It iterates through a prioritized list and picks the first one that works.
    """
    global SELECTED_GEMINI_MODEL
    # A prioritized list of models to try. We prioritize models like 'gemini-1.5-flash-latest'
    # and the stable 'gemini-pro' as they are more likely to be included in the free tier.
    # This avoids models that may require a paid plan.
    preferred_models = ['gemini-1.5-flash-latest', 'gemini-pro', 'gemini-flash-latest', 'gemini-pro-latest', 'gemini-1.0-pro']
    
    try:
        print("🔎 Checking for available models that support 'generateContent'...")
        supported_models = {
            m.name.split('/')[-1] for m in genai.list_models() 
            if 'generateContent' in m.supported_generation_methods
        }
        
        for model_name in preferred_models:
            if model_name in supported_models:
                print(f"✅ Found and selected suitable model: '{model_name}'")
                SELECTED_GEMINI_MODEL = model_name
                return # Exit after finding the first (highest priority) suitable model
        
        print(f"⚠️ Could not find any of the preferred models {preferred_models} that support 'generateContent'.")
        print(f"   Available and supported models are: {list(supported_models)}")
    except Exception as e:
        print(f"❌ Error while listing models: {e}")

try:
    # Load the API key from the environment variable.
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: # Check if the key was loaded successfully
        print("⚠️ environment variable not set.  features will be disabled.")
    else:
        genai.configure(api_key=api_key)
        print("✅ configured successfully.")
        # After configuring, find a working model to use for predictions.
        find_available_gemini_model()
except Exception as e:
    print(f"❌ Error configuring : {e}")

# Initialize the Flask application.
# By default, Flask looks for templates in a folder named 'templates'
# in the same directory as the script, which is exactly your project's structure.
app = Flask(__name__)

@app.route('/')
def home():
    """
    This function runs when you visit the main page (e.g., http://127.0.0.1:5001/).
    It finds and returns the 'base.html' file from the 'templates' folder.
    """
    return render_template('base.html')

@app.route('/predict.html')
def predict_page():
    """
    This function serves the predict.html page.
    """
    return render_template('predict.html')

# Load models once when the app starts
model, vectorizer, side_effects_map, interaction_map, disease_medicine_map = None, None, None, None, {}

def load_all_data():
    """Loads ML models and the disease-to-medicine mapping."""
    global model, vectorizer, side_effects_map, interaction_map, disease_medicine_map
    
    # Load ML models
    try:
        model, vectorizer, side_effects_map, interaction_map = load_models()
        if not all([model, vectorizer, side_effects_map, interaction_map]):
            print("ERROR: Failed to load one or more models. Prediction functionality will be limited.")
    except Exception as e:
        print(f"ERROR: Exception during model loading: {e}")

    # Load the disease-to-medicine map from the JSON file for frontend suggestions
    suggestions_path = os.path.join(os.path.dirname(__file__), 'suggestions.json')
    try:
        with open(suggestions_path, 'r') as f:
            disease_medicine_map = json.load(f)
        print("✅ Successfully loaded disease-to-medicine map from suggestions.json.")
    except FileNotFoundError:
        print(f"WARNING: suggestions.json not found at {suggestions_path}. Suggestions will be unavailable.")
        disease_medicine_map = {}
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse suggestions.json. Suggestions will be unavailable.")
        disease_medicine_map = {}

# --- NEW: In-memory cache for Gemini API responses ---
gemini_cache = {} # Cache for full prediction requests
medicine_info_cache = {} # Cache for individual medicine side effect lookups

load_all_data()

def get_side_effects_via_gemini(medicine_name):
    """
    Gets side effects for a given medicine, prioritizing Gemini API with a local fallback.
    Results are cached to minimize API calls.
    """
    cache_key = medicine_name.strip().lower()
    if cache_key in medicine_info_cache:
        print(f"✅ Returning cached side effects for '{medicine_name}'.")
        return medicine_info_cache[cache_key]

    # Primary strategy: Use Gemini if it's configured
    if api_key and SELECTED_GEMINI_MODEL:
        print(f"🚀 No cache hit for '{medicine_name}'. Calling Gemini API for its side effects...")
        prompt = f"""
        You are a medical information AI. Your task is to list the common side effects for a specific medication.
        **Medication Name:** {medicine_name}
        **Your Task:** List the most common potential side effects for the provided medicine. Categorize their severity as "low", "medium", or "high".
        **Output Format:** Provide your response ONLY as a single, valid JSON object. Do not include any introductory text, explanations, or markdown formatting outside of the JSON structure.
        The JSON structure must be exactly as follows:
        ```json
        {{
          "side_effects": [
            {{"effect": "Common side effect 1", "severity": "low"}},
            {{"effect": "Common side effect 2", "severity": "medium"}}
          ]
        }}
        ```
        If you cannot find information for the medicine, the "side_effects" array should be empty.
        """
        max_retries, delay = 3, 2
        for attempt in range(max_retries):
            try:
                gemini_model = genai.GenerativeModel(SELECTED_GEMINI_MODEL)
                response = gemini_model.generate_content(prompt)
                parsed_data = parse_gemini_response(response.text)
                # If Gemini returns a non-empty list, we use it.
                if parsed_data and "side_effects" in parsed_data and parsed_data["side_effects"]:
                    side_effects = parsed_data["side_effects"]
                    print(f"✅ Caching Gemini-provided side effects for '{medicine_name}'.")
                    medicine_info_cache[cache_key] = side_effects
                    return side_effects
                break # Exit retry loop if we got a valid (even if empty) response
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit for side effect lookup. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    print(f"❌ Error getting side effects for '{medicine_name}' from Gemini: {e}")
                    break # Exit loop on non-retriable error

    # Fallback strategy: Use local data if Gemini failed, was unavailable, or returned no effects
    print(f"ℹ️ Using local data as primary source or fallback for '{medicine_name}'.")
    local_effects = get_side_effects(medicine_name, side_effects_map)
    print(f"✅ Caching local side effects for '{medicine_name}'.")
    medicine_info_cache[cache_key] = local_effects
    return local_effects

def parse_gemini_response(response_text):
    """Extracts and parses the JSON from Gemini's response."""
    # Gemini might wrap the JSON in ```json ... ```, so we extract it.
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        # If no markdown, assume the whole text is the JSON
        json_str = response_text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"❌ JSON Decode Error. Response was:\n{response_text}")
        return None

@app.route('/predict_drug', methods=['POST'])
def predict_drug():
    """
    Handles drug prediction using the Gemini API and checks local data for side effects.
    """
    # Check if the API key was loaded
    # This check ensures that the api_key variable set during startup is used.
    if not api_key:
        return jsonify({"error": "key not configured on the server."}), 500
    if not SELECTED_GEMINI_MODEL:
        return jsonify({"error": "No suitable model is available on the server. Please check server logs."}), 500

    # We still need the local maps for current medication info
    if not all([side_effects_map, interaction_map]):
        return jsonify({"error": "Backend data maps not loaded. Please check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data received."}), 400

    current_medications = data.get('current_medications', [])
    new_condition = data.get('new_condition')
    age_group = data.get('age_group')
    ongoing_diseases = data.get('ongoing_diseases', [])

    if not new_condition:
        return jsonify({"error": "New condition is required for prediction."}), 400

    # --- Create a cache key from the request data ---
    # Sorting ensures that the order of items doesn't create different keys
    current_meds_tuple = tuple(sorted([med['name'] for med in current_medications]))
    ongoing_diseases_tuple = tuple(sorted(ongoing_diseases))
    cache_key = (current_meds_tuple, new_condition, age_group, ongoing_diseases_tuple)

    # --- Check if the response is already in the cache ---
    if cache_key in gemini_cache:
        print("✅ Returning cached response.")
        gemini_data = gemini_cache[cache_key]
    else:
        print("🚀 No cache hit and Calling...")
        # --- 1. Construct the prompt for the Gemini API ---
        current_meds_str = ", ".join([med['name'] for med in current_medications]) if current_medications else "None"
        ongoing_diseases_str = ", ".join(ongoing_diseases) if ongoing_diseases else "None"
        
        prompt = f"""
        You are an expert medical AI assistant. Your task is to analyze patient information and suggest a suitable medication for a new condition, while also checking for potential interactions with their current medications.

        **Patient Information:**
        - **Age Group:** {age_group}
        - **Ongoing Diseases:** {ongoing_diseases_str}
        - **Current Medications:** {current_meds_str}
        - **New Condition/Symptom:** {new_condition}

        **Your Task:**
        1.  Based on the "New Condition/Symptom", predict a single, common, and appropriate medication.
        2.  List the most common potential side effects for the *predicted* medicine. Categorize their severity as "low", "medium", or "high".
        3.  Analyze potential interactions between the *predicted* medicine and *each* of the "Current Medications".

        **Output Format:**
        Please provide your response ONLY as a single, valid JSON object. Do not include any introductory text, explanations, or markdown formatting outside of the JSON structure.

        The JSON structure must be exactly as follows:
        ```json
        {{
          "predicted_medicine": {{
            "name": "PredictedMedicineName",
            "side_effects": [
              {{"effect": "Common side effect 1", "severity": "low"}},
              {{"effect": "Common side effect 2", "severity": "medium"}}
            ]
          }},
          "interactions": [
            {{
              "medicines": "PredictedMedicineName + CurrentMedicineName1",
              "effect": "Description of the potential interaction."
            }}
          ]
        }}
        ```
        If no interactions are found for a given pair, state that clearly in the "effect" field. If there are no current medications, the "interactions" array should be empty.
        """

        # --- NEW: Retry logic with exponential backoff for handling rate limits ---
        gemini_data = None
        max_retries = 3
        delay = 2  # Start with a 2-second delay

        for attempt in range(max_retries):
            try:
                gemini_model = genai.GenerativeModel(SELECTED_GEMINI_MODEL)
                response = gemini_model.generate_content(prompt)
                gemini_data = parse_gemini_response(response.text)
                if gemini_data:
                    print("✅ Caching new response.")
                    gemini_cache[cache_key] = gemini_data
                    break  # Success, exit the loop
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2  # Double the delay for the next attempt
                else:
                    print(f"❌ An error occurred while calling the data: {e}")
                    return jsonify({"error": f"An error occurred with the prediction service: {e}"}), 500
        
        if not gemini_data:
            return jsonify({"error": "Failed to get a response from the after multiple retries. Please try again later."}), 500

    try:
        current_medication_info = []
        for med in current_medications:
            med_name = med.get('name')
            if med_name:
                current_medication_info.append({
                    "name": med_name,
                    "disease": med.get('disease'),
                    "side_effects": get_side_effects_via_gemini(med_name)
                })

        # --- 5. Assemble the final response for the frontend ---
        response_data = {
            "predicted_medicine": gemini_data.get("predicted_medicine", {"name": "N/A", "side_effects": []}),
            "current_medication_info": current_medication_info,
            "interactions": gemini_data.get("interactions", []),
            "new_condition": new_condition
        }
        # Gemini doesn't provide a confidence score, so we can use a fixed value or remove it
        response_data["predicted_medicine"]["confidence"] = 0.95

        return jsonify(response_data)

    except Exception as e: # This handles errors during the final data assembly
        print(f"❌ An error occurred during response assembly: {e}")
        return jsonify({"error": f"An error occurred while assembling the final response: {e}"}), 500

@app.route('/api/disease_medicines')
def get_disease_medicines():
    """Provides a JSON object mapping diseases to a list of common medicines."""
    return jsonify(disease_medicine_map)

if __name__ == '__main__':
    """
    This starts the Flask development server.
    """
    print("✅ Flask server is running. Open http://127.0.0.1:5001 in your browser.")
    app.run(host='0.0.0.0', port=5001, debug=True)