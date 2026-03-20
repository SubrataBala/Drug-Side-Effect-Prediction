from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd

# Add the project root to the path to import backend utils and predict functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from backend.predict import load_models, get_side_effects, get_interaction

# Initialize the Flask application.
# By default, Flask looks for templates in a folder named 'templates'
# in the same directory as the script, which is exactly your project's structure.
app = Flask(__name__)

@app.route('/')
def home():
    """
    This function runs when you visit the main page (e.g., http://127.0.0.1:5000/).
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

    # Hardcode the disease-to-medicine map for frontend suggestions based on user request
    disease_medicine_map = {
        'Diabetes': [
            'Metformin', 'Glimepiride', 'Sitagliptin', 'Insulin Glargine', 'Regular insulin', 
            'Dapagliflozin', 'Pioglitazone', 'Acarbose'
        ],
        'Hypertension': [
            'Amlodipine', 'Losartan', 'Telmisartan', 'Atenolol', 'Metoprolol', 
            'Hydrochlorothiazide', 'Enalapril'
        ],
        'Heart Disease': [
            'Aspirin', 'Clopidogrel', 'Atorvastatin', 'Nitroglycerin', 'Metoprolol', 
            'Ramipril', 'Isosorbide mononitrate'
        ],
        'Tuberculosis': [
            'Isoniazid', 'Rifampicin', 'Pyrazinamide', 'Ethambutol', 'Streptomycin', 
            'Bedaquiline', 'Levofloxacin'
        ],
        'Asthma': [
            'Salbutamol inhaler', 'Budesonide', 'Formoterol', 'Montelukast', 'Theophylline', 
            'Ipratropium bromide', 'Fluticasone'
        ],
        'Cancer': [
            'Cyclophosphamide', 'Methotrexate', 'Doxorubicin', 'Cisplatin', 'Paclitaxel', 
            'Imatinib', 'Pembrolizumab'
        ],
        'Dengue': [
            'Paracetamol (Acetaminophen)', 'Oral Rehydration Salts (ORS)', 'IV Fluids', 
            'Platelet transfusion', 'Vitamin C', 'Electrolyte solutions'
        ],
        'Malaria': [
            'Chloroquine', 'Artemether-Lumefantrine', 'Artesunate', 'Quinine', 'Primaquine', 
            'Mefloquine', 'Doxycycline'
        ],
        'Thyroid Disorder': [
            'Levothyroxine', 'Methimazole', 'Propylthiouracil (PTU)', 'Propranolol', 
            'Carbimazole', 'Radioactive iodine'
        ],
        'Mental Health Disorders': [
            'Fluoxetine', 'Sertraline', 'Escitalopram', 'Alprazolam', 'Diazepam', 
            'Risperidone', 'Olanzapine'
        ]
    }
    print("✅ Successfully loaded hardcoded disease-to-medicine map for suggestions.")

load_all_data()

@app.route('/predict_drug', methods=['POST'])
def predict_drug():
    """
    Handles drug prediction and interaction checking requests from the frontend.
    """
    if not all([model, vectorizer, side_effects_map, interaction_map]):
        return jsonify({"error": "Backend models not loaded. Please check server logs."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON data received."}), 400

    ongoing_diseases = data.get('ongoing_diseases', [])
    current_medications = data.get('current_medications', [])
    new_condition = data.get('new_condition')
    age_group = data.get('age_group') # Not used in current ML model, but good to pass

    if not new_condition:
        return jsonify({"error": "New condition is required for prediction."}), 400

    # Prepare input for the ML model (using new_condition as the primary input for prediction)
    from backend.utils import clean_text # Import clean_text
    cleaned_new_condition = clean_text(new_condition)
    input_vector = vectorizer.transform([cleaned_new_condition])
    predicted_medicine_name = model.predict(input_vector)[0]

    # Get side effects for the predicted medicine
    predicted_medicine_side_effects = get_side_effects(predicted_medicine_name, side_effects_map)
    
    # Get side effects for current medications
    current_medication_info = []
    for med in current_medications:
        med_name = med.get('name')
        med_disease = med.get('disease')
        if med_name:
            current_medication_info.append({
                "name": med_name,
                "disease": med_disease,
                "side_effects": get_side_effects(med_name, side_effects_map)
            })

    # Check for interactions between predicted and current medicines
    interactions_found = []
    for current_med_obj in current_medications:
        current_med_name = current_med_obj.get('name')
        if current_med_name:
            interaction_desc = get_interaction(predicted_medicine_name, current_med_name, interaction_map)
            if "No specific interaction" not in interaction_desc: # Filter out "no interaction" messages
                interactions_found.append({"medicines": f"{predicted_medicine_name} + {current_med_name}", "effect": interaction_desc})

    response_data = {
        "predicted_medicine": {
            "name": predicted_medicine_name,
            "side_effects": predicted_medicine_side_effects
        },
        "current_medication_info": current_medication_info,
        "interactions": interactions_found,
        "new_condition": new_condition # Include new_condition in the response
    }

    return jsonify(response_data)

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