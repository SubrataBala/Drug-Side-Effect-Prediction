import os
import pickle
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.utils import clean_text

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "saved_models")
model_path = os.path.join(models_dir, "drug_model.pkl")
vectorizer_path = os.path.join(models_dir, "hashing_vectorizer.pkl")
side_effects_path = os.path.join(models_dir, "side_effects_map.pkl")
interaction_path = os.path.join(models_dir, "interaction_map.pkl") # New path

def load_models():
    """Loads all necessary models and data maps from disk."""
    paths = [model_path, vectorizer_path, side_effects_path, interaction_path]
    if not all(os.path.exists(p) for p in paths):
        print(f"Error: One or more model files not found in {models_dir}.")
        print("Please run 'train_model.py' first to generate all models.")
        return None, None, None, None
    
    print(f"Loading models from {models_dir}...")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
        with open(side_effects_path, "rb") as f:
            original_side_effects_map = pickle.load(f)
        with open(interaction_path, "rb") as f:
            original_interaction_map = pickle.load(f)

        # Convert map keys to lowercase for case-insensitive lookups
        side_effects_map = {k.lower(): v for k, v in original_side_effects_map.items()}
        interaction_map = {k.lower(): v for k, v in original_interaction_map.items()}

        return model, vectorizer, side_effects_map, interaction_map
    except Exception as e:
        print(f"An error occurred while loading models: {e}")
        return None, None, None, None

def get_side_effects(medicine_name, side_effects_map):
    """Looks up side effects from the pre-loaded map."""
    if not medicine_name:
        return [{"effect": "Side effect data not available for this medicine.", "severity": "low"}]

    # Use lowercased and stripped medicine name for robust, case-insensitive lookup
    lookup_key = medicine_name.strip().lower()
    side_effects_str = side_effects_map.get(lookup_key)

    # If the medicine is not found or has no side effects listed, return a default message.
    if not side_effects_str or not side_effects_str.strip() or "not available" in side_effects_str.lower():
        return [{"effect": "Side effect data not available for this medicine.", "severity": "low"}]

    # Split the string and create the list of side effect objects
    effects = [{"effect": s.strip(), "severity": "medium"} for s in side_effects_str.split(',') if s.strip()]

    # If splitting resulted in an empty list (e.g., the string was just ',,'), return a default.
    if not effects:
        return [{"effect": "No specific side effects listed in the dataset.", "severity": "low"}]
    return effects

def get_interaction(med1, med2, interaction_map):
    """Checks for interactions between two medicines using the pre-loaded map."""
    if not med1 or not med2:
        return "Cannot check interaction for empty medicine name."

    if not interaction_map:
        return "Interaction data not available."
    
    med1_lower, med2_lower = med1.strip().lower(), med2.strip().lower()

    # Check interaction from med1's perspective
    if med1_lower in interaction_map:
        med1_info = interaction_map[med1_lower]
        interacts_with_lower = [d.lower() for d in med1_info.get('interacts_with', [])]
        if med2_lower in interacts_with_lower:
            return med1_info.get('effect', f"Interaction found between {med1} and {med2}, but no description is available.")

    # Check interaction from med2's perspective
    if med2_lower in interaction_map:
        med2_info = interaction_map[med2_lower]
        interacts_with_lower = [d.lower() for d in med2_info.get('interacts_with', [])]
        if med1_lower in interacts_with_lower:
            return med2_info.get('effect', f"Interaction found between {med2} and {med1}, but no description is available.")
            
    return f"No specific interaction found between {med1} and {med2} in the dataset."

def main():
    # Load all models and maps
    model, vectorizer, side_effects_map, interaction_map = load_models()
    if not all([model, vectorizer, side_effects_map, interaction_map]):
        return

    print("\n✅ Models loaded successfully!")
    print("--- Drug Recommendation & Interaction Check (Terminal Version) ---")
    
    # Get user input interactively
    regular_medicine = input("💊 Enter the medicine you take regularly (e.g., Metformin): ")
    new_issue = input("🤒 Describe your new symptoms or health condition: ")

    if not regular_medicine.strip() or not new_issue.strip():
        print("\n⚠️ Please provide both your regular medicine and new health issue.")
        return

    # 1. Clean the input for the new issue
    cleaned_text = clean_text(new_issue)
    
    # 2. Vectorize the input
    features = vectorizer.transform([cleaned_text])
    
    # 3. Predict the medicine for the new issue
    predicted_medicine = model.predict(features)[0]
    
    # 4. Get side effects and check for interactions
    interaction = get_interaction(regular_medicine, predicted_medicine, interaction_map)

    # 5. Display all results in the terminal
    print("\n" + "="*25 + " RESULTS " + "="*25)
    print(f"\n🚨 Interaction Check: '{regular_medicine}' + '{predicted_medicine}'")
    print(f"   └── {interaction}")
    print(f"\n✅ Recommended Medicine for New Issue: {predicted_medicine}")
    print(f"   └── Possible Side Effects: {get_side_effects(predicted_medicine, side_effects_map)}")
    print(f"\nℹ️ Information on Your Regular Medicine: {regular_medicine}")
    print(f"   └── Known Side Effects: {get_side_effects(regular_medicine, side_effects_map)}")
    print("\n" + "="*60)
    print("⚠️ Disclaimer: This is an AI suggestion. Always consult a doctor before taking any medicine.")

if __name__ == "__main__": # Guard the main function so it doesn't run on import
    main()