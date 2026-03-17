import os
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Ensure resources are downloaded
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Same cleaning function as used in training to ensure consistency.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "saved_models")
model_path = os.path.join(models_dir, "drug_model.pkl")
vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
side_effects_path = os.path.join(models_dir, "side_effects_map.pkl")

def load_models():
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        print(f"Error: Models not found in {models_dir}.")
        print("Please run 'train_model.py' first to generate the models.")
        return None, None, None
    
    print(f"Loading models from {models_dir}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    side_effects_map = {}
    if os.path.exists(side_effects_path):
        with open(side_effects_path, "rb") as f:
            side_effects_map = pickle.load(f)
            
    return model, vectorizer, side_effects_map

def run_prediction():
    model, vectorizer, side_effects_map = load_models()
    if not model:
        return

    print("\n✅ Model loaded successfully!")
    print("Enter a condition, symptom, or review text to get a drug recommendation.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input(">> Enter text: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        if not user_input.strip():
            continue

        # 1. Clean the input
        cleaned_text = clean_text(user_input)
        
        # 2. Vectorize the input (transform expects a list)
        features = vectorizer.transform([cleaned_text])
        
        # 3. Predict
        prediction = model.predict(features)[0]
        
        # Get Side Effects
        side_effect_raw = side_effects_map.get(prediction)
        # Check if the retrieved side effect is a valid, meaningful string
        if not side_effect_raw or side_effect_raw.lower() == 'not available':
            side_effect = "Information not available"
        else:
            side_effect = side_effect_raw
        # 4. Get Confidence (Probability)
        try:
            probs = model.predict_proba(features)
            confidence = probs.max() * 100
            print(f"💊 Recommended Medicine: {prediction} (Confidence: {confidence:.2f}%)")
        except AttributeError:
            # Some models don't support predict_proba
            print(f"💊 Recommended Medicine: {prediction}")
            
        print(f"⚠️  Side Effects: {side_effect}")
        print("-" * 40)

if __name__ == "__main__":
    run_prediction()