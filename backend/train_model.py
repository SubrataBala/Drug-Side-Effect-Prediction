import pandas as pd
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import re
import nltk
from nltk.corpus import stopwords
import gc

# Ensure resources are downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
models_dir = os.path.join(script_dir, "saved_models")

# Ensure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

print("Loading dataset...")
target_csv = os.path.join(data_dir, "specific_medicine_data.csv")
print(f"Attempting to load: {os.path.abspath(target_csv)}")

if not os.path.exists(target_csv):
    print(f"❌ Error: Training file '{os.path.basename(target_csv)}' not found in '{os.path.abspath(data_dir)}'.")
    print("   Please run 'extract_specific_data.py' and 'clean_data.py' first to generate it.")
    exit(1)

csv_files = [target_csv]
print(f"✅ Found target training file: {os.path.basename(target_csv)}")

# ---------------------------------------------------------
# STEP 1: First Pass - Find all unique Medicine Names
# ---------------------------------------------------------
print("🔍 Pass 1: Scanning all files to find unique medicines...")
all_classes = set()
side_effects_map = {}

for file_path in csv_files:
    try:
        temp_df = pd.read_csv(file_path, low_memory=False)
        temp_df.columns = temp_df.columns.str.strip() # Clean column names
        
        # Normalize target column name (Handle 'Drug', 'Drug Name', etc.)
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
            # Add unique medicines to the set
            unique_meds = temp_df['Medicine Name'].dropna().unique()
            all_classes.update(unique_meds)
            
            if 'Side Effects' in temp_df.columns:
                # Create a mapping of Medicine Name -> Side Effects
                temp_map = temp_df[['Medicine Name', 'Side Effects']].dropna().drop_duplicates(subset=['Medicine Name'])
                side_effects_map.update(dict(zip(temp_map['Medicine Name'], temp_map['Side Effects'])))
        
        # Free memory immediately
        del temp_df

    except Exception as e:
        print(f"Error scanning {os.path.basename(file_path)}: {e}")

if not all_classes:
    print("❌ No medicine data found in any file.")
    exit(1)

all_classes = sorted(list(all_classes))
print(f"✅ Found {len(all_classes)} unique medicines to predict.")

# ---------------------------------------------------------
# STEP 2: Initialize Model for Incremental Learning
# ---------------------------------------------------------
# HashingVectorizer is used as it's stateless and memory-efficient.
# alternate_sign=False is recommended for models like MultinomialNB that expect non-negative features.
# n_features is set to 1000; this can be tuned. A larger value may reduce collisions but increase memory usage.
vectorizer = HashingVectorizer(stop_words='english', alternate_sign=False, n_features=1000)
model = MultinomialNB()

# ---------------------------------------------------------
# STEP 3: Second Pass - Train on each file sequentially
# ---------------------------------------------------------
print(f"\n🚀 Pass 2: Starting model training on {os.path.basename(target_csv)}...")

try:
    df = pd.read_csv(target_csv, low_memory=False)
    df.columns = df.columns.str.strip()

    # Apply same column normalization
    for col in ['Drug', 'Drug Name', 'drugName', 'Medicine', 'drug', 'Drug_Name', 'medicine', 'drug_name', 'name', 'Name']:
        if col in df.columns:
            df.rename(columns={col: 'Medicine Name'}, inplace=True)

    if 'Medicine Name' in df.columns:
        df = df.dropna(subset=['Medicine Name'])

        # Prepare Features
        exclude_cols = ['Medicine Name', 'Excellent Review %', 'Average Review %', 'Poor Review %']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']
        df[feature_cols] = df[feature_cols].fillna('')

        # Process in batches to save memory
        batch_size = 5000
        print(f"   Processing in batches of {batch_size} rows...")
        for start in range(0, len(df), batch_size):
            end = min(start + batch_size, len(df))
            df_batch = df.iloc[start:end].copy()

            df_batch['combined_text'] = df_batch[feature_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            df_batch['clean_text'] = df_batch['combined_text'].apply(clean_text)
            X_batch = vectorizer.transform(df_batch['clean_text'])
            y_batch = df_batch['Medicine Name']
            
            model.partial_fit(X_batch, y_batch, classes=all_classes)
            
            del df_batch, X_batch, y_batch
            gc.collect()
    else:
        print(f"⚠️ 'Medicine Name' column not found in {os.path.basename(target_csv)}. Skipping.")
except Exception as e:
    print(f"⚠️ Error training on {os.path.basename(target_csv)}: {e}")

# Save Models
print(f"Saving models to {models_dir}...")
pickle.dump(model, open(f"{models_dir}/drug_model.pkl", "wb"))
pickle.dump(vectorizer, open(f"{models_dir}/tfidf_vectorizer.pkl", "wb"))
pickle.dump(side_effects_map, open(f"{models_dir}/side_effects_map.pkl", "wb"))

print("✅ Models regenerated successfully!")