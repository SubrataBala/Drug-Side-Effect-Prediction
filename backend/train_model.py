import pandas as pd
import pickle
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
import os
import gc
import sys

# Add the project root to the Python path to allow for absolute imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.utils import clean_text

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
models_dir = os.path.join(script_dir, "saved_models")

# Ensure models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

print("Loading dataset...")
target_csv = os.path.join(data_dir, "final_real_fixed_rows.csv")
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
print("🔍 Scanning data to find unique medicines and side effects...")
try:
    # Since clean_data.py has run, we can assume the CSV is well-formed.
    # We only need to read it once to get classes and build the side_effects_map.
    df_for_scan = pd.read_csv(target_csv, low_memory=False)
    
    # --- Pre-process the new CSV format ---
    # Combine side effect columns
    side_effect_cols = [f'sideEffect{i}' for i in range(5)]
    existing_se_cols = [col for col in side_effect_cols if col in df_for_scan.columns]
    if existing_se_cols:
        df_for_scan['Side Effects'] = df_for_scan[existing_se_cols].astype(str).fillna('').agg(', '.join, axis=1)
        df_for_scan['Side Effects'] = df_for_scan['Side Effects'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')
        print("✅ Combined multiple 'sideEffect' columns into a single 'Side Effects' column.")

    # Combine 'InteractsWith' columns
    interacts_with_cols = [f'InteractsWith{i}' for i in range(3)]
    existing_iw_cols = [col for col in interacts_with_cols if col in df_for_scan.columns]
    if existing_iw_cols:
        df_for_scan['Interacts With'] = df_for_scan[existing_iw_cols].astype(str).fillna('').agg(', '.join, axis=1)
        df_for_scan['Interacts With'] = df_for_scan['Interacts With'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')
        print("✅ Combined multiple 'InteractsWith' columns into a single 'Interacts With' column.")
    # --- End Pre-processing ---

    all_classes = sorted(list(df_for_scan['Medicine Name'].dropna().unique()))
    
    # Create a map of medicine names to their side effects for quick lookup later.
    # This is more efficient than searching the dataframe in the app.
    side_effects_map = dict(zip(df_for_scan['Medicine Name'], df_for_scan['Side Effects']))
    
    # Create a map for interactions for quick lookup in the app
    interaction_map = {}
    interaction_cols = ['Medicine Name', 'InteractionEffect', 'Interacts With']
    if all(col in df_for_scan.columns for col in interaction_cols):
        interaction_df = df_for_scan[interaction_cols].dropna(subset=['Medicine Name'])
        for _, row in interaction_df.iterrows():
            med_name = row['Medicine Name']
            # Store interaction effect and a cleaned list of interacting drugs
            interaction_map[med_name] = {
                'effect': row['InteractionEffect'],
                'interacts_with': [drug.strip() for drug in str(row['Interacts With']).split(',') if drug.strip()]
            }
        print("✅ Created interaction map for quick lookup.")
    else:
        print("⚠️ Interaction-related columns not found. Interaction map will be empty.")

    del df_for_scan # Free up memory
except Exception as e:
    print(f"❌ Error during initial scan: {e}")
    exit(1)

if not all_classes:
    print("❌ No medicine data found in any file.")
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
    if 'Medicine Name' in df.columns:
        df = df.dropna(subset=['Medicine Name'])

        # --- Pre-process the new CSV format ---
        # This is repeated from Pass 1 to ensure the dataframe for training is correct.
        # Combine side effect columns
        side_effect_cols = [f'sideEffect{i}' for i in range(5)]
        existing_se_cols = [col for col in side_effect_cols if col in df.columns]
        if existing_se_cols:
            df['Side Effects'] = df[existing_se_cols].astype(str).fillna('').agg(', '.join, axis=1)
            df['Side Effects'] = df['Side Effects'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')

        # Combine 'InteractsWith' columns
        interacts_with_cols = [f'InteractsWith{i}' for i in range(3)]
        existing_iw_cols = [col for col in interacts_with_cols if col in df.columns]
        if existing_iw_cols:
            df['Interacts With'] = df[existing_iw_cols].astype(str).fillna('').agg(', '.join, axis=1)
            df['Interacts With'] = df['Interacts With'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')
        # --- End Pre-processing ---

        # Prepare Features
        exclude_cols = ['Medicine Name', 'Excellent Review %', 'Average Review %', 'Poor Review %']
        feature_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype == 'object']
        df[feature_cols] = df[feature_cols].fillna('')

        # Process in batches to save memory
        batch_size = 2000
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
pickle.dump(vectorizer, open(f"{models_dir}/hashing_vectorizer.pkl", "wb"))
pickle.dump(side_effects_map, open(f"{models_dir}/side_effects_map.pkl", "wb"))
pickle.dump(interaction_map, open(f"{models_dir}/interaction_map.pkl", "wb"))

print("✅ Models regenerated successfully!")