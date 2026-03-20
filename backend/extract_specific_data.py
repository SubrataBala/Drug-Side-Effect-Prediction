import pandas as pd
import os
import json

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
config_file = os.path.join(script_dir, "config.json")
output_file = os.path.join(data_dir, "specific_medicine_data.csv")

def load_config():
    """Loads the configuration from the JSON file."""
    if not os.path.exists(config_file):
        print(f"❌ Error: Config file '{config_file}' not found.")
        return None, None, 1500, None # Default rows
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return (config.get("target_medicines", []), 
                config.get("substitutes_mapping", {}), 
                config.get("target_rows", 1500),
                config.get("source_csv_filename"))
    except Exception as e:
        print(f"❌ Error reading config file: {e}")
        return None, None, None, None

def extract_and_save():
    # Load configuration
    target_medicines, substitutes_mapping, target_rows, source_csv = load_config()
    input_file = os.path.join(data_dir, source_csv) if source_csv else None

    # Check if input file exists
    if not input_file or not os.path.exists(input_file):
        print(f"❌ Error: Source CSV file not found at '{input_file}'.")
        print("   Please ensure the file name in 'backend/config.json' is correct and the file is in the 'data/' directory.")
        return

    if not target_medicines:
        print("❌ No target medicines found in config. Exiting.")
        return

    print(f"📖 Reading '{input_file}'...")
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # --- NEW: Pre-process the specific format of final_ultra_clean_dataset.csv ---
    print("⚙️ Pre-processing 'final_ultra_clean_dataset.csv' format...")

    # 1. Combine side effect columns
    side_effect_cols = [f'sideEffect{i}' for i in range(5)]
    # Ensure columns exist before trying to combine
    existing_se_cols = [col for col in side_effect_cols if col in df.columns]
    if existing_se_cols:
        # Convert to string, fill NaNs with empty string, and join
        df['Side Effects'] = df[existing_se_cols].astype(str).fillna('').agg(', '.join, axis=1)
        # Clean up the joined string: remove trailing commas/spaces, 'nan' strings
        df['Side Effects'] = df['Side Effects'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')
        print("   ✅ Combined multiple 'sideEffect' columns into a single 'Side Effects' column.")

    # 2. Combine 'InteractsWith' columns for later use
    interacts_with_cols = [f'InteractsWith{i}' for i in range(3)]
    existing_iw_cols = [col for col in interacts_with_cols if col in df.columns]
    if existing_iw_cols:
        df['Interacts With'] = df[existing_iw_cols].astype(str).fillna('').agg(', '.join, axis=1)
        df['Interacts With'] = df['Interacts With'].str.replace(r'\b(nan|None)\b', '', regex=True).str.replace(r'(, )+', ', ', regex=True).str.strip(', ')
        print("   ✅ Combined multiple 'InteractsWith' columns into a single 'Interacts With' column.")

    # Normalize column names in the CSV
    df.columns = df.columns.str.strip()
    
    # Identify the Medicine Name column
    med_col = None
    for col in ['Medicine Name', 'Drug', 'Drug Name', 'drugName', 'Medicine']:
        if col in df.columns:
            med_col = col
            break
    
    if not med_col:
        print("❌ Error: Could not find a 'Medicine Name' column in the source CSV.")
        return

    print(f"🔍 Searching for {len(target_medicines)} specific medicines...")

    # Create a lowercase map for case-insensitive matching
    target_meds_lower = {m.lower(): m for m in target_medicines}
    
    # Filter the dataframe
    # We use a lambda to check if the medicine name (lowercased) is in our target list
    mask = df[med_col].astype(str).str.lower().isin(target_meds_lower.keys())
    filtered_df = df[mask].copy()

    # Check which medicines were found
    found_meds = filtered_df[med_col].astype(str).str.lower().unique()
    found_names = [target_meds_lower[m] for m in found_meds if m in target_meds_lower]
    
    print(f"✅ Found {len(found_names)} unique medicines from your list in the dataset.")
    
    missing = set(target_medicines) - set(found_names)
    if missing:
        print(f"⚠️ The following medicines were NOT found in the source data:\n   {', '.join(missing)}")

    if filtered_df.empty:
        print("❌ No matching records found. Exiting.")
        return

    # Update Substitute column with the mapping
    print("🔄 Updating Substitute information (verifying availability in source data)...")
    if 'Substitute' not in filtered_df.columns:
        filtered_df['Substitute'] = None

    # Get all available medicines from the FULL source dataframe
    all_source_meds = set(df[med_col].astype(str).str.lower().unique())

    for med, subs in substitutes_mapping.items():
        # Split provided substitutes
        potential_subs = [s.strip() for s in subs.split(',')]
        
        # Filter: Keep only substitutes that exist in the source dataset
        confirmed_subs = [s for s in potential_subs if s.lower() in all_source_meds]
        
        if confirmed_subs:
            valid_subs_str = ", ".join(confirmed_subs)
            # Case-insensitive match for medicine name
            mask = filtered_df[med_col].astype(str).str.lower() == med.lower()
            if mask.any():
                filtered_df.loc[mask, 'Substitute'] = valid_subs_str

    print(f"📊 Total matching rows found in source: {len(filtered_df)}")

    # Sample rows (with replacement if we have fewer rows than target, without if we have more)
    print(f"🎯 Sampling to get {target_rows} rows for the final dataset...")
    replace_flag = len(filtered_df) < target_rows
    final_df = filtered_df.sample(n=target_rows, replace=replace_flag, random_state=42)

    # Save to new CSV
    print(f"💾 Saving extracted data to '{output_file}'...")
    final_df.to_csv(output_file, index=False)
    print(f"✅ Done! Created '{os.path.basename(output_file)}' with {len(final_df)} rows.")

if __name__ == "__main__":
    extract_and_save()