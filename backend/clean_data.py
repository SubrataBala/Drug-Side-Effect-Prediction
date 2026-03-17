import pandas as pd
import os

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "../data")
target_file = os.path.join(data_dir, "specific_medicine_data.csv")
additional_effects_file = os.path.join(data_dir, "additional_side_effects.csv")

def run_cleaning():
    if not os.path.exists(target_file):
        print(f"❌ Error: File '{target_file}' not found.")
        print("   Please run 'extract_specific_data.py' first to generate it.")
        return

    print(f"🧹 Cleaning {target_file}...")
    
    try:
        df = pd.read_csv(target_file, low_memory=False)
        
        # Standardize Column Names
        df.columns = df.columns.str.strip()
        
        # Rename to 'Medicine Name'
        for col in ['Drug', 'Drug Name', 'drugName', 'Medicine', 'drug', 'Drug_Name', 'medicine', 'drug_name', 'name', 'Name', 'Medicine_Name']:
            if col in df.columns:
                df.rename(columns={col: 'Medicine Name'}, inplace=True)
                break
        
        # Rename to 'Side Effects'
        for col in ['Side Effects', 'SideEffects', 'sideEffects', 'side_effects', 'sideEffect', 'SideEffect', 'Side_Effect']:
            if col in df.columns:
                df.rename(columns={col: 'Side Effects'}, inplace=True)
                break
                
        # Rename to 'Substitute'
        for col in ['Substitute', 'substitute', 'Alternative', 'alternative', 'substitutes']:
            if col in df.columns:
                df.rename(columns={col: 'Substitute'}, inplace=True)
                break

        initial_count = len(df)
        
        # Remove duplicates
        df.drop_duplicates(inplace=True)
        
        # Remove rows with empty Medicine Name
        if 'Medicine Name' in df.columns:
            df.dropna(subset=['Medicine Name'], inplace=True)
            
        # Fill missing values for better UX
        if 'Side Effects' in df.columns:
            df['Side Effects'] = df['Side Effects'].fillna("Not Available")
        if 'Substitute' in df.columns:
            df['Substitute'] = df['Substitute'].fillna("Not Available")
    
        final_count = len(df)
        print(f"   Rows before merge: {final_count}")

        # --- NEW: Merge additional side effects ---
        if os.path.exists(additional_effects_file):
            print(f"🔄 Found '{os.path.basename(additional_effects_file)}'. Merging new side effect data...")
            try:
                effects_df = pd.read_csv(additional_effects_file)
                # Create a dictionary for quick lookups
                effects_map = dict(zip(effects_df['Medicine Name'].str.lower(), effects_df['Side Effects']))
                
                # Function to apply the update
                def update_effects(row):
                    if row['Side Effects'].lower() == 'not available':
                        return effects_map.get(row['Medicine Name'].lower(), row['Side Effects'])
                    return row['Side Effects']
                df['Side Effects'] = df.apply(update_effects, axis=1)
            except Exception as e:
                print(f"   ⚠️ Could not merge additional effects: {e}")

        print(f"   Rows after cleaning: {final_count} (Removed {initial_count - final_count} duplicates/empty rows)")

        print(f"💾 Saving cleaned data back to {target_file}...")
        df.to_csv(target_file, index=False)
        print("✅ Data cleaning completed successfully!")

    except Exception as e:
        print(f"❌ Error cleaning file: {e}")

if __name__ == "__main__":
    run_cleaning()