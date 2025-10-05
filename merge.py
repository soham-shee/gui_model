import os
import pandas as pd

# 🔁 Change this to the path of your main folder
main_folder = r"/Users/macm1/Downloads/Test Dataset"

# Store all dataframes
dfs = []

# Traverse all subdirectories
for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(root, file)
            try:
                df = pd.read_excel(file_path)

                # Ensure required columns exist
                if 'Date N Time' in df.columns and 'M1_PWR' in df.columns and 'M2_PWR' in df.columns and 'M3_PWR' in df.columns:
                    df = df[['Date N Time', 'M1_PWR', 'M2_PWR', 'M3_PWR']]
                    dfs.append(df)
                    print(f"✅ Loaded: {file_path}")
                else:
                    print(f"⚠️ Skipping (missing columns): {file_path}")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

# Combine all valid DataFrames
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)

    # Convert date column to datetime
    combined_df['Date N Time'] = pd.to_datetime(combined_df['Date N Time'], errors='coerce')

    # Drop rows where datetime conversion failed
    combined_df.dropna(subset=['Date N Time'], inplace=True)

    # Set datetime as index and sort
    combined_df.set_index('Date N Time', inplace=True)
    combined_df.sort_index(inplace=True)

    # Save the combined DataFrame to Excel
    output_path = os.path.join(main_folder, "combined_data.xlsx")
    combined_df.to_excel(output_path)

    print(f"\n🎉 Successfully combined {len(dfs)} files.")
    print(f"📁 Combined file saved at: {output_path}")
else:
    print("⚠️ No valid Excel files found.")
