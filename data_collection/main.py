# main.py
import subprocess
import pandas as pd
import os

# Paths to individual data collector scripts
SCRIPTS = ["collect_reddit.py", "collect_facebook.py", "collect_instagram.py"]

# Output CSVs from each script
OUTPUT_FILES = ["reddit_hoax_data.csv", "facebook_hoax_data.csv", "instagram_hoax_data.csv"]

# Final combined CSV
FINAL_OUTPUT = "hoax_social_data.csv"

def run_collectors():
    for script in SCRIPTS:
        print(f"[*] Running {script} ...")
        try:
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"[!] Error running {script}: {e}")

def combine_csvs():
    files = ["reddit_hoax_data.csv", "facebook_hoax_data.csv", "instagram_hoax_data.csv"]
    combined = []

    for file in files:
        print(f"[+] Loading data from {file}")
        try:
            if os.path.getsize(file) == 0:
                print(f"[!] Skipping {file} — file is empty.")
                continue
            df = pd.read_csv(file)
            if df.empty or df.columns.size == 0:
                print(f"[!] Skipping {file} — no data.")
                continue
            combined.append(df)
        except Exception as e:
            print(f"[!] Error reading {file}: {e}")

    if combined:
        final_df = pd.concat(combined, ignore_index=True)
        final_df.to_csv("combined_hoax_data.csv", index=False)
        print("[✓] Combined data saved to 'combined_hoax_data.csv'")
    else:
        print("[!] No data collected to combine.")


if __name__ == "__main__":
    run_collectors()
    combine_csvs()
