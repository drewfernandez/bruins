import csv
from collections import defaultdict
from baml import run_function  # Make sure BAML is installed in your Python environment

# --- Config ---
CSV_PATH = "your_input.csv"  # Replace with your CSV path
BAML_FUNCTION_NAME = "LabelAudioScene"

# --- Step 1: Read and group captions by file_name ---
grouped_captions = defaultdict(list)

with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = row["file_name"]
        caption = row["generated_captions"]
        grouped_captions[filename].append(caption)

# --- Step 2: Call BAML function for each file ---
results = []

for filename, captions in grouped_captions.items():
    print(f"Classifying {filename}...")

    # Run the BAML function
    result = run_function(BAML_FUNCTION_NAME, {
        "window_labels": captions
    })

    results.append({
        "file_name": filename,
        "category": result["category"],
        "confidence": result["confidence"],
        "rational": result["rational"]
    })

# --- Step 3: Print or save results ---
print("\nClassification Results:")
for r in results:
    print(r)
