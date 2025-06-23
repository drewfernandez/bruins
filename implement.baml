config:
  input: csv
  path: "<INSERT_PATH_TO_CSV_HERE>"  # Replace with your CSV path
  output: categorized_files.csv

vars:
  model_name: google/gemma-3-4b  # Configurable
  lmstudio_api: http://localhost:1234/v1/chat

  categories: [
    "Animals",
    "Natural soundscapes & water sounds",
    "Human non-speech sounds",
    "Interior/domestic sounds",
    "Exterior/urban noises"
  ]

# Group captions by filename before sending to the LLM
preprocess:
  - name: group_captions_by_file
    run: |
      from collections import defaultdict

      grouped = defaultdict(list)
      for row in data:
          grouped[row["filename"]].append(row["generated_captions"])

      grouped_data = []
      for filename, captions in grouped.items():
          joined_captions = "\n".join(f"- {caption}" for caption in captions)
          grouped_data.append({
              "filename": filename,
              "joined_captions": joined_captions
          })

      return grouped_data

tasks:
  - name: classify_grouped_captions
    for_each: row in group_captions_by_file
    input: |
      You are an expert sound analyst. A set of audio captions (from different segments of the same file) is provided below.

      Captions:
      {{ row.joined_captions }}

      Based on the overall content of these captions, classify the **entire audio file** into one of the following categories:
      - Animals
      - Natural soundscapes & water sounds
      - Human non-speech sounds
      - Interior/domestic sounds
      - Exterior/urban noises

      Only respond with the exact category name from the list.
    model:
      provider: openai
      model: "{{ model_name }}"
      base_url: "{{ lmstudio_api }}"
    output: row.category
