config:
  input: csv
  path: "<INSERT_PATH_TO_CSV_HERE>"  # Replace this with your actual CSV file path
  output: categorized_captions.csv

vars:
  model_name: google/gemma-3-4b  # You can change this to any model loaded in LM Studio
  lmstudio_api: http://localhost:1234/v1/chat

  # Define your allowed categories
  categories: [
    "Animals",
    "Natural soundscapes & water sounds",
    "Human non-speech sounds",
    "Interior/domestic sounds",
    "Exterior/urban noises"
  ]

tasks:
  - name: categorize_caption
    for_each: row in data
    input: |
      You are an expert sound analyst. Given the following caption of an audio clip:
      
      "{{ row.generated_captions }}"

      Please classify the sound into one of the following categories:
      - Animals
      - Natural soundscapes & water sounds
      - Human non-speech sounds
      - Interior/domestic sounds
      - Exterior/urban noises

      Only respond with the exact category name from the list above.
    model:
      provider: openai
      model: "{{ model_name }}"
      base_url: "{{ lmstudio_api }}"
    output: row.category
