import os
import random
import argparse
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

# 📊 High-level ESC-50 category mapping
CATEGORY_MAP = {
    0: ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"],  # Animals
    1: ["rain", "sea_waves", "crackling_fire", "crickets", "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush", "thunderstorm"],  # Natural
    2: ["crying_baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing_teeth", "snoring", "drinking_sipping"],  # Human
    3: ["door_wood_knock", "mouse_click", "keyboard_typing", "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner", "clock_alarm", "clock_tick", "glass_breaking"],  # Interior
    4: ["helicopter", "chainsaw", "siren", "car_horn", "engine", "train", "church_bells", "airplane", "fireworks", "hand_saw"]  # Urban
}

def create_test_dataset(esc_csv_path, audio_dir, output_dir, output_csv, num_samples=100, clips_per_sample=12, sample_rate=44100, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(esc_csv_path)
    
    results = []

    for i in tqdm(range(num_samples), desc="Generating test samples"):
        # Randomly select a category index (0–4)
        category_index = random.randint(0, 4)
        valid_labels = CATEGORY_MAP[category_index]
        
        # Subset ESC-50 metadata
        subset = df[df['category'].isin(valid_labels)]
        
        # Sample 12 unique files
        sample_rows = subset.sample(n=clips_per_sample, random_state=seed + i)
        file_paths = [os.path.join(audio_dir, row["filename"]) for _, row in sample_rows.iterrows()]
        
        # Load and concatenate
        audio_segments = []
        for path in file_paths:
            y, sr = librosa.load(path, sr=sample_rate)
            audio_segments.append(y)
        
        combined = np.concatenate(audio_segments)
        
        # Normalize to avoid clipping
        combined = combined / np.max(np.abs(combined))
        
        # Save audio file
        test_filename = f"test_sample_{i:03d}.wav"
        output_path = os.path.join(output_dir, test_filename)
        sf.write(output_path, combined, sample_rate)
        
        # Add to metadata
        result_row = {
            "file_name": test_filename,
            "category": category_index,
        }
        for j, esc_file in enumerate(sample_rows["filename"].tolist()):
            result_row[f"esc_file_{j+1}"] = esc_file
        results.append(result_row)
    
    # Write metadata CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    print(f"\n✅ Generated {num_samples} test samples.")
    print(f"📁 Audio files saved in: {output_dir}")
    print(f"📄 Metadata CSV saved at: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ESC-50 Test Dataset")
    parser.add_argument("--esc_csv", type=str, required=True, help="Path to esc50.csv")
    parser.add_argument("--audio_dir", type=str, required=True, help="Path to ESC-50 audio directory")
    parser.add_argument("--output_dir", type=str, default="test_dataset_audio", help="Directory to save generated audio")
    parser.add_argument("--output_csv", type=str, default="test_dataset_metadata.csv", help="CSV file to save test metadata")
    parser.add_argument("--samples", type=int, default=100, help="Number of test files to generate")
    parser.add_argument("--clips_per_sample", type=int, default=12, help="Number of ESC-50 clips per test file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    create_test_dataset(
        esc_csv_path=args.esc_csv,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        output_csv=args.output_csv,
        num_samples=args.samples,
        clips_per_sample=args.clips_per_sample,
        seed=args.seed
    )
