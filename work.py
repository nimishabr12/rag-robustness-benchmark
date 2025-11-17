from datasets import load_dataset
import pandas as pd
import os

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Load dataset
dataset = load_dataset("ms_marco", "v2.1")

# Pick the VALIDATION split, then sample 200
sample = dataset["validation"].shuffle(seed=42).select(range(200))

# Look at first example
print("First example:")
print(sample[0])
print("\nColumns:", sample.column_names)

# Save to CSV for easy viewing
df = pd.DataFrame(sample)
df.to_csv("data/raw/ms_marco_sample.csv", index=False)
print(f"\n✓ Saved {len(sample)} examples to data/raw/ms_marco_sample.csv")