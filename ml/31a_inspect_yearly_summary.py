import os
import pandas as pd

print("=" * 70)
print("STEP 31A - INSPECT YEARLY TURBIDITY SUMMARY")
print("=" * 70)

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

FILE = os.path.join(
    BASE_DIR,
    "ml_outputs",
    "temporal_shift_analysis",
    "yearly_turbidity_summary.csv"
)

print()
print("Loading:")
print(FILE)

df = pd.read_csv(FILE)

print()
print("=" * 70)
print("COLUMNS")
print("=" * 70)

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")

print()
print("=" * 70)
print("SHAPE")
print("=" * 70)

print(df.shape)

print()
print("=" * 70)
print("FIRST 10 ROWS")
print("=" * 70)

print(
    df.head(10).to_string(index=False)
)

print()
print("=" * 70)
print("STEP 31A COMPLETED")
print("=" * 70)