import pandas as pd
import os

# ============================================================
# AQUASENSE - DATASET INSPECTION
# ============================================================

# Project folder
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "dataset",
    "SEAN_FQ_Q_2011-2024_KLGO_TA.csv"
)

print("=" * 70)
print("AQUASENSE DATASET INSPECTION")
print("=" * 70)

print("\nDataset path:")
print(DATASET_PATH)

# Check file
if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset file not found.")
    print("Please check that the CSV is inside the dataset folder.")
    exit()

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("\nDataset loaded successfully.")

# ------------------------------------------------------------
# Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# ------------------------------------------------------------
# Column names
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(column)

# ------------------------------------------------------------
# First 5 rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(df.head())

# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(df.dtypes)

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

# ------------------------------------------------------------
# Basic statistics
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NUMERICAL STATISTICS")
print("=" * 70)

print(df.describe())

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)