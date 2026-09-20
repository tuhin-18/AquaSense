import os
import pandas as pd

print("=" * 70)
print("STEP 31C - INSPECT MODEL RESULT FILES")
print("=" * 70)

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"
OUTPUT_BASE = os.path.join(BASE_DIR, "ml_outputs")

files = {
    "Logistic Regression": os.path.join(
        OUTPUT_BASE,
        "baseline",
        "logistic_regression_results.csv"
    ),

    "Random Forest": os.path.join(
        OUTPUT_BASE,
        "random_forest",
        "random_forest_results.csv"
    ),

    "XGBoost": os.path.join(
        OUTPUT_BASE,
        "xgboost",
        "xgboost_results.csv"
    )
}

for name, path in files.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print("File:")
    print(path)

    df = pd.read_csv(path)

    print()
    print("Columns:")

    for i, column in enumerate(df.columns):
        print(f"{i}: {column}")

    print()
    print("Shape:", df.shape)

    print()
    print("First rows:")

    print(
        df.head().to_string(index=False)
    )

print()
print("=" * 70)
print("STEP 31C COMPLETED")
print("=" * 70)