import os
import joblib

print("=" * 70)
print("STEP 29B - INSPECT IMPUTER FEATURE NAMES")
print("=" * 70)

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

IMPUTER_FILE = os.path.join(
    BASE_DIR,
    "models",
    "median_imputer.joblib"
)

print()
print("Loading imputer:")
print(IMPUTER_FILE)

imputer = joblib.load(IMPUTER_FILE)

print()
print("Imputer type:")
print(type(imputer))

print()
print("Original input features:")
print(len(imputer.feature_names_in_))

print()
print("Output feature count:")

try:
    output_count = imputer.get_feature_names_out().shape[0]
    print(output_count)

except Exception as e:
    print("Could not get output feature count:", e)

print()
print("=" * 70)
print("FIRST 30 OUTPUT FEATURE NAMES")
print("=" * 70)

feature_names = imputer.get_feature_names_out()

for i, name in enumerate(feature_names[:30]):
    print(f"{i:3d} -> {name}")

print()
print("=" * 70)
print("FEATURES AROUND ORIGINAL/MISSING-INDICATOR BOUNDARY")
print("=" * 70)

start = max(0, len(feature_names) - 20)

for i in range(start, len(feature_names)):
    print(f"{i:3d} -> {feature_names[i]}")

print()
print("=" * 70)
print("STEP 29B COMPLETED")
print("=" * 70)