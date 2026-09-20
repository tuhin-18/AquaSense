import os

BASE_DIR = r"C:\Users\tusha\Documents\AquaSense"

print("=" * 70)
print("SEARCHING FOR YEARLY TURBIDITY SUMMARY")
print("=" * 70)

for root, dirs, files in os.walk(BASE_DIR):

    for file in files:

        if "yearly" in file.lower() and "turbidity" in file.lower() and file.lower().endswith(".csv"):

            full_path = os.path.join(root, file)

            print()
            print("FOUND:")
            print(full_path)

print()
print("=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)