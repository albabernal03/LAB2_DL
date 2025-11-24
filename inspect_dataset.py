import pandas as pd
import os

def inspect_pkl(path):
    print(f"\n=== {path} ===")
    df = pd.read_pickle(path)

    # Si no es un DataFrame, lo ignoramos
    if not isinstance(df, pd.DataFrame):
        print("❌ No es un DataFrame, se ignora.")
        print("="*50)
        return

    print(df.head())
    print("\nShape:", df.shape)
    print("Columns:", list(df.columns))
    print("="*50)


BASE = "."

for root, dirs, files in os.walk(BASE):
    # ❗ Evitar inspeccionar venv
    if "venv" in root:
        continue

    for file in files:
        if file.endswith(".pkl"):
            full_path = os.path.join(root, file)
            inspect_pkl(full_path)
