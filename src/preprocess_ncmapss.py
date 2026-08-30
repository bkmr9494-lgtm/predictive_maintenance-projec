import os
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

RAW_FILE = "../data/raw/N-CMAPSS_DS02-006.h5"
OUTPUT_DIR = "../data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# 2. Load the N-CMAPSS development data
# --------------------------------------------------

print("Opening N-CMAPSS dataset...")

with h5py.File(RAW_FILE, "r") as f:

    A = f["A_dev"][:]
    W = f["W_dev"][:]
    X_s = f["X_s_dev"][:]
    X_v = f["X_v_dev"][:]
    Y = f["Y_dev"][:].flatten()


print("Data loaded successfully!")

print("A shape:", A.shape)
print("W shape:", W.shape)
print("X_s shape:", X_s.shape)
print("X_v shape:", X_v.shape)
print("Y shape:", Y.shape)


# --------------------------------------------------
# 3. Create column names
# --------------------------------------------------

A_columns = [
    "A_1",
    "A_2",
    "A_3",
    "A_4"
]

W_columns = [
    "W_1",
    "W_2",
    "W_3",
    "W_4"
]

Xs_columns = [
    f"sensor_{i+1}"
    for i in range(14)
]

Xv_columns = [
    f"virtual_sensor_{i+1}"
    for i in range(14)
]


# --------------------------------------------------
# 4. Create DataFrames
# --------------------------------------------------

df_A = pd.DataFrame(A, columns=A_columns)

df_W = pd.DataFrame(W, columns=W_columns)

df_Xs = pd.DataFrame(X_s, columns=Xs_columns)

df_Xv = pd.DataFrame(X_v, columns=Xv_columns)

df_Y = pd.DataFrame({
    "RUL": Y
})


# --------------------------------------------------
# 5. Combine the data
# --------------------------------------------------

processed_data = pd.concat(
    [
        df_A,
        df_W,
        df_Xs,
        df_Xv,
        df_Y
    ],
    axis=1
)


print("\nProcessed dataset created!")

print("Rows:", len(processed_data))
print("Columns:", len(processed_data.columns))

print("\nFirst 5 rows:")
print(processed_data.head())


# --------------------------------------------------
# 6. Save raw combined processed dataset
# --------------------------------------------------

raw_processed_file = os.path.join(
    OUTPUT_DIR,
    "processed_data.csv"
)

processed_data.to_csv(
    raw_processed_file,
    index=False
)

print("\nSaved:")
print(raw_processed_file)