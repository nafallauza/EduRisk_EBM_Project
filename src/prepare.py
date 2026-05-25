import os
import pandas as pd


RAW_DATA = "data/raw/students_dropout_academic_success.csv"
PROCESSED_DIR = "data/processed"
PROCESSED_DATA = f"{PROCESSED_DIR}/student_data_processed.csv"
DATASET_INFO = f"{PROCESSED_DIR}/dataset_info.txt"
TARGET_DIST = f"{PROCESSED_DIR}/target_distribution.csv"


def read_dataset(path):
    try:
        data = pd.read_csv(path)
    except Exception:
        data = pd.read_csv(path, sep=";")

    if data.shape[1] == 1:
        data = pd.read_csv(path, sep=";")

    data.columns = data.columns.str.strip()
    return data


def save_dataset_info(data, target_column):
    with open(DATASET_INFO, "w", encoding="utf-8") as file:
        file.write("INFORMASI DATASET\n")
        file.write("=================\n")
        file.write(f"Jumlah data  : {data.shape[0]}\n")
        file.write(f"Jumlah kolom : {data.shape[1]}\n\n")

        file.write("DAFTAR KOLOM\n")
        file.write("------------\n")
        for col in data.columns:
            file.write(f"- {col}\n")

        file.write("\nDISTRIBUSI TARGET\n")
        file.write("-----------------\n")
        file.write(str(data[target_column].value_counts()))


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = read_dataset(RAW_DATA)
    target_column = "target"

    if target_column not in df.columns:
        print("Kolom target tidak ditemukan.")
        print("Kolom yang tersedia:")
        print(df.columns.tolist())
        return

    df = df[df[target_column].isin(["Graduate", "Dropout"])].copy()

    df["status_mahasiswa"] = df[target_column].map({
        "Graduate": "Lulus",
        "Dropout": "Keluar"
    })

    df = df.drop(columns=[target_column])

    print("Dataset berhasil dibaca.")
    print(f"Jumlah data  : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")

    print("\nDistribusi target:")
    print(df["status_mahasiswa"].value_counts())

    missing_value = df.isnull().sum()
    missing_value = missing_value[missing_value > 0]

    print("\nMissing value:")
    if missing_value.empty:
        print("Tidak ada missing value.")
    else:
        print(missing_value)

    df.to_csv(PROCESSED_DATA, index=False)
    df["status_mahasiswa"].value_counts().to_csv(TARGET_DIST)
    save_dataset_info(df, "status_mahasiswa")

    print("\nData berhasil disimpan.")
    print(f"- {PROCESSED_DATA}")
    print(f"- {DATASET_INFO}")
    print(f"- {TARGET_DIST}")


if __name__ == "__main__":
    main()