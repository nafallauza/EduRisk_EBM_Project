import os
import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = "data/processed/student_data_processed.csv"
REPORT_DIR = "output/reports"
PLOT_DIR = "output/plots"


def save_dataset_summary(df):
    summary_path = f"{REPORT_DIR}/eda_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("RINGKASAN EDA DATASET\n")
        file.write("=====================\n\n")

        file.write("Ukuran Dataset\n")
        file.write("--------------\n")
        file.write(f"Jumlah data  : {df.shape[0]}\n")
        file.write(f"Jumlah kolom : {df.shape[1]}\n\n")

        file.write("Distribusi Target\n")
        file.write("-----------------\n")
        file.write(str(df["status_mahasiswa"].value_counts()))
        file.write("\n\n")

        file.write("Missing Value\n")
        file.write("-------------\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            file.write("Tidak ada missing value.\n")
        else:
            file.write(str(missing))
            file.write("\n")

        file.write("\nStatistik Deskriptif\n")
        file.write("--------------------\n")
        file.write(str(df.describe().T))

    print(f"Ringkasan EDA disimpan: {summary_path}")


def plot_target_distribution(df):
    target_counts = df["status_mahasiswa"].value_counts()

    plt.figure(figsize=(7, 5))
    target_counts.plot(kind="bar")
    plt.title("Distribusi Status Mahasiswa")
    plt.xlabel("Status Mahasiswa")
    plt.ylabel("Jumlah Data")
    plt.xticks(rotation=0)
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_1_distribusi_status_mahasiswa.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def plot_age_distribution(df):
    column = "Age at enrollment"

    if column not in df.columns:
        return

    plt.figure(figsize=(7, 5))
    df[column].plot(kind="hist", bins=20)
    plt.title("Distribusi Usia Saat Masuk")
    plt.xlabel("Usia")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_2_distribusi_usia.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def plot_semester_grade_distribution(df):
    columns = [
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)"
    ]

    for column in columns:
        if column not in df.columns:
            return

    plt.figure(figsize=(7, 5))
    df[columns].plot(kind="box")
    plt.title("Distribusi Nilai Semester 1 dan Semester 2")
    plt.ylabel("Nilai Akademik")
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_3_distribusi_nilai_semester.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.empty:
        return

    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.title("Heatmap Korelasi Antar Fitur Numerik")
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_4_heatmap_korelasi.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def plot_grade_by_status(df):
    grade_column = "Curricular units 2nd sem (grade)"

    if grade_column not in df.columns:
        return

    labels = []
    values = []

    for label, group in df.groupby("status_mahasiswa"):
        labels.append(label)
        values.append(group[grade_column].values)

    plt.figure(figsize=(7, 5))
    plt.boxplot(values, tick_labels=labels)
    plt.title("Nilai Semester 2 Berdasarkan Status Mahasiswa")
    plt.xlabel("Status Mahasiswa")
    plt.ylabel("Nilai Semester 2")
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_5_nilai_semester_2_vs_status.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def plot_selected_correlation_heatmap(df):
    selected_columns = [
        "Age at enrollment",
        "Admission grade",
        "Debtor",
        "Tuition fees up to date",
        "Scholarship holder",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)"
    ]

    available_columns = [
        column for column in selected_columns
        if column in df.columns
    ]

    selected_df = df[available_columns]

    if selected_df.empty:
        return

    corr = selected_df.corr()

    plt.figure(figsize=(9, 7))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
        fontsize=8
    )
    plt.yticks(
        range(len(corr.columns)),
        corr.columns,
        fontsize=8
    )
    plt.title("Heatmap Korelasi Fitur Akademik Utama")
    plt.tight_layout()

    output_path = f"{PLOT_DIR}/eda_6_heatmap_fitur_utama.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot disimpan: {output_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    if "status_mahasiswa" not in df.columns:
        raise ValueError(
            "Kolom status_mahasiswa tidak ditemukan. "
            "Jalankan prepare.py terlebih dahulu."
        )

    print("EDA dimulai.")
    print(f"Jumlah data  : {df.shape[0]}")
    print(f"Jumlah kolom : {df.shape[1]}")

    print("\nDistribusi target:")
    print(df["status_mahasiswa"].value_counts())

    print("\nMissing value:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("Tidak ada missing value.")
    else:
        print(missing)

    save_dataset_summary(df)
    plot_target_distribution(df)
    plot_age_distribution(df)
    plot_semester_grade_distribution(df)
    plot_correlation_heatmap(df)
    plot_selected_correlation_heatmap(df)
    plot_grade_by_status(df)

    print("\nEDA selesai.")
    print("File EDA tersimpan di output/reports dan output/plots.")


if __name__ == "__main__":
    main()
