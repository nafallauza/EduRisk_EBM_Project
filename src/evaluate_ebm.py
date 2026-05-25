import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_PATH = "data/processed/student_data_processed.csv"

MODEL_PATH = "output/models/ebm_model.pkl"
ENCODER_PATH = "output/models/label_encoder.pkl"
FEATURE_PATH = "output/models/feature_columns.pkl"

REPORT_DIR = "output/reports"
PLOT_DIR = "output/plots"

FINAL_REPORT_PATH = f"{REPORT_DIR}/evaluation_report.txt"
CONFUSION_MATRIX_PATH = f"{PLOT_DIR}/confusion_matrix_ebm.png"
FEATURE_IMPORTANCE_PATH = f"{PLOT_DIR}/feature_importance_ebm.png"
FEATURE_IMPORTANCE_CSV = f"{REPORT_DIR}/feature_importance_ebm.csv"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    target_column = "status_mahasiswa"

    if target_column not in df.columns:
        raise ValueError("Kolom target 'status_mahasiswa' tidak ditemukan.")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

    X = df[feature_columns]
    y = df[target_column]

    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    print("HASIL EVALUASI MODEL EBM")
    print("========================")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print(report)

    with open(FINAL_REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("HASIL EVALUASI MODEL EBM\n")
        file.write("========================\n")
        file.write("Target: Lulus vs Keluar\n")
        file.write(f"Accuracy: {accuracy:.4f}\n\n")
        file.write(report)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot()
    plt.title("Confusion Matrix - EBM")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=300)
    plt.close()

    global_explanation = model.explain_global()
    global_data = global_explanation.data()

    feature_importance = pd.DataFrame({
        "Feature": global_data["names"],
        "Importance": global_data["scores"]
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    feature_importance.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    top_features = feature_importance.head(10).sort_values(
        by="Importance",
        ascending=True
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["Feature"], top_features["Importance"])
    plt.title("Top 10 Fitur Paling Berpengaruh - EBM")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=300)
    plt.close()

    print("\nFile evaluasi berhasil disimpan:")
    print(f"- {FINAL_REPORT_PATH}")
    print(f"- {CONFUSION_MATRIX_PATH}")
    print(f"- {FEATURE_IMPORTANCE_PATH}")
    print(f"- {FEATURE_IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()