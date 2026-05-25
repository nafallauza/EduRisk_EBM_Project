import os
import warnings
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from interpret.glassbox import ExplainableBoostingClassifier


warnings.filterwarnings("ignore")


DATA_PATH = "data/processed/student_data_processed.csv"

MODEL_DIR = "output/models"
REPORT_DIR = "output/reports"

MODEL_PATH = f"{MODEL_DIR}/ebm_model.pkl"
ENCODER_PATH = f"{MODEL_DIR}/label_encoder.pkl"
FEATURE_PATH = f"{MODEL_DIR}/feature_columns.pkl"
METRICS_PATH = f"{REPORT_DIR}/training_metrics.txt"
REPORT_PATH = f"{REPORT_DIR}/classification_report_train.txt"


PARAMETER_LIST = [
    {
        "name": "ebm_interaction_5",
        "interactions": 5
    },
    {
        "name": "ebm_interaction_10",
        "interactions": 10
    },
    {
        "name": "ebm_interaction_15",
        "interactions": 15
    },
    {
        "name": "ebm_interaction_20",
        "interactions": 20
    },
    {
        "name": "ebm_interaction_30",
        "interactions": 30
    }
]


def build_model(params):
    return ExplainableBoostingClassifier(
        random_state=42,
        interactions=params["interactions"]
    )


def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return accuracy, precision, recall, f1


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    target_column = "status_mahasiswa"

    if target_column not in df.columns:
        raise ValueError("Kolom target 'status_mahasiswa' tidak ditemukan. Jalankan ulang prepare.py versi 2 kelas.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Mapping label:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"{index} = {label}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    mlflow.set_experiment("edurisk_ebm_binary_experiment")

    best_model = None
    best_params = None
    best_prediction = None
    best_result = None

    for number, params in enumerate(PARAMETER_LIST, start=1):
        print(f"\nPercobaan {number}: {params['name']}")

        model = build_model(params)

        with mlflow.start_run(run_name=params["name"]):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)

            mlflow.log_param("model", "ExplainableBoostingClassifier")
            mlflow.log_param("target", "Lulus_vs_Keluar")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("interactions", params["interactions"])

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision_weighted", precision)
            mlflow.log_metric("recall_weighted", recall)
            mlflow.log_metric("f1_weighted", f1)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")

        if best_result is None or accuracy > best_result["accuracy"]:
            best_model = model
            best_params = params
            best_prediction = y_pred
            best_result = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(X.columns.tolist(), FEATURE_PATH)

    report = classification_report(
        y_test,
        best_prediction,
        target_names=label_encoder.classes_,
        zero_division=0
    )

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        file.write("HASIL TRAINING MODEL EBM\n")
        file.write("========================\n")
        file.write("Target        : Lulus vs Keluar\n")
        file.write(f"Model terbaik : {best_params['name']}\n")
        file.write(f"Parameter     : {best_params}\n\n")
        file.write(f"Accuracy      : {best_result['accuracy']:.4f}\n")
        file.write(f"Precision     : {best_result['precision']:.4f}\n")
        file.write(f"Recall        : {best_result['recall']:.4f}\n")
        file.write(f"F1-score      : {best_result['f1']:.4f}\n")

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("CLASSIFICATION REPORT MODEL EBM\n")
        file.write("================================\n\n")
        file.write(report)

    print("\nModel terbaik berhasil disimpan.")
    print(f"Model terbaik : {best_params['name']}")
    print(f"Accuracy      : {best_result['accuracy']:.4f}")
    print(f"Precision     : {best_result['precision']:.4f}")
    print(f"Recall        : {best_result['recall']:.4f}")
    print(f"F1-score      : {best_result['f1']:.4f}")

    print("\nClassification report:")
    print(report)

    print("\nFile tersimpan:")
    print(f"- {MODEL_PATH}")
    print(f"- {ENCODER_PATH}")
    print(f"- {FEATURE_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {REPORT_PATH}")


if __name__ == "__main__":
    main()