import os
import pandas as pd
import joblib
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_url"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

REGISTERED_MODEL_NAME = "student_dropout"

df = pd.read_csv(
    config["data"]["path"],
    delimiter=config["data"]["delimiter"]
)

df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Daytime/evening attendance\t": "Daytime/evening attendance"})
df = df.drop_duplicates()

df["Target"] = df["Target"].str.strip()
df["Label"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

features = {
    "Marital status": "Status_Pernikahan",
    "Nacionality": "Kewarganegaraan",
    "Gender": "Jenis_Kelamin",
    "Age at enrollment": "Usia_Saat_Daftar",
    "Application mode": "Jalur_Pendaftaran",
    "Application order": "Urutan_Pilihan",
    "Course": "Program_Studi",
    "Daytime/evening attendance": "Kelas_Siang_Malam",
    "Previous qualification": "Pendidikan_Sebelumnya",
    "Previous qualification (grade)": "Nilai_Pendidikan_Sebelumnya",
    "Admission grade": "Nilai_Masuk",
    "Mother's qualification": "Pendidikan_Ibu",
    "Father's qualification": "Pendidikan_Ayah",
    "Mother's occupation": "Pekerjaan_Ibu",
    "Father's occupation": "Pekerjaan_Ayah",
    "Displaced": "Pindahan",
    "Educational special needs": "Berkebutuhan_Khusus",
    "Debtor": "Punya_Tunggakan",
    "Tuition fees up to date": "SPP_Lunas",
    "Scholarship holder": "Penerima_Beasiswa",
    "International": "Mahasiswa_Internasional",
    "Curricular units 1st sem (credited)": "SKS_Sem1_Diakui",
    "Curricular units 1st sem (enrolled)": "SKS_Sem1_Diambil",
    "Curricular units 1st sem (evaluations)": "SKS_Sem1_Evaluasi",
    "Curricular units 1st sem (approved)": "SKS_Sem1_Lulus",
    "Curricular units 1st sem (grade)": "Nilai_Sem1",
    "Curricular units 1st sem (without evaluations)": "SKS_Sem1_Tanpa_Evaluasi"
}

X = df[list(features.keys())].rename(columns=features)
y = df["Label"]

feature_names = list(X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["data"]["test_size"],
    random_state=config["data"]["random_state"],
    stratify=y
)

print("========================================")
print("Training Model")
print("Train Size :", X_train.shape[0])
print("Test Size  :", X_test.shape[0])
print("========================================")

with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

    mlflow.log_param("algorithm", "RandomForest")
    mlflow.log_param("test_size", config["data"]["test_size"])

    model = RandomForestClassifier(
        random_state=config["data"]["random_state"],
        class_weight=config["model"]["class_weight"]
    )

    grid = GridSearchCV(
        model,
        config["model"]["param_grid"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    mlflow.log_params(grid.best_params_)

    # EVALUATION
    y_pred = best_model.predict(X_test)

    acc_train = best_model.score(X_train, y_train)
    acc_test = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy_train", acc_train)
    mlflow.log_metric("accuracy_test", acc_test)
    mlflow.log_metric("f1_score", f1)

    # CLASSIFICATION REPORT
    report_text = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Dropout", "Dropout"]
    )

    with open("classification_report.txt", "w") as f:
        f.write(report_text)

    mlflow.log_artifact("classification_report.txt")
    os.remove("classification_report.txt")

    # CREATE SIGNATURE
    signature = infer_signature(X_train, best_model.predict(X_train))

    # REGISTER MODEL WITH SIGNATURE
    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="model",
        signature=signature,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    # SAVE LOCAL ARTIFACT
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/model_dropout.pkl")
    joblib.dump(feature_names, "models/fitur.pkl")

    print("========================================")
    print("Training Selesai")
    print("Model Registrasi :", REGISTERED_MODEL_NAME)
    print("Accuracy Testing    :", round(acc_test, 2))
    print("F1 SCORE         :", round(f1, 2))
    print("========================================")