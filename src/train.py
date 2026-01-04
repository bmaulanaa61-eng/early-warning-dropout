import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.config_loader import load_config
from utils.logging import setup_logger


# LOGGER
logger = setup_logger("train")


# LOAD CONFIG
config = load_config()
logger.info("Config berhasil dimuat")


# MLFLOW SETUP
mlflow.set_tracking_uri(config["mlflow"]["tracking_url"])
mlflow.set_experiment(config["mlflow"]["experiment_name"])

REGISTERED_MODEL_NAME = config["mlflow"]["registered_model_name"]
logger.info(f"Registered model name: {REGISTERED_MODEL_NAME}")


# LOAD DATA
data_cfg = config["data"]

df = pd.read_csv(
    data_cfg["path"],
    delimiter=data_cfg["delimiter"]
)

logger.info(f"Data berhasil dimuat dengan shape: {df.shape}")


# DATA CLEANING
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

df["Target"] = df["Target"].str.strip()
df["Label"] = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)


# FEATURE SELECTION & RENAME (FROM CONFIG)
features = config["features"]

missing_columns = set(features.keys()) - set(df.columns)
if missing_columns:
    raise ValueError(f"Kolom berikut tidak ditemukan di data: {missing_columns}")

X = df[list(features.keys())].rename(columns=features)
y = df["Label"]

feature_names = list(X.columns)
logger.info(f"Jumlah fitur digunakan: {len(feature_names)}")


# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=data_cfg["test_size"],
    random_state=data_cfg["random_state"],
    stratify=y
)

logger.info(
    f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}"
)


# TRAINING WITH MLFLOW
with mlflow.start_run(run_name=config["mlflow"]["run_name"]):

    # LOG PARAMS
    mlflow.log_param("algorithm", config["model"]["algorithm"])
    mlflow.log_param("test_size", data_cfg["test_size"])
    mlflow.log_param("random_state", data_cfg["random_state"])

    # MODEL INIT
    model = RandomForestClassifier(
        random_state=data_cfg["random_state"],
        class_weight=config["model"]["class_weight"]
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=config["model"]["param_grid"],
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    logger.info("Training model dimulai...")
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

    logger.info(f"Accuracy Train : {acc_train:.4f}")
    logger.info(f"Accuracy Test  : {acc_test:.4f}")
    logger.info(f"F1 Score       : {f1:.4f}")

    # CLASSIFICATION REPORT
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Non-Dropout", "Dropout"]
    )

    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(report_path)
    os.remove(report_path)

    # MODEL SIGNATURE
    signature = infer_signature(
        X_train,
        best_model.predict(X_train)
    )

    # LOG & REGISTER MODEL
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    # SAVE LOCAL ARTIFACT (FALLBACK)
    os.makedirs(os.path.dirname(config["output"]["model_path"]), exist_ok=True)

    joblib.dump(best_model, config["output"]["model_path"])
    joblib.dump(feature_names, config["output"]["features_path"])

    logger.info("========================================")
    logger.info("TRAINING SELESAI")
    logger.info(f"Registered Model : {REGISTERED_MODEL_NAME}")
    logger.info(f"Accuracy Test    : {acc_test:.4f}")
    logger.info(f"F1 Score         : {f1:.4f}")
    logger.info("========================================")
