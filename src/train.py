import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score


with open("config.yaml","r") as settings:
    config = yaml.safe_load(settings)

mlflow.set_tracking_uri(config['mlflow']['tracking_url'])
mlflow.set_experiment(config['mlflow']['experiment_name'])

#Load Data
df = pd.read_csv(config['data']['path'],delimiter=config['data']['delimiter'])
df.columns = df.columns.str.strip()
df = df.rename(columns={'Daytime/evening attendance\t': 'Daytime/evening attendance'})
df.drop_duplicates()
df['Target'] = df['Target'].str.strip()
df['Label'] = df['Target'].apply(lambda x: 1 if x == 'Dropout' else 0 )

#Features
features  = {
    'Marital status': 'Status_Pernikahan',
    'Nacionality': 'Kewarganegaraan',
    'Gender': 'Jenis_Kelamin',
    'Age at enrollment': 'Usia_Saat_Daftar',
    'Application mode': 'Jalur_Pendaftaran',
    'Application order': 'Urutan_Pilihan',
    'Course': 'Program_Studi',
    'Daytime/evening attendance': 'Kelas_Siang_Malam',
    'Previous qualification': 'Pendidikan_Sebelumnya',
    'Previous qualification (grade)': 'Nilai_Pendidikan_Sebelumnya',
    'Admission grade': 'Nilai_Masuk',
    "Mother's qualification": 'Pendidikan_Ibu',
    "Father's qualification": 'Pendidikan_Ayah',
    "Mother's occupation": 'Pekerjaan_Ibu',
    "Father's occupation": 'Pekerjaan_Ayah',
    'Displaced': 'Pindahan',
    'Educational special needs': 'Berkebutuhan_Khusus',
    'Debtor': 'Punya_Tunggakan',
    'Tuition fees up to date': 'SPP_Lunas',
    'Scholarship holder': 'Penerima_Beasiswa',
    'International': 'Mahasiswa_Internasional',
    'Curricular units 1st sem (credited)': 'SKS_Sem1_Diakui',
    'Curricular units 1st sem (enrolled)': 'SKS_Sem1_Diambil',
    'Curricular units 1st sem (evaluations)': 'SKS_Sem1_Evaluasi',
    'Curricular units 1st sem (approved)': 'SKS_Sem1_Lulus',
    'Curricular units 1st sem (grade)': 'Nilai_Sem1',
    'Curricular units 1st sem (without evaluations)': 'SKS_Sem1_Tanpa_Evaluasi'
}

X = df[list(features.keys())].copy()
X = X.rename(columns=features)
y = df['Label']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=config['data']['test_size'],
    random_state=config['data']['random_state'],
    stratify=y
)

print("=" * 50)
print("Training Model")
print("=" * 50)
print(f"Data Training: {X_train.shape[0]}")
print(f"Data Testing: {X_test.shape[0]}")

with mlflow.start_run():
    mlflow.log_param("algorithm", config['model']['algorithm'])
    mlflow.log_param("test_size", config['data']['test_size'])
    mlflow.log_param("param_grid",config['model']['param_grid'])

    rf = RandomForestClassifier(
        random_state=config['data']['random_state'], 
        class_weight=config['model']['class_weight']
    )
    grid_search = GridSearchCV(
        rf,
        config['model']['param_grid'],
        cv=5,
        scoring='f1',
        n_jobs= 1
    )
    grid_search.fit(X_train,y_train)
    best_model = grid_search.best_estimator_

    mlflow.log_params(grid_search.best_params_)

    y_pred = best_model.predict(X_test)
    acc_train = best_model.score(X_train,y_train)
    acc_test = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy_train", acc_train)
    mlflow.log_metric("accuracy_test", acc_test)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(best_model, "model")
    print("\n" + "=" * 50)
    print("Hasil Evaluasi")
    print("=" * 50)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Akurasi Training: {acc_train*100:.2f}%")
    print(f"Akurasi Testing: {acc_test*100:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print("\nHasil Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=['Non-Dropout', 'Dropout']))
    print(f"\nModel tersimpan di MLflow")
