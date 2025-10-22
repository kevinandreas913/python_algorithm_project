import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import warnings

def load_data():
    df_train = pd.read_csv("diabetes_preprocessing/train_set.csv", sep=",")
    df_test = pd.read_csv("diabetes_preprocessing/test_set.csv", sep=",")
    
    target_col = 'Outcome'
    
    X_train = df_train.drop(target_col, axis=1)
    y_train = df_train[target_col]
    X_test = df_test.drop(target_col, axis=1)
    y_test = df_test[target_col]
    
    return X_train, X_test, y_train, y_test


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    
    mlflow.set_experiment("Diabetes Classification - Skilled Tuning")
    print("MLflow experiment set to 'Diabetes Classification - Skilled Tuning'")

    try:
        X_train, X_test, y_train, y_test = load_data()
        print("Data berhasil dimuat.")
    except FileNotFoundError:
        print(f"Error: Data files tidak ditemukan.")
        print(f"Pastikan 'train_set.csv' dan 'test_set.csv' ada di dalam folder")
        return

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100], 
        'penalty': ['l1', 'l2'], 
        'solver': ['liblinear']  
    }

    model = LogisticRegression(random_state=42)

    print("hyperparameter tuning dengan GridSearchCV...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1) 
    
    grid_search.fit(X_train, y_train)
    print("Hyperparameter Tuning Selesai.")

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score_cv = grid_search.best_score_ 

    print(f"Parameter Terbaik: {best_params}")
    print(f"Akurasi Training (CV): {best_score_cv:.4f}")

    with mlflow.start_run() as run:
        print(f"\nStarting run: {run.info.run_id} (Manual Logging)")
        
        print("Mencatat parameter terbaik ke MLflow...")
        mlflow.log_params(best_params)
        
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        print("Mencatat metrik ke MLflow...")
        mlflow.log_metric("train_accuracy_cv", best_score_cv)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1_score", f1)
        
        print("Mencatat model sebagai artefak MLflow...")
        mlflow.sklearn.log_model(best_model, "model")

        print("\nRun complete. Artefak, parameter, dan metrik dicatat secara manual.")
        print("Jalankan 'mlflow ui' di terminal Anda untuk melihat hasilnya.")

if __name__ == "__main__":
    main()

