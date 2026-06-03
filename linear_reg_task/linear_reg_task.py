import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import os
from pathlib import Path
import joblib

# Пути (без изменений)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
PIPELINE_PATH = RESULTS_DIR / "model_pipeline.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"

def save_object(model, filename):
    joblib.dump(model, filename)

def load_data():
    data = np.load(TRAIN_DATASET_PATH)
    return data['X_train'], data['y_train']

def create_pipeline():
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=0.5, random_state=42))
    ])
    return pipeline

def train_model(X_train, y_train):
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    print(f"R² модели: {r2:.4f}")
    print(f"MSE модели: {mse:.2f}")
    return r2, mse

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    pipeline = train_model(X_train_split, y_train_split)
    r2_val, mse_val = evaluate_model(pipeline, X_val, y_val)
    
    final_pipeline = train_model(X_train, y_train)
    r2_train, mse_train = evaluate_model(final_pipeline, X_train, y_train)
    
    save_object(final_pipeline, PIPELINE_PATH)
    print(f"Пайплайн сохранен в {PIPELINE_PATH}")
    print("\nПримечание: Для прохождения теста R² должен быть не менее 0.9489")

# Вызываем main() для автоматического обучения и сохранения
main()