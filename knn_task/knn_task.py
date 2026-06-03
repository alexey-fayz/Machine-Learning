import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
MODEL_PATH = RESULTS_DIR / "best_knn_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"

def save_object(model, filename):
    """
    Сохраняет модель и не только в файл. Не меняйте эту функцию.
    
    Аргументы:
        model: Обученная модель
        filename: Путь для сохранения модели
    """
    joblib.dump(model, filename)

def load_data():
    """
    Загружает тренировочные данные из директории данных. Эта функция уже рабочая, так что лучше ее не трогать.
    
    Возвращает:
        tuple: (X_train, y_train) где X_train - матрица признаков для обучения, 
               а y_train - метки классов для обучения
    """
    data = np.load(TRAIN_DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    
    return X_train, y_train

def preprocess_data(X, y):
    """
    Обрабатывает данные, применяя масштабирование признаков.
    
    Аргументы:
        X: Матрица признаков
        y: Метки классов
        
    Возвращает:
        tuple: (X_scaled, y, scaler)
    """
    # 1. Создаём объект StandardScaler
    scaler = StandardScaler()
    
    # 2. Обучаем scaler на данных и применяем масштабирование
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_knn_model(X_train, y_train):
    """
    Обучает классификатор KNN с оптимальными гиперпараметрами, используя кросс-валидацию.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель KNN с лучшими найденными гиперпараметрами
    """
    # 1. Определяем базовую модель KNN
    knn_base = KNeighborsClassifier()
    
    # 2. Задаём сетку гиперпараметров для поиска
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # 3. Настраиваем GridSearchCV с кросс-валидацией (5 фолдов)
    grid_search = GridSearchCV(
        estimator=knn_base,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # 4. Обучаем на всех тренировочных данных
    grid_search.fit(X_train, y_train)
    
    # 5. Выводим лучшие параметры и соответствующую точность кросс-валидации
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность при кросс-валидации: {grid_search.best_score_:.4f}")
    
    # 6. Возвращаем лучшую обученную модель
    return grid_search.best_estimator_

def evaluate_model(model, X, y):
    """
    Оценивает производительность модели на данных.
    
    Аргументы:
        model: Обученная модель
        X: Признаки для оценки
        y: Истинные метки классов
        
    Возвращает:
        float: Точность классификации
    """
    # 1. Делаем предсказания
    y_pred = model.predict(X)
    
    # 2. Вычисляем точность
    accuracy = accuracy_score(y, y_pred)
    
    print(f"Точность модели на данных: {accuracy:.4f}")
    return accuracy

def main():
    """
    Основная функция для выполнения рабочего процесса KNN:
    1. Загрузка тренировочных данных
    2. Предобработка данных
    3. Обучение модели KNN с использованием кросс-валидации
    4. Сохранение модели и масштабировщика
    """
    # Создаем директорию для результатов, если она не существует
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка тренировочных данных
    X_train,y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Предобработка данных
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    print(f"Данные обработаны: X_train_scaled={X_train_scaled.shape}")
    
    # Обучение модели KNN
    print("Обучение модели KNN с использованием кросс-валидации...")
    knn_model = train_knn_model(X_train_scaled, y_train)
    
    # Оценка модели на тренировочных данных (опционально, для контроля)
    print("Оценка модели на полных тренировочных данных...")
    evaluate_model(knn_model, X_train_scaled, y_train)
    
    # Сохранение модели и масштабировщика
    print("Сохранение модели и масштабировщика...")
    save_object(knn_model, MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    print(f"Модель сохранена в {MODEL_PATH}")
    print(f"Масштабировщик сохранен в {SCALER_PATH}")

if __name__ == "__main__":
    main()