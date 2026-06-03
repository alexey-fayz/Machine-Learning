import numpy as np
from sklearn.model_selection import GridSearchCV
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
    Обучает классификатор KNN с оптимальными гиперпараметрами, используя расширенную кросс-валидацию.
    
    Аргументы:
        X_train: Признаки для обучения
        y_train: Метки классов для обучения
        
    Возвращает:
        object: Обученная модель KNN с лучшими найденными гиперпараметрами
    """
    # Базовая модель
    knn_base = KNeighborsClassifier()
    
    # Расширенная сетка гиперпараметров для достижения точности >= 90.5%
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15],          # более широкий диапазон соседей
        'weights': ['uniform', 'distance'],                # равные или обратные веса
        'metric': ['euclidean', 'manhattan', 'minkowski'], # различные метрики расстояния
        'p': [1, 2]                                        # параметр для метрики Минковского (1 - Манхэттен, 2 - Евклид)
    }
    
    # Кросс-валидация с 5 фолдами, использование всех ядер процессора
    grid_search = GridSearchCV(
        estimator=knn_base,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Обучение на всех тренировочных данных
    grid_search.fit(X_train, y_train)
    
    # Вывод результатов подбора
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность при кросс-валидации: {grid_search.best_score_:.4f}")
    
    # Возвращаем лучшую модель
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
    # Предсказание
    y_pred = model.predict(X)
    
    # Расчёт точности
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
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Предобработка данных
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    print(f"Данные обработаны: X_train_scaled={X_train_scaled.shape}")
    
    # Обучение модели KNN
    print("Обучение модели KNN с использованием расширенной кросс-валидации...")
    knn_model = train_knn_model(X_train_scaled, y_train)
    
    # Оценка модели на тренировочных данных (опционально)
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