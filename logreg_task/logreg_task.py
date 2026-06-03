import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, average_precision_score, precision_recall_curve
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
MODEL_PATH = RESULTS_DIR / "best_logreg_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
THRESHOLD_PATH = RESULTS_DIR / "optimal_threshold.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"

def save_object(model, filename):
    """Сохраняет модель и не только в файл."""
    joblib.dump(model, filename)

def load_data():
    """Загружает тренировочные данные."""
    data = np.load(TRAIN_DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    return X_train, y_train

def preprocess_data(X, y):
    """
    Обрабатывает данные, применяя масштабирование признаков.
    Масштабирование критично для логистической регрессии, чтобы признаки с большими 
    значениями не доминировали при расчете градиента.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_logreg_model(X_train, y_train):
    """
    Максимально точная настройка для преодоления порога 0.5121.
    """
    logreg = LogisticRegression(
        C=1.0,               # Стандартное значение, обычно самое стабильное
        solver='lbfgs',      # Один из самых точных решателей
        max_iter=5000,       # Даем модели огромный запас по итерациям
        tol=1e-8,            # Экстремально высокая точность сходимости
        class_weight=None,   # Не искажаем вероятности для максимизации AP
        random_state=42
    )
    
    logreg.fit(X_train, y_train)
    return logreg

def get_optimal_threshold(model, X, y):
    """
    Находит порог, где recall >= 0.8 с максимальной precision.
    """
    # 1. Получаем вероятности только для положительного класса (1)
    y_proba = model.predict_proba(X)[:, 1]
    
    # 2. Вычисляем кривую precision-recall
    # precision_recall_curve возвращает массивы разной длины: 
    # thresholds на 1 короче, чем precision и recall.
    precisions, recalls, thresholds = precision_recall_curve(y, y_proba)
    
    # 3. Ищем индексы, где recall удовлетворяет условию $Recall \ge 0.8$
    valid_indices = np.where(recalls >= 0.8)[0]
    
    # 4. Среди этих индексов ищем тот, где precision максимален
    # Важно: берем индекс из thresholds, поэтому ограничиваем выборку
    best_index = valid_indices[np.argmax(precisions[valid_indices])]
    
    # Если вдруг индекс вышел за пределы thresholds (последний элемент recall всегда 0)
    if best_index >= len(thresholds):
        best_index = len(thresholds) - 1
        
    optimal_threshold = thresholds[best_index]
    
    return optimal_threshold

def evaluate_model(model, X, y):
    """Оценивает производительность модели."""
    # Предсказание классов (стандартный порог 0.5)
    y_pred = model.predict(X)
    # Вероятности для расчета Average Precision
    y_proba = model.predict_proba(X)[:, 1]
    
    accuracy = accuracy_score(y, y_pred)
    avg_precision = average_precision_score(y, y_proba)
    
    print(f"Точность модели (порог 0.5): {accuracy:.4f}")
    print(f"Average Precision модели: {avg_precision:.4f}")
    
    return accuracy, avg_precision

def main():
    """Основной рабочий процесс."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Загрузка
    X_train, y_train = load_data()
    
    # Предобработка
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    
    # Разделение для валидации
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Обучение и промежуточная оценка
    logreg_model = train_logreg_model(X_train_split, y_train_split)
    evaluate_model(logreg_model, X_val, y_val)
    
    # Обучение финальной модели на всех данных
    final_model = train_logreg_model(X_train_scaled, y_train)
    final_optimal_threshold = get_optimal_threshold(final_model, X_train_scaled, y_train)
    
    # Сохранение результатов
    save_object(final_model, MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    save_object(float(final_optimal_threshold), THRESHOLD_PATH)
    
    print(f"\nИтог:")
    print(f"Оптимальный порог: {final_optimal_threshold:.4f}")
    print(f"Объекты сохранены в директорию: {RESULTS_DIR}")

if __name__ == "__main__":
    main()