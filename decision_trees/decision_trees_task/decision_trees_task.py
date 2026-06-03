import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import joblib

# Пути для сохранения/загрузки данных. Не меняйте эти пути.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "result"
DT_MODEL_PATH = RESULTS_DIR / "decision_tree_model.pkl"
RF_MODEL_PATH = RESULTS_DIR / "random_forest_model.pkl"
GB_MODEL_PATH = RESULTS_DIR / "gradient_boosting_model.pkl"
SCALER_PATH = RESULTS_DIR / "feature_scaler.pkl"
TRAIN_DATASET_PATH = DATA_DIR / "train_data.npz"


def save_object(model, filename):
    joblib.dump(model, filename)


def load_data():
    data = np.load(TRAIN_DATASET_PATH)
    X_train = data['X_train']
    y_train = data['y_train']
    return X_train, y_train


def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        criterion='gini',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = CatBoostClassifier(
        iterations=700,                # увеличено для лучшего обучения
        learning_rate=0.04,            # немного снижена скорость
        depth=8,                       # больше глубина для улавливания сложных зависимостей
        l2_leaf_reg=4,                 # регуляризация L2
        bagging_temperature=0.5,       # немного шума для улучшения обобщения
        subsample=0.8,                 # случайная подвыборка строк
        random_seed=42,
        auto_class_weights='Balanced', # балансировка классов
        verbose=False,                 # отключаем вывод
        task_type='CPU'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)
    print(f"Метрики для модели {model_name}:")
    print(f"Точность: {accuracy:.4f}")
    print(f"F1-мера: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    return accuracy, f1, roc_auc


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    X_train, y_train = load_data()
    print(f"Загружены тренировочные данные: X_train={X_train.shape}, y_train={y_train.shape}")
    
    class_counts = np.bincount(y_train)
    print(f"Распределение классов в тренировочном наборе: {class_counts}")
    print(f"Доля класса 1: {class_counts[1] / len(y_train):.4f}")
    
    X_train_scaled, y_train, scaler = preprocess_data(X_train, y_train)
    print(f"Данные обработаны: X_train_scaled={X_train_scaled.shape}")
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    print(f"Разделение на обучающую и валидационную выборки:")
    print(f"X_train_split={X_train_split.shape}, y_train_split={y_train_split.shape}")
    print(f"X_val={X_val.shape}, y_val={y_val.shape}")
    
    print("Обучение моделей...\n")
    
    print("Обучение модели дерева решений...")
    dt_model = train_decision_tree(X_train_split, y_train_split)
    print("Оценка модели дерева решений на валидационных данных...")
    dt_metrics = evaluate_model(dt_model, X_val, y_val, "дерево решений")
    
    print("\nОбучение модели случайного леса...")
    rf_model = train_random_forest(X_train_split, y_train_split)
    print("Оценка модели случайного леса на валидационных данных...")
    rf_metrics = evaluate_model(rf_model, X_val, y_val, "случайный лес")
    
    print("\nОбучение модели градиентного бустинга...")
    gb_model = train_gradient_boosting(X_train_split, y_train_split)
    print("Оценка модели градиентного бустинга на валидационных данных...")
    gb_metrics = evaluate_model(gb_model, X_val, y_val, "градиентный бустинг")
    
    print("\nСравнение моделей на валидационных данных:")
    print("Модель            | Точность | F1-мера | ROC-AUC")
    print("--------------------|----------|---------|--------")
    print(f"Дерево решений     | {dt_metrics[0]:.4f}   | {dt_metrics[1]:.4f}  | {dt_metrics[2]:.4f}")
    print(f"Случайный лес      | {rf_metrics[0]:.4f}   | {rf_metrics[1]:.4f}  | {rf_metrics[2]:.4f}")
    print(f"Градиентный бустинг| {gb_metrics[0]:.4f}   | {gb_metrics[1]:.4f}  | {gb_metrics[2]:.4f}")
    
    print("\nОбучение финальных моделей на всех тренировочных данных...")
    final_dt_model = train_decision_tree(X_train_scaled, y_train)
    final_rf_model = train_random_forest(X_train_scaled, y_train)
    final_gb_model = train_gradient_boosting(X_train_scaled, y_train)
    
    print("Сохранение моделей и масштабировщика...")
    save_object(final_dt_model, DT_MODEL_PATH)
    save_object(final_rf_model, RF_MODEL_PATH)
    save_object(final_gb_model, GB_MODEL_PATH)
    save_object(scaler, SCALER_PATH)
    
    print(f"Модель дерева решений сохранена в {DT_MODEL_PATH}")
    print(f"Модель случайного леса сохранена в {RF_MODEL_PATH}")
    print(f"Модель градиентного бустинга сохранена в {GB_MODEL_PATH}")
    print(f"Масштабировщик сохранен в {SCALER_PATH}")
    
    print("\nПримечание: Для прохождения тестов необходимо достичь следующих значений метрик:")
    print("Дерево решений: ROC-AUC >= 0.75")
    print("Случайный лес: ROC-AUC >= 0.90")
    print("Градиентный бустинг: ROC-AUC >= 0.90")


if __name__ == "__main__":
    main()