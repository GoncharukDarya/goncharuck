import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def classification_training(data):
    """
    Функция для обучения моделей классификации на данных о ментальном здоровье
    
    Args:
        data: DataFrame с обработанными данными из предыдущей лабораторной работы
    """
    # 1. Трансформация целевой переменной в бинарную
    
    # Создаем копию данных чтобы не изменять оригинал
    df = data.copy()
    
    # Используем медиану для балансировки классов
    median_value = df['mental_wellness_index_0_100'].median()
    
    # Трансформируем mental_wellness_index_0_100 в бинарную переменную
    df['target'] = df['mental_wellness_index_0_100'].apply(lambda x: 0 if x < median_value else 1)
    
    # Удаляем исходный столбец mental_wellness_index_0_100
    df = df.drop('mental_wellness_index_0_100', axis=1)
    
    # 2. Подготовка данных для обучения
    
    # Разделяем на признаки (X) и целевую переменную (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделяем на тренировочную и тестовую выборки (70%/30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Стандартизируем признаки (важно для SVM и логистической регрессии)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Обучение и оценка моделей
    
    # Модель 1: Support Vector Machine (SVM)
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    
    # Предсказания SVM
    y_pred_svm = svm_model.predict(X_test_scaled)
    
    # Метрики для SVM
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    
    # Модель 2: Логистическая регрессия
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Предсказания логистической регрессии
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Метрики для логистической регрессии
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    
    # 4. Вывод результатов по заданному шаблону
    print(f"SVM: {accuracy_svm:.4f}; {f1_svm:.4f}")
    print(f"LR: {accuracy_lr:.4f}; {f1_lr:.4f}")
