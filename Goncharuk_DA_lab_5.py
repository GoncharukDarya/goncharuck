import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

def classification_training(data):
    df = data.copy()
    median_value = df['mental_wellness_index_0_100'].median()
    df['target'] = df['mental_wellness_index_0_100'].apply(lambda x: 0 if x < median_value else 1)
    df = df.drop('mental_wellness_index_0_100', axis=1)
    
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn)
    
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)
    
    print(f"KNN: {accuracy_knn:.4f}; {f1_knn:.4f}")
    print(f"DT: {accuracy_dt:.4f}; {f1_dt:.4f}")
    
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns.tolist(), class_names=['Low', 'High'], filled=True)
    plt.show()