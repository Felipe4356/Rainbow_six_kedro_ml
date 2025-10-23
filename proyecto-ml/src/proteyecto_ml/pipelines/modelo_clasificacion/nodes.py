"""
Nodes for the `modelo_clasificacion` pipeline.

These nodes implement training and evaluation helper functions for the
classification models. The functions include GridSearchCV with cross-validation
for hyperparameter tuning and comprehensive model evaluation.

Each train_* function receives a DataFrame and returns a trained model plus 
evaluation metrics with cross-validation results.
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import joblib

MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "data", "06_models", "classification"))

def _ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

def _prepare_target(data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Prepare target variable for classification."""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Encode target if it's categorical
    le = LabelEncoder()
    if y.dtype == 'object':
        y_encoded = le.fit_transform(y)
        return X, pd.Series(y_encoded, index=y.index), le
    else:
        # For numerical targets, we might need to create classes
        # Let's create binary classification based on median
        median_val = y.median()
        y_binary = (y > median_val).astype(int)
        le.fit(['low', 'high'])
        return X, y_binary, le

def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Calculate comprehensive classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    
    # Add ROC AUC if probabilities are available and it's binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim > 1 else y_proba))
        except:
            pass
    
    return metrics

def _cross_validation_scores(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """Perform cross-validation and return mean and std scores."""
    cv_scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42), 
                               scoring='accuracy', n_jobs=-1)
    return {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist()
    }

def train_logistic_regression(data: pd.DataFrame, target_col: str = "outcome", test_size: float = 0.2, 
                             random_state: int = 42, cv: int = 5) -> Tuple[Any, Dict]:
    """Train Logistic Regression with GridSearchCV and cross-validation."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X, y, label_encoder = _prepare_target(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # GridSearchCV parameters
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 500]
    }
    
    base_model = LogisticRegression(random_state=random_state)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv)
    metrics.update(cv_metrics)
    metrics.update({
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_)
    })
    
    # Save model
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_knn_classifier(data: pd.DataFrame, target_col: str = "outcome", test_size: float = 0.2,
                        random_state: int = 42, cv: int = 5) -> Tuple[Any, Dict]:
    """Train K-Nearest Neighbors with GridSearchCV and cross-validation."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X, y, label_encoder = _prepare_target(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # GridSearchCV parameters
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    base_model = KNeighborsClassifier()
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv)
    metrics.update(cv_metrics)
    metrics.update({
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_)
    })
    
    # Save model
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "knn_classifier.pkl")
    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_svm_classifier(data: pd.DataFrame, target_col: str = "outcome", test_size: float = 0.2,
                        random_state: int = 42, cv: int = 5) -> Tuple[Any, Dict]:
    """Train Support Vector Machine with GridSearchCV and cross-validation."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X, y, label_encoder = _prepare_target(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # GridSearchCV parameters
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01]
    }
    
    base_model = SVC(random_state=random_state, probability=True)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv)
    metrics.update(cv_metrics)
    metrics.update({
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_)
    })
    
    # Save model
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "svm_classifier.pkl")
    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_decision_tree_classifier(data: pd.DataFrame, target_col: str = "outcome", test_size: float = 0.2,
                                  random_state: int = 42, cv: int = 5) -> Tuple[Any, Dict]:
    """Train Decision Tree with GridSearchCV and cross-validation."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X, y, label_encoder = _prepare_target(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # GridSearchCV parameters
    param_grid = {
        'max_depth': [None, 3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    base_model = DecisionTreeClassifier(random_state=random_state)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv)
    metrics.update(cv_metrics)
    metrics.update({
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_)
    })
    
    # Save model
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "decision_tree_classifier.pkl")
    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_random_forest_classifier(data: pd.DataFrame, target_col: str = "outcome", test_size: float = 0.2,
                                  random_state: int = 42, cv: int = 5) -> Tuple[Any, Dict]:
    """Train Random Forest with GridSearchCV and cross-validation."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X, y, label_encoder = _prepare_target(data, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state, stratify=y)
    
    # GridSearchCV parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    base_model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv)
    metrics.update(cv_metrics)
    metrics.update({
        "best_params": grid_search.best_params_,
        "best_score": float(grid_search.best_score_)
    })
    
    # Save model
    _ensure_models_dir()
    model_path = os.path.join(MODELS_DIR, "random_forest_classifier.pkl")
    joblib.dump({
        'model': best_model,
        'label_encoder': label_encoder,
        'feature_names': X.columns.tolist()
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics
