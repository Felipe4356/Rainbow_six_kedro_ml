"""
Nodes for the `modelo_clasificacion` pipeline.

These nodes implement training and evaluation helper functions for the
classification models. The functions include GridSearchCV with cross-validation
for hyperparameter tuning and comprehensive model evaluation.

Each train_* function receives a DataFrame and returns a trained model plus 
evaluation metrics with cross-validation results.
"""

from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline as SkPipeline
import os
import pickle
import joblib

MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "data", "06_models", "classification"))

def _ensure_models_dir() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

def _prepare_target(data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """Prepare target variable for classification.

    Rules:
    - If target is object/categorical: label-encode categories.
    - If target is numeric but strictly in {0,1}: treat as already-class labels (no binarization).
    - Otherwise (numeric continuous): binarize by median threshold.
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    le = LabelEncoder()

    # Categorical/object target: label-encode
    if y.dtype == 'object' or str(y.dtype).startswith('category'):
        y_encoded = le.fit_transform(y)
        return X, pd.Series(y_encoded, index=y.index), le

    # Numeric target: check for strict binary labels {0,1}
    y_non_na = y.dropna()
    if not y_non_na.empty and set(pd.unique(y_non_na.astype(int))) <= {0, 1} and y_non_na.nunique() <= 2:
        # Use as-is and fit a simple encoder mapping 0->0, 1->1 for consistency
        le.classes_ = np.array([0, 1])
        return X, y.astype(int), le

    # Fallback for continuous numeric target: binarize by median
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

def _get_scaler(scaler_name: Optional[str]) -> Optional[Any]:
    if not scaler_name:
        return None
    name = str(scaler_name).lower()
    if name in ["none", "null", "false"]:
        return None
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    # default: no scaler
    return None


def _clean_params(params: Optional[Dict[str, Any]], drop_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return a copy of params without conflicting keys (e.g., random_state).

    This prevents passing the same keyword twice when we also pass it explicitly.
    """
    if not params:
        return {}
    drop = set(drop_keys or [])
    return {k: v for k, v in params.items() if k not in drop}


def _cross_validation_scores(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5, scaler_name: Optional[str] = None) -> Dict[str, float]:
    """Perform cross-validation with optional scaling and return mean and std scores."""
    scaler = _get_scaler(scaler_name)
    estimator = model
    if scaler is not None:
        estimator = SkPipeline([
            ("scaler", _get_scaler(scaler_name)),
            ("model", model),
        ])

    cv_scores = cross_val_score(
        estimator,
        X,
        y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='accuracy',
        n_jobs=-1,
    )
    return {
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist()
    }

def train_logistic_regression(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]],
    test_size: float,
    random_state: int,
    cv: int,
    model_params: Dict[str, Any],
    grid_search_cfg: Dict[str, Any],
    scaler_name: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Train Logistic Regression with optional GridSearchCV using params from YAML."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    # Selección de features si se indican; si no, usa todo excepto la etiqueta
    X_full, y, label_encoder = _prepare_target(data, target_col)
    X = X_full[feature_cols].copy() if feature_cols else X_full
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    base_model = LogisticRegression(random_state=random_state, **_clean_params(model_params, ["random_state"]))

    use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
    if use_gs:
        param_grid = grid_search_cfg.get("param_grid", {})
        gs_cv = int(grid_search_cfg.get("cv", cv))
        scoring = grid_search_cfg.get("scoring", "accuracy")
        n_jobs = int(grid_search_cfg.get("n_jobs", -1))
        verbose = int(grid_search_cfg.get("verbose", 0))
        estimator = base_model
        grid_search = GridSearchCV(estimator, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv, scaler_name)
    metrics.update(cv_metrics)
    if 'grid_search' in locals():
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
        'feature_names': X.columns.tolist(),
        'scaler': scaler_name,
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_knn_classifier(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]],
    test_size: float,
    random_state: int,
    cv: int,
    model_params: Dict[str, Any],
    grid_search_cfg: Dict[str, Any],
    scaler_name: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Train K-Nearest Neighbors with optional GridSearchCV using params from YAML."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X_full, y, label_encoder = _prepare_target(data, target_col)
    X = X_full[feature_cols].copy() if feature_cols else X_full
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    base_model = KNeighborsClassifier(**_clean_params(model_params))
    use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
    if use_gs:
        param_grid = grid_search_cfg.get("param_grid", {})
        gs_cv = int(grid_search_cfg.get("cv", cv))
        scoring = grid_search_cfg.get("scoring", "accuracy")
        n_jobs = int(grid_search_cfg.get("n_jobs", -1))
        verbose = int(grid_search_cfg.get("verbose", 0))
        estimator = base_model
        grid_search = GridSearchCV(estimator, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv, scaler_name)
    metrics.update(cv_metrics)
    if 'grid_search' in locals():
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
        'feature_names': X.columns.tolist(),
        'scaler': scaler_name,
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_svm_classifier(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]],
    test_size: float,
    random_state: int,
    cv: int,
    model_params: Dict[str, Any],
    grid_search_cfg: Dict[str, Any],
    scaler_name: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Train SVC with optional GridSearchCV using params from YAML."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X_full, y, label_encoder = _prepare_target(data, target_col)
    X = X_full[feature_cols].copy() if feature_cols else X_full
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Optional fast training via subsampling
    mp_all = dict(model_params or {})
    max_train_samples = mp_all.pop("max_train_samples", None)
    # ensure probability=True default can be overridden
    mp = _clean_params(mp_all, ["random_state"])  # random_state is passed explicitly
    mp.setdefault("probability", True)
    base_model = SVC(random_state=random_state, **mp)

    use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
    if use_gs:
        param_grid = grid_search_cfg.get("param_grid", {})
        gs_cv = int(grid_search_cfg.get("cv", cv))
        scoring = grid_search_cfg.get("scoring", "accuracy")
        n_jobs = int(grid_search_cfg.get("n_jobs", -1))
        verbose = int(grid_search_cfg.get("verbose", 0))
        estimator = base_model
        grid_search = GridSearchCV(estimator, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
        # optional subsample inside GS as well
        if max_train_samples:
            # Stratified subsample
            if isinstance(max_train_samples, float) and 0 < max_train_samples < 1:
                train_size = max_train_samples
            elif isinstance(max_train_samples, int):
                train_size = min(max_train_samples, len(y_train)) / len(y_train)
            else:
                train_size = None
            if train_size and 0 < train_size < 1:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
                idx_train, _ = next(sss.split(X_train, y_train))
                X_tr_sub, y_tr_sub = X_train[idx_train], y_train.iloc[idx_train]
                grid_search.fit(X_tr_sub, y_tr_sub)
            else:
                grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        if max_train_samples:
            if isinstance(max_train_samples, float) and 0 < max_train_samples < 1:
                train_size = max_train_samples
            elif isinstance(max_train_samples, int):
                train_size = min(max_train_samples, len(y_train)) / len(y_train)
            else:
                train_size = None
            if train_size and 0 < train_size < 1:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
                idx_train, _ = next(sss.split(X_train, y_train))
                X_tr_sub, y_tr_sub = X_train[idx_train], y_train.iloc[idx_train]
                base_model.fit(X_tr_sub, y_tr_sub)
            else:
                base_model.fit(X_train, y_train)
        else:
            base_model.fit(X_train, y_train)
        best_model = base_model
    y_pred = best_model.predict(X_test)
    # Obtain probability or decision scores, whichever is available
    y_scores = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_scores = best_model.predict_proba(X_test)
        except Exception:
            y_scores = None
    if y_scores is None and hasattr(best_model, "decision_function"):
        try:
            y_scores = best_model.decision_function(X_test)
        except Exception:
            y_scores = None
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_scores)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv, scaler_name)
    metrics.update(cv_metrics)
    if 'grid_search' in locals():
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
        'feature_names': X.columns.tolist(),
        'scaler': scaler_name,
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_decision_tree_classifier(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]],
    test_size: float,
    random_state: int,
    cv: int,
    model_params: Dict[str, Any],
    grid_search_cfg: Dict[str, Any],
    scaler_name: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Train Decision Tree with optional GridSearchCV using params from YAML."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X_full, y, label_encoder = _prepare_target(data, target_col)
    X = X_full[feature_cols].copy() if feature_cols else X_full
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    base_model = DecisionTreeClassifier(random_state=random_state, **_clean_params(model_params, ["random_state"]))
    use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
    if use_gs:
        param_grid = grid_search_cfg.get("param_grid", {})
        gs_cv = int(grid_search_cfg.get("cv", cv))
        scoring = grid_search_cfg.get("scoring", "accuracy")
        n_jobs = int(grid_search_cfg.get("n_jobs", -1))
        verbose = int(grid_search_cfg.get("verbose", 0))
        estimator = base_model
        grid_search = GridSearchCV(estimator, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv, scaler_name)
    metrics.update(cv_metrics)
    if 'grid_search' in locals():
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
        'feature_names': X.columns.tolist(),
        'scaler': scaler_name,
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics

def train_random_forest_classifier(
    data: pd.DataFrame,
    target_col: str,
    feature_cols: Optional[List[str]],
    test_size: float,
    random_state: int,
    cv: int,
    model_params: Dict[str, Any],
    grid_search_cfg: Dict[str, Any],
    scaler_name: Optional[str] = None,
) -> Tuple[Any, Dict]:
    """Train Random Forest with optional GridSearchCV using params from YAML."""
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    
    X_full, y, label_encoder = _prepare_target(data, target_col)
    X = X_full[feature_cols].copy() if feature_cols else X_full
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = _get_scaler(scaler_name)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    base_model = RandomForestClassifier(random_state=random_state, **_clean_params(model_params, ["random_state"]))
    use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
    if use_gs:
        param_grid = grid_search_cfg.get("param_grid", {})
        gs_cv = int(grid_search_cfg.get("cv", cv))
        scoring = grid_search_cfg.get("scoring", "accuracy")
        n_jobs = int(grid_search_cfg.get("n_jobs", -1))
        verbose = int(grid_search_cfg.get("verbose", 0))
        estimator = base_model
        grid_search = GridSearchCV(estimator, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        base_model.fit(X_train, y_train)
        best_model = base_model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = _classification_metrics(y_test, y_pred, y_proba)
    cv_metrics = _cross_validation_scores(best_model, X, y, cv, scaler_name)
    metrics.update(cv_metrics)
    if 'grid_search' in locals():
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
        'feature_names': X.columns.tolist(),
        'scaler': scaler_name,
    }, model_path)
    metrics["model_path"] = model_path
    
    return best_model, metrics
