"""
Nodes for the `modelo_regresion` pipeline.

These nodes implement training and evaluation helper functions for the
regression models with GridSearchCV and cross-validation for hyperparameter tuning.
The functions are kept small and focused so they can be composed by the pipeline.

Each train_* function receives a DataFrame and returns a trained model plus 
evaluation metrics with cross-validation results.
"""

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
try:
	import xgboost as xgb
	from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost is optional in some environments
	xgb = None
	XGBRegressor = None
import os
import pickle
import joblib


MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "data", "06_models", "regression"))


def _ensure_models_dir() -> None:
	os.makedirs(MODELS_DIR, exist_ok=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	"""Calculate comprehensive regression metrics."""
	mse = mean_squared_error(y_true, y_pred)
	return {
		"mse": float(mse),
		"rmse": float(np.sqrt(mse)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred)),
	}


def _cross_validation_scores(model: Any, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
	"""Perform cross-validation and return mean and std scores."""
	cv_scores = cross_val_score(model, X, y, cv=KFold(n_splits=cv, shuffle=True, random_state=42), 
	                           scoring='r2', n_jobs=-1)
	return {
		"cv_mean": float(cv_scores.mean()),
		"cv_std": float(cv_scores.std()),
		"cv_scores": cv_scores.tolist()
	}


def train_linear_regression(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, 
                           random_state: int = 42, cv: int = 5) -> Tuple[LinearRegression, Dict]:
	"""Train Linear Regression with GridSearchCV and cross-validation."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data.drop(columns=[target_col])
	y = data[target_col]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	
	# GridSearchCV parameters for Linear Regression
	param_grid = {
		'fit_intercept': [True, False],
		'normalize': [True, False] if hasattr(LinearRegression(), 'normalize') else [False]
	}
	
	base_model = LinearRegression()
	grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	best_model = grid_search.best_estimator_
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	metrics.update({
		"best_params": grid_search.best_params_,
		"best_score": float(grid_search.best_score_)
	})
	
	# Persist model
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "linear_model.pkl")
	joblib.dump({
		'model': best_model,
		'feature_names': X.columns.tolist()
	}, path)
	metrics["model_path"] = path
	return best_model, metrics


def train_decision_tree(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, 
                       random_state: int = 42, cv: int = 5) -> Tuple[DecisionTreeRegressor, Dict]:
	"""Train Decision Tree Regressor with GridSearchCV and cross-validation."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")
		
	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	
	# GridSearchCV parameters
	param_grid = {
		'max_depth': [None, 3, 5, 7, 10],
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4],
		'criterion': ['mse', 'mae'] if hasattr(DecisionTreeRegressor(), 'criterion') else ['squared_error', 'absolute_error']
	}
	
	base_model = DecisionTreeRegressor(random_state=random_state)
	grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	best_model = grid_search.best_estimator_
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	metrics.update({
		"best_params": grid_search.best_params_,
		"best_score": float(grid_search.best_score_)
	})
	
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "dt_model.pkl")
	joblib.dump({
		'model': best_model,
		'feature_names': X.columns.tolist()
	}, path)
	metrics["model_path"] = path
	return best_model, metrics


def train_random_forest(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, 
                       random_state: int = 42, cv: int = 5) -> Tuple[RandomForestRegressor, Dict]:
	"""Train Random Forest Regressor with GridSearchCV and cross-validation."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")
		
	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	
	# GridSearchCV parameters
	param_grid = {
		'n_estimators': [50, 100, 200],
		'max_depth': [None, 3, 5, 7],
		'min_samples_split': [2, 5, 10],
		'min_samples_leaf': [1, 2, 4]
	}
	
	base_model = RandomForestRegressor(random_state=random_state, n_jobs=-1)
	grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	best_model = grid_search.best_estimator_
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	metrics.update({
		"best_params": grid_search.best_params_,
		"best_score": float(grid_search.best_score_)
	})
	
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "rf_model.pkl")
	joblib.dump({
		'model': best_model,
		'feature_names': X.columns.tolist()
	}, path)
	metrics["model_path"] = path
	return best_model, metrics


def train_xgboost(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, 
                 random_state: int = 42, cv: int = 5) -> Tuple[object, Dict]:
	"""Train XGBoost Regressor with GridSearchCV and cross-validation."""
	if XGBRegressor is None:
		raise ImportError("xgboost is not installed in the current environment")

	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	
	# GridSearchCV parameters
	param_grid = {
		'n_estimators': [50, 100, 200],
		'max_depth': [3, 5, 7],
		'learning_rate': [0.01, 0.1, 0.2],
		'subsample': [0.8, 0.9, 1.0]
	}
	
	base_model = XGBRegressor(random_state=random_state, n_jobs=-1)
	grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	best_model = grid_search.best_estimator_
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	metrics.update({
		"best_params": grid_search.best_params_,
		"best_score": float(grid_search.best_score_)
	})
	
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "xgb_model.pkl")
	joblib.dump({
		'model': best_model,
		'feature_names': X.columns.tolist()
	}, path)
	metrics["model_path"] = path
	return best_model, metrics


def train_svr(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, 
             random_state: int = 42, cv: int = 5) -> Tuple[SVR, Dict]:
	"""Train Support Vector Regressor with GridSearchCV and cross-validation."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data.drop(columns=[target_col])
	y = data[target_col]
	
	# Scale features for SVR
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
	
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
	
	# GridSearchCV parameters
	param_grid = {
		'C': [0.1, 1, 10, 100],
		'kernel': ['rbf', 'linear', 'poly'],
		'gamma': ['scale', 'auto'],
		'epsilon': [0.01, 0.1, 0.2]
	}
	
	base_model = SVR()
	grid_search = GridSearchCV(base_model, param_grid, cv=cv, scoring='r2', n_jobs=-1)
	grid_search.fit(X_train, y_train)
	
	best_model = grid_search.best_estimator_
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X_scaled, y, cv)
	metrics.update(cv_metrics)
	metrics.update({
		"best_params": grid_search.best_params_,
		"best_score": float(grid_search.best_score_)
	})
	
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "svr_model.pkl")
	joblib.dump({
		'model': best_model,
		'scaler': scaler,
		'feature_names': X.columns.tolist()
	}, path)
	metrics["model_path"] = path
	return best_model, metrics


