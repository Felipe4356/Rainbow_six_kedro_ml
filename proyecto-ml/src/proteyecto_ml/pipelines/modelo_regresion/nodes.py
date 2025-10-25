"""
Nodes for the `modelo_regresion` pipeline.

These nodes implement training and evaluation helper functions for the
regression models with GridSearchCV and cross-validation for hyperparameter tuning.
The functions are kept small and focused so they can be composed by the pipeline.

Each train_* function receives a DataFrame and returns a trained model plus 
evaluation metrics with cross-validation results.
"""

from typing import Dict, Tuple, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# SVR removed: replaced by XGBoost for regression models
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline as SkPipeline
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


def _get_scaler(name: Optional[str]) -> Optional[Any]:
	if not name:
		return None
	n = str(name).lower()
	if n in ["none", "null", "false"]:
		return None
	if n == "standard":
		return StandardScaler()
	if n == "minmax":
		return MinMaxScaler()
	return None


def _clean_params(params: Optional[Dict[str, Any]], drop: Optional[List[str]] = None) -> Dict[str, Any]:
	if not params:
		return {}
	drop_set = set(drop or [])
	return {k: v for k, v in params.items() if k not in drop_set}


def train_linear_regression(
	data: pd.DataFrame,
	target_col: str,
	feature_cols: Optional[List[str]],
	test_size: float,
	random_state: int,
	cv: int,
	model_params: Dict[str, Any],
	grid_search_cfg: Dict[str, Any],
	scaler_name: Optional[str] = None,
) -> Tuple[LinearRegression, Dict]:
	"""Train Linear Regression with optional GridSearchCV."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data[feature_cols].copy() if feature_cols else data.drop(columns=[target_col])
	y = data[target_col]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	scaler = _get_scaler(scaler_name)
	if scaler is not None:
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	base_model = LinearRegression(**_clean_params(model_params))
	use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
	if use_gs:
		param_grid = grid_search_cfg.get("param_grid", {})
		gs_cv = int(grid_search_cfg.get("cv", cv))
		scoring = grid_search_cfg.get("scoring", "r2")
		n_jobs = int(grid_search_cfg.get("n_jobs", -1))
		verbose = int(grid_search_cfg.get("verbose", 0))
		grid_search = GridSearchCV(base_model, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
	else:
		base_model.fit(X_train, y_train)
		best_model = base_model
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	if 'grid_search' in locals():
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


def train_decision_tree(
	data: pd.DataFrame,
	target_col: str,
	feature_cols: Optional[List[str]],
	test_size: float,
	random_state: int,
	cv: int,
	model_params: Dict[str, Any],
	grid_search_cfg: Dict[str, Any],
	scaler_name: Optional[str] = None,
) -> Tuple[DecisionTreeRegressor, Dict]:
	"""Train Decision Tree Regressor with optional GridSearchCV."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")
		
	X = data[feature_cols].copy() if feature_cols else data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	scaler = _get_scaler(scaler_name)
	if scaler is not None:
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	base_model = DecisionTreeRegressor(random_state=random_state, **_clean_params(model_params, ["random_state"]))
	use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
	if use_gs:
		param_grid = grid_search_cfg.get("param_grid", {})
		gs_cv = int(grid_search_cfg.get("cv", cv))
		scoring = grid_search_cfg.get("scoring", "r2")
		n_jobs = int(grid_search_cfg.get("n_jobs", -1))
		verbose = int(grid_search_cfg.get("verbose", 0))
		grid_search = GridSearchCV(base_model, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
	else:
		base_model.fit(X_train, y_train)
		best_model = base_model
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	if 'grid_search' in locals():
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


def train_random_forest(
	data: pd.DataFrame,
	target_col: str,
	feature_cols: Optional[List[str]],
	test_size: float,
	random_state: int,
	cv: int,
	model_params: Dict[str, Any],
	grid_search_cfg: Dict[str, Any],
	scaler_name: Optional[str] = None,
) -> Tuple[RandomForestRegressor, Dict]:
	"""Train Random Forest Regressor with optional GridSearchCV."""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")
		
	X = data[feature_cols].copy() if feature_cols else data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	scaler = _get_scaler(scaler_name)
	if scaler is not None:
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	base_model = RandomForestRegressor(random_state=random_state, **_clean_params(model_params, ["random_state"]))
	use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
	if use_gs:
		param_grid = grid_search_cfg.get("param_grid", {})
		gs_cv = int(grid_search_cfg.get("cv", cv))
		scoring = grid_search_cfg.get("scoring", "r2")
		n_jobs = int(grid_search_cfg.get("n_jobs", -1))
		verbose = int(grid_search_cfg.get("verbose", 0))
		grid_search = GridSearchCV(base_model, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
	else:
		base_model.fit(X_train, y_train)
		best_model = base_model
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	if 'grid_search' in locals():
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


def train_xgboost(
	data: pd.DataFrame,
	target_col: str,
	feature_cols: Optional[List[str]],
	test_size: float,
	random_state: int,
	cv: int,
	model_params: Dict[str, Any],
	grid_search_cfg: Dict[str, Any],
	scaler_name: Optional[str] = None,
) -> Tuple[object, Dict]:
	"""Train XGBoost Regressor with optional GridSearchCV."""
	if XGBRegressor is None:
		raise ImportError("xgboost is not installed in the current environment")

	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data[feature_cols].copy() if feature_cols else data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	scaler = _get_scaler(scaler_name)
	if scaler is not None:
		X_train = scaler.fit_transform(X_train)
		X_test = scaler.transform(X_test)

	base_model = XGBRegressor(random_state=random_state, **_clean_params(model_params, ["random_state"]))
	use_gs = bool(grid_search_cfg.get("enabled", False)) if isinstance(grid_search_cfg, dict) else False
	if use_gs:
		param_grid = grid_search_cfg.get("param_grid", {})
		gs_cv = int(grid_search_cfg.get("cv", cv))
		scoring = grid_search_cfg.get("scoring", "r2")
		n_jobs = int(grid_search_cfg.get("n_jobs", -1))
		verbose = int(grid_search_cfg.get("verbose", 0))
		grid_search = GridSearchCV(base_model, param_grid, cv=gs_cv, scoring=scoring, n_jobs=n_jobs, refit=True, verbose=verbose)
		grid_search.fit(X_train, y_train)
		best_model = grid_search.best_estimator_
	else:
		base_model.fit(X_train, y_train)
		best_model = base_model
	preds = best_model.predict(X_test)
	
	# Calculate metrics
	metrics = _metrics(y_test.to_numpy(), preds)
	cv_metrics = _cross_validation_scores(best_model, X, y, cv)
	metrics.update(cv_metrics)
	if 'grid_search' in locals():
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



