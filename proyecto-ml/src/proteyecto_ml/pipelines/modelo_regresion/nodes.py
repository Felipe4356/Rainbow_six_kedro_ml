"""
Nodes for the `modelo_regresion` pipeline.

These nodes implement training and evaluation helper functions for the
regression models that were prototyped in the notebooks. The functions are
kept small and focused so they can be composed by the pipeline.

Each train_* function receives a DataFrame X and Series y (or a single
DataFrame with features) and returns a trained model plus a dict with
evaluation metrics. The evaluate function can be used separately if needed.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
try:
	import xgboost as xgb
	from xgboost import XGBRegressor
except Exception:  # pragma: no cover - xgboost is optional in some environments
	xgb = None
	XGBRegressor = None
import os
import pickle


MODELS_DIR = os.path.abspath(os.path.join(os.getcwd(), "data", "06_models"))


def _ensure_models_dir() -> None:
	os.makedirs(MODELS_DIR, exist_ok=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
	mse = mean_squared_error(y_true, y_pred)
	return {
		"mse": float(mse),
		"rmse": float(np.sqrt(mse)),
		"mae": float(mean_absolute_error(y_true, y_pred)),
		"r2": float(r2_score(y_true, y_pred)),
	}


def train_linear_regression(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, random_state: int = 42) -> Tuple[LinearRegression, Dict]:
	"""Train a simple LinearRegression on the provided DataFrame.

	Expects that the DataFrame contains the target column. Uses all other
	columns as features.
	Returns the trained model and a metrics dict evaluated on the test set.
	"""
	if target_col not in data.columns:
		raise KeyError(f"Target column '{target_col}' not found in data")

	X = data.drop(columns=[target_col])
	y = data[target_col]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	model = LinearRegression()
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	metrics = _metrics(y_test.to_numpy(), preds)
	# Persist model
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "linear_model.pkl")
	with open(path, "wb") as fp:
		pickle.dump(model, fp)
	metrics["model_path"] = path
	return model, metrics


def train_decision_tree(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, random_state: int = 42) -> Tuple[DecisionTreeRegressor, Dict]:
	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	model = DecisionTreeRegressor(random_state=random_state)
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	metrics = _metrics(y_test.to_numpy(), preds)
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "dt_model.pkl")
	with open(path, "wb") as fp:
		pickle.dump(model, fp)
	metrics["model_path"] = path
	return model, metrics


def train_random_forest(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, random_state: int = 42) -> Tuple[RandomForestRegressor, Dict]:
	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	metrics = _metrics(y_test.to_numpy(), preds)
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "rf_model.pkl")
	with open(path, "wb") as fp:
		pickle.dump(model, fp)
	metrics["model_path"] = path
	return model, metrics


def train_xgboost(data: pd.DataFrame, target_col: str = "impact_score", test_size: float = 0.2, random_state: int = 42) -> Tuple[object, Dict]:
	"""Train an XGBoost regressor if xgboost is available.

	If XGBoost is not installed the function will raise ImportError. The return
	follows the same (model, metrics) contract as the other trainers.
	"""
	if XGBRegressor is None:
		raise ImportError("xgboost is not installed in the current environment")

	X = data.drop(columns=[target_col])
	y = data[target_col]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
	model = XGBRegressor(n_estimators=100, max_depth=6, random_state=random_state, n_jobs=-1)
	model.fit(X_train, y_train)
	preds = model.predict(X_test)
	metrics = _metrics(y_test.to_numpy(), preds)
	_ensure_models_dir()
	path = os.path.join(MODELS_DIR, "xgb_model.pkl")
	with open(path, "wb") as fp:
		pickle.dump(model, fp)
	metrics["model_path"] = path
	return model, metrics


