import os
import sys
import pandas as pd
import numpy as np

# Ensure the package `proteyecto_ml` can be imported when tests run from repo root
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))
if SRC_PATH not in sys.path:
	sys.path.insert(0, SRC_PATH)

from proteyecto_ml.pipelines.modelo_regresion import nodes, pipeline


def _make_dummy_data(n_rows: int = 100) -> pd.DataFrame:
	rng = np.random.RandomState(42)
	return pd.DataFrame({
		"gamemode": rng.randint(0, 3, size=n_rows),
		"winrole": rng.randint(0, 2, size=n_rows),
		"endroundreason": rng.randint(0, 4, size=n_rows),
		"roundduration": rng.normal(60, 10, size=n_rows),
		"isdead": rng.randint(0, 2, size=n_rows),
		"nbkills": rng.poisson(1.5, size=n_rows),
		"impact_score": rng.normal(0.5, 0.2, size=n_rows),
	})


def test_pipeline_nodes_run():
	data = _make_dummy_data(50)

	# Test linear regression trainer
	model, metrics = nodes.train_linear_regression(data)
	assert hasattr(model, "predict")
	assert isinstance(metrics, dict)
	assert "r2" in metrics
	# model file should be present
	assert "model_path" in metrics
	assert os.path.exists(metrics["model_path"]) is True

	# Decision tree
	dt_model, dt_metrics = nodes.train_decision_tree(data)
	assert hasattr(dt_model, "predict")
	assert isinstance(dt_metrics, dict)
	assert "model_path" in dt_metrics
	assert os.path.exists(dt_metrics["model_path"]) is True

	# Random forest
	rf_model, rf_metrics = nodes.train_random_forest(data)
	assert hasattr(rf_model, "predict")
	assert isinstance(rf_metrics, dict)
	assert "model_path" in rf_metrics
	assert os.path.exists(rf_metrics["model_path"]) is True

	# XGBoost: if not available the function raises ImportError; otherwise it should work
	try:
		xgb_model, xgb_metrics = nodes.train_xgboost(data)
		assert hasattr(xgb_model, "predict")
		assert isinstance(xgb_metrics, dict)
		assert "model_path" in xgb_metrics
		assert os.path.exists(xgb_metrics["model_path"]) is True
	except ImportError:
		# xgboost isn't installed in CI environment - that's acceptable
		pass


def test_create_pipeline_structure():
	pl = pipeline.create_pipeline()
	# basic sanity: should contain nodes
	assert len(pl.nodes) >= 3
