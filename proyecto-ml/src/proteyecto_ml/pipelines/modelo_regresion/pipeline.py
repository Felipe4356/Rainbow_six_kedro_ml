"""
This is a boilerplate pipeline 'modelo_regresion'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from proteyecto_ml.pipelines.modelo_regresion import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the regression pipeline with 5+ models and GridSearchCV."""
    nodes_list = [
        Node(
            func=nodes.train_linear_regression,
            inputs=[
                "data_final",
                "params:regression_target_col",
                "params:regression_feature_cols",
                "params:regression_test_size",
                "params:regression_random_state",
                "params:regression_cv",
                "params:regression_models.linear_regression.params",
                "params:regression_models.linear_regression.grid_search",
                "params:regression_scaler",
            ],
            outputs=["linear_model", "linear_metrics"],
            name="train_linear_regression",
        ),
        Node(
            func=nodes.train_decision_tree,
            inputs=[
                "data_final",
                "params:regression_target_col",
                "params:regression_feature_cols",
                "params:regression_test_size",
                "params:regression_random_state",
                "params:regression_cv",
                "params:regression_models.decision_tree.params",
                "params:regression_models.decision_tree.grid_search",
                "params:regression_scaler",
            ],
            outputs=["dt_model", "dt_metrics"],
            name="train_decision_tree",
        ),
        Node(
            func=nodes.train_random_forest,
            inputs=[
                "data_final",
                "params:regression_target_col",
                "params:regression_feature_cols",
                "params:regression_test_size",
                "params:regression_random_state",
                "params:regression_cv",
                "params:regression_models.random_forest.params",
                "params:regression_models.random_forest.grid_search",
                "params:regression_scaler",
            ],
            outputs=["rf_model", "rf_metrics"],
            name="train_random_forest",
        ),
        # Multiple Linear Regression (using the existing linear trainer but kept as a separate pipeline node)
        Node(
            func=nodes.train_linear_regression,
            inputs=[
                "data_final",
                "params:regression_target_col",
                "params:regression_feature_cols",
                "params:regression_test_size",
                "params:regression_random_state",
                "params:regression_cv",
                "params:regression_models.linear_regression.params",
                "params:regression_models.linear_regression.grid_search",
                "params:regression_scaler",
            ],
            outputs=["multiple_linear_model", "multiple_linear_metrics"],
            name="train_multiple_linear_regression",
        ),
        Node(
            func=nodes.train_xgboost,
            inputs=[
                "data_final",
                "params:regression_target_col",
                "params:regression_feature_cols",
                "params:regression_test_size",
                "params:regression_random_state",
                "params:regression_cv",
                "params:regression_models.xgboost.params",
                "params:regression_models.xgboost.grid_search",
                "params:regression_scaler",
            ],
            outputs=["xgb_model", "xgb_metrics"],
            name="train_xgboost",
        ),
    ]

    return Pipeline(nodes_list)
