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
            inputs="data_final",
            outputs=["linear_model", "linear_metrics"],
            name="train_linear_regression",
        ),
        Node(
            func=nodes.train_decision_tree,
            inputs="data_final",
            outputs=["dt_model", "dt_metrics"],
            name="train_decision_tree",
        ),
        Node(
            func=nodes.train_random_forest,
            inputs="data_final",
            outputs=["rf_model", "rf_metrics"],
            name="train_random_forest",
        ),
        Node(
            func=nodes.train_xgboost,
            inputs="data_final",
            outputs=["xgb_model", "xgb_metrics"],
            name="train_xgboost",
        ),
        Node(
            func=nodes.train_svr,
            inputs="data_final",
            outputs=["svr_model", "svr_metrics"],
            name="train_svr",
        ),
    ]

    return Pipeline(nodes_list)
