"""
This is a boilerplate pipeline 'modelo_clasificacion'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from proteyecto_ml.pipelines.modelo_clasificacion import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create the classification pipeline with 5+ models and GridSearchCV."""
    nodes_list = [
        Node(
            func=nodes.train_logistic_regression,
            inputs="data_final",
            outputs=["logistic_model", "logistic_metrics"],
            name="train_logistic_regression",
        ),
        Node(
            func=nodes.train_knn_classifier,
            inputs="data_final", 
            outputs=["knn_model", "knn_metrics"],
            name="train_knn_classifier",
        ),
        Node(
            func=nodes.train_svm_classifier,
            inputs="data_final",
            outputs=["svm_model", "svm_metrics"],
            name="train_svm_classifier",
        ),
        Node(
            func=nodes.train_decision_tree_classifier,
            inputs="data_final",
            outputs=["dt_clf_model", "dt_clf_metrics"],
            name="train_decision_tree_classifier",
        ),
        Node(
            func=nodes.train_random_forest_classifier,
            inputs="data_final",
            outputs=["rf_clf_model", "rf_clf_metrics"],
            name="train_random_forest_classifier",
        ),
    ]
    
    return Pipeline(nodes_list)
