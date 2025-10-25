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
            inputs=[
                "data_final",
                "params:classification_target_col",
                "params:classification_feature_cols",
                "params:classification_test_size",
                "params:classification_random_state",
                "params:classification_cv",
                "params:classification_models.logistic_regression.params",
                "params:classification_models.logistic_regression.grid_search",
                "params:classification_scaler",
            ],
            outputs=["logistic_model", "logistic_metrics"],
            name="train_logistic_regression",
        ),
        Node(
            func=nodes.train_knn_classifier,
            inputs=[
                "data_final",
                "params:classification_target_col",
                "params:classification_feature_cols",
                "params:classification_test_size",
                "params:classification_random_state",
                "params:classification_cv",
                "params:classification_models.knn.params",
                "params:classification_models.knn.grid_search",
                "params:classification_scaler",
            ], 
            outputs=["knn_model", "knn_metrics"],
            name="train_knn_classifier",
        ),
        Node(
            func=nodes.train_svm_classifier,
            inputs=[
                "data_final",
                "params:classification_target_col",
                "params:classification_feature_cols",
                "params:classification_test_size",
                "params:classification_random_state",
                "params:classification_cv",
                "params:classification_models.svc.params",
                "params:classification_models.svc.grid_search",
                "params:classification_scaler",
            ],
            outputs=["svm_model", "svm_metrics"],
            name="train_svm_classifier",
        ),
        Node(
            func=nodes.train_decision_tree_classifier,
            inputs=[
                "data_final",
                "params:classification_target_col",
                "params:classification_feature_cols",
                "params:classification_test_size",
                "params:classification_random_state",
                "params:classification_cv",
                "params:classification_models.decision_tree.params",
                "params:classification_models.decision_tree.grid_search",
                "params:classification_scaler",
            ],
            outputs=["dt_clf_model", "dt_clf_metrics"],
            name="train_decision_tree_classifier",
        ),
        Node(
            func=nodes.train_random_forest_classifier,
            inputs=[
                "data_final",
                "params:classification_target_col",
                "params:classification_feature_cols",
                "params:classification_test_size",
                "params:classification_random_state",
                "params:classification_cv",
                "params:classification_models.random_forest.params",
                "params:classification_models.random_forest.grid_search",
                "params:classification_scaler",
            ],
            outputs=["rf_clf_model", "rf_clf_metrics"],
            name="train_random_forest_classifier",
        ),
    ]
    
    return Pipeline(nodes_list)
