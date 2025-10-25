"""
Model Comparison Pipeline - Consolidates results from classification and regression pipelines.
"""

from kedro.pipeline import Node, Pipeline  # noqa
from proteyecto_ml.pipelines.model_comparison import nodes


def create_pipeline(**kwargs) -> Pipeline:
    """Create model comparison pipeline that consolidates all model results."""
    
    nodes_list = [
        Node(
            func=nodes.consolidate_classification_metrics,
            inputs=[
                "logistic_metrics",
                "knn_metrics", 
                "svm_metrics",
                "dt_clf_metrics",
                "rf_clf_metrics"
            ],
            outputs="classification_consolidated_results",
            name="consolidate_classification_metrics",
        ),
        Node(
            func=nodes.consolidate_regression_metrics,
            inputs=[
                "linear_metrics",
                "multiple_linear_metrics",
                "dt_metrics",
                "rf_metrics", 
                "xgb_metrics",
            ],
            outputs="regression_consolidated_results",
            name="consolidate_regression_metrics",
        ),
        Node(
            func=nodes.create_classification_visualization,
            inputs="classification_consolidated_results",
            outputs="classification_report",
            name="create_classification_visualization",
        ),
        Node(
            func=nodes.create_regression_visualization,
            inputs="regression_consolidated_results", 
            outputs="regression_report",
            name="create_regression_visualization",
        ),
        Node(
            func=nodes.generate_final_report,
            inputs=["classification_consolidated_results", "regression_consolidated_results"],
            outputs="model_comparison_results",
            name="generate_final_report",
        ),
    ]
    
    return Pipeline(nodes_list)
