
import pandas as pd
from kedro.pipeline import Pipeline, node
from .nodes import (
    preprocess_data, apply_pca,
    apply_dbscan, apply_kmeans, apply_hierarchical,
    metrics_all
)

def create_pipeline(**kwargs):
    return Pipeline([

        node(
            func=lambda df: preprocess_data(df)["X_scaled"],
            inputs="data_final",
            outputs="preprocessed_data",
            name="preprocessing_node"
        ),

        node(
            func=lambda X_scaled: pd.DataFrame(apply_pca(X_scaled)["X_pca"]),
            inputs="preprocessed_data",
            outputs="pca_data",
            name="pca_node"
        ),

        node(
            func=lambda X_pca: pd.DataFrame(apply_dbscan(X_pca)["labels_dbscan"], columns=["cluster"]),
            inputs="pca_data",
            outputs="dbscan_result",
            name="dbscan_node"
        ),

        node(
            func=lambda X_pca: pd.DataFrame(apply_kmeans(X_pca)["labels_kmeans"], columns=["cluster"]),
            inputs="pca_data",
            outputs="kmeans_result",
            name="kmeans_node"
        ),

        node(
            func=lambda X_pca: pd.DataFrame(apply_hierarchical(X_pca)["labels_hierarchical"], columns=["cluster"]),
            inputs="pca_data",
            outputs="hierarchical_result",
            name="hierarchical_node"
        ),

        node(
            func=metrics_all,
            inputs=["pca_data", "dbscan_result", "kmeans_result", "hierarchical_result"],
            outputs="clustering_metrics",
            name="metrics_node"
        ),

    ])
