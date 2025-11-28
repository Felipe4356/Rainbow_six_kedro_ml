"""
This is a boilerplate pipeline 'clustering'
generated using Kedro 1.0.0
"""
# src/<tu_proyecto>/pipelines/clustering/nodes.py

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score



# 1. Preprocesamiento


def preprocess_data(df: pd.DataFrame):
   
    df_limits = df.head(7000)

    X = df_limits[['mapname', 'operator', 'primaryweapon']]

    encoder = OneHotEncoder(sparse_output=True)
    X_encoded = encoder.fit_transform(X)

    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_encoded)

    # Convertir X_scaled a DataFrame para guardar en CSVDataset
    X_scaled_df = pd.DataFrame(X_scaled.toarray() if hasattr(X_scaled, "toarray") else X_scaled)
    return {
        "X_scaled": X_scaled_df,
        "encoder": encoder,
        "scaler": scaler
    }


# 2. PCA (85% varianza)


def apply_pca(X_scaled: pd.DataFrame, explained_var: float = 0.85):
    pca = PCA(n_components=explained_var)
    X_pca = pca.fit_transform(X_scaled)
    return {
        "X_pca": X_pca,
        "pca": pca
    }



# 3. Clustering: DBSCAN


def apply_dbscan(X_pca: pd.DataFrame, eps=0.5, min_samples=10):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_pca)
    return {
        "labels_dbscan": labels,
        "X_pca": X_pca
    }



# 4. Clustering: K-Mean

def apply_kmeans(X_pca: pd.DataFrame, n_clusters=5):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_pca)
    return {
        "labels_kmeans": labels,
        "X_pca": X_pca
    }



# 5. Clustering: Jerárquico (Agglomerative)


def apply_hierarchical(X_pca: pd.DataFrame, n_clusters=5, linkage="ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_pca)
    return {
        "labels_hierarchical": labels,
        "X_pca": X_pca
    }


# 6. Métricas para todos los algoritmos


def compute_clustering_metrics(X_pca, labels):
    """Devuelve un dict con las 3 métricas más usadas."""

    if len(set(labels)) <= 1:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None
        }

    return {
        "silhouette": silhouette_score(X_pca, labels),
        "davies_bouldin": davies_bouldin_score(X_pca, labels),
        "calinski_harabasz": calinski_harabasz_score(X_pca, labels)
    }


def metrics_all(X_pca: pd.DataFrame, dbscan_labels: pd.DataFrame, kmeans_labels: pd.DataFrame, hierarchical_labels: pd.DataFrame):
    return {
        "dbscan": compute_clustering_metrics(X_pca, dbscan_labels["cluster"].values),
        "kmeans": compute_clustering_metrics(X_pca, kmeans_labels["cluster"].values),
        "hierarchical": compute_clustering_metrics(X_pca, hierarchical_labels["cluster"].values)
    }


# Ejemplo: Detector de anomalías con Isolation Forest
from sklearn.ensemble import IsolationForest

def detect_anomalies_isolation_forest(X, contamination=0.05, random_state=42):
    """
    Aplica Isolation Forest para detectar anomalías en los datos.
    Args:
        X (pd.DataFrame): Datos de entrada.
        contamination (float): Proporción esperada de anomalías.
        random_state (int): Semilla para reproducibilidad.
    Returns:
        pd.Series: Etiquetas de anomalía (1=anomalía, 0=normal).
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    labels = iso_forest.fit_predict(X)
    # Isolation Forest devuelve -1 para anomalía, 1 para normal
    return pd.Series((labels == -1).astype(int), index=X.index, name="anomaly_iforest")