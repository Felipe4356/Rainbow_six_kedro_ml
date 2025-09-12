# src/pipelines/data_processing/pipeline.py

from kedro.pipeline import Pipeline, node
from .nodes import (
    combinar_raw,
    limpiar_datos,
    eliminar_atipicos,
    preparar_datos_basico,
    crear_kill_death_ratio,
    crear_impact_score,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=combinar_raw,
                inputs=["data_r2s-0", "data_r2s-1", "data_r2s-2"],
                outputs="raw_data",
                name="combinar_raw_node",
            ),
            node(
                func=limpiar_datos,
                inputs="raw_data",
                outputs="clean_data",
                name="limpiar_datos_node",
            ),
            node(
                func=eliminar_atipicos,
                inputs=dict(df="clean_data", columnas="params:columnas_numericas"),
                outputs="data_sin_atipicos",
                name="eliminar_atipicos_node",
            ),
            node(
                func=preparar_datos_basico,
                inputs="data_sin_atipicos",
                outputs="data_preparado",
                name="preparar_datos_node",
            ),
            node(
                func=crear_kill_death_ratio,
                inputs="data_preparado",
                outputs="data_kdr",
                name="crear_kdr_node",
            ),
            node(
                func=crear_impact_score,
                inputs="data_kdr",
                outputs="data_final",
                name="crear_impact_score_node",
            ),
        ]
    )
