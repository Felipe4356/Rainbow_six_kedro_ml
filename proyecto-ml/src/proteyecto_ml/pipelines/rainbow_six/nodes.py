# src/pipelines/data_processing/nodes.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder



def combinar_raw(data0, data1, data2):
    """Une los datasets crudos en un solo DataFrame"""
    combined = pd.concat([data0, data1, data2], ignore_index=True)
    return combined


def limpiar_datos(combined_data: pd.DataFrame) -> pd.DataFrame:
    df = combined_data.copy()
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def eliminar_atipicos(df: pd.DataFrame, columnas: list) -> pd.DataFrame:
    df_clean = df.copy()
    for col in columnas:
        if col not in df_clean.columns:
            print(f"Advertencia: la columna '{col}' no existe en el DataFrame y será ignorada.")
            continue
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
    return df_clean


def preparar_datos_basico(df: pd.DataFrame, label_encoders=None):
    if label_encoders is None:
        label_encoders = {}

    for col in df.select_dtypes(include=["object"]).columns:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        else:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    return df


def crear_kill_death_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "kills" in df.columns and "deaths" in df.columns:
        df["KDR"] = df["kills"] / df["deaths"].replace(0, np.nan)
    return df


def crear_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "kills" in df.columns and "assists" in df.columns:
        df["Impact_Score"] = df["kills"] * 0.7 + df["assists"] * 0.3
    return df
