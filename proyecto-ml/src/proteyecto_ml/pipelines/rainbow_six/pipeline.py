import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# === Tus funciones ===
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    filas_iniciales = df.shape[0]
    df = df.dropna()
    filas_sin_nulos = df.shape[0]
    df = df.drop_duplicates()
    filas_finales = df.shape[0]
    return df

def crear_kill_death_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df["kill_death_ratio"] = df["nbkills"] / (df["isdead"] + 1)
    return df

def crear_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    df["impact_score"] = df["nbkills"] * df["haswon"]
    return df

# === Pipeline con FunctionTransformer ===
pipeline = Pipeline([
    ("limpieza", FunctionTransformer(limpiar_datos, validate=False)),
    ("feature_kd", FunctionTransformer(crear_kill_death_ratio, validate=False)),
    ("feature_impact", FunctionTransformer(crear_impact_score, validate=False)),
])

# === Ejecutar pipeline ===
combined_data = pipeline.fit_transform(combined_data)

print("✅ Pipeline ejecutado")
print(combined_data.head())



