# Análisis No Supervisado

Este documento describe el proceso y los resultados del análisis no supervisado realizado en el proyecto Rainbow Six ML. El objetivo principal es descubrir patrones ocultos y estructuras subyacentes en los datos sin utilizar etiquetas predefinidas.

### Variables utilizadas del dataset Rainbow Six
- **mapname**: Nombre del mapa donde se desarrolla la partida.
- **operator**: Operador seleccionado por el jugador.
- **primaryweapon**: Arma principal utilizada.

## Objetivos
- Identificar grupos o segmentos dentro de los datos.
- Reducir la dimensionalidad para facilitar la visualización y el análisis.
- Explorar relaciones y tendencias no evidentes.

## Algoritmos Utilizados
- **Clustering:** K-Means, DBSCAN, Agglomerative Clustering.
- **Reducción de dimensionalidad:** PCA (Análisis de Componentes Principales)

## Estructura del Notebook/Archivo
1. Carga y preprocesamiento de datos.
2. Análisis exploratorio inicial.
3. Aplicación de algoritmos de clustering.
4. Reducción de dimensionalidad y visualización.
5. Interpretación de resultados y conclusiones.


## Ejemplo de Uso
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.labels_
```

## Resultados Esperados
- Identificación de segmentos relevantes en los datos.
- Visualizaciones que muestran la distribución y agrupación de los datos.
- Recomendaciones para el uso de los clusters en modelos supervisados o en la toma de decisiones.

## Referencias
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [PCA en Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

---
*Este README puede ser adaptado según los resultados y métodos específicos utilizados en el proyecto.*
