

## Pipeline completo (todos los modelos)

- este comando ejecutara todo los pipeline
```powershell
kedro run
```


## Pipelines específicos


- otra forma de ejecutar los pipeline seria invididual 
```powershell
kedro run --pipeline=rainbow_six           # Preparación de datos
kedro run --pipeline=modelo_clasificacion  # 5 modelos clasificación
kedro run --pipeline=modelo_regresion      # 5 modelos regresión  
kedro run --pipeline=model_comparison      # Comparación final
kedro run --pipeline unsupervised_lear     # 3 modelos no supervisado

```
### 📊 Comandos de Información

```bash
# Ver información del proyecto
kedro info

# Listar todos los pipelines disponibles
kedro registry list

# Ver datasets configurados
kedro catalog list

# Verificar configuración
kedro config list
```
