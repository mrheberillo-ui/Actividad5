# Actividad 5 · Preparación de datos, entrenamiento y registro de experimentos con MLflow

**MSCA2102 · Universidad Tecmilenio**

Proyecto aplicado de aprendizaje automático: se predice si un cliente de telecomunicaciones abandona el servicio (*churn*), un problema de **clasificación binaria**. Se comparan **regresión logística** y **random forest**, ajustados con `GridSearchCV` y validación cruzada estratificada de 5 pliegues, y cada experimento queda registrado en **MLflow** (parámetros, métricas, artefactos, modelos y Model Registry).

## Estructura

```
Actividad5/
│── datos/
│   ├── datos_ini/                         # datos originales (v1.0.0, sin modificar)
│   │   └── Telco-Customer-Churn.csv
│   └── datos_limp/                        # datos limpios (v1.1.0)
│       ├── telco_churn_limpio.csv
│       └── telco_churn_limpio.meta.json   # versión, SHA-256 y bitácora de limpieza
│── fuentes/
│   ├── entrena.ipynb                      # notebook de Colab (entrenamiento y evaluación)
│   ├── datos_prep.py                      # obtención, limpieza, validación y versionado
│   └── train.py                           # entrenamiento, GridSearchCV y registro en MLflow
│── resultados/                            # tablas y figuras generadas por la corrida
│── evidencias/                            # capturas del panel de MLflow
│── requirements.txt
│── README.md                              # este archivo
└── CHANGELOG.md                           # versiones y reporte final
```

`mlflow.db` (backend de seguimiento) y `mlartifacts/` (artefactos y modelos) se crean al ejecutar y están en `.gitignore`; la tabla `resultados/mlflow_runs.csv` y las capturas de `evidencias/` dejan constancia de lo registrado.

## Dataset

| Concepto | Valor |
|---|---|
| Nombre | IBM Telco Customer Churn |
| Fuente | https://github.com/IBM/telco-customer-churn-on-icp4d (archivo `data/Telco-Customer-Churn.csv`) |
| Tamaño original | 7 043 clientes × 21 columnas |
| Objetivo | `Churn` (Yes/No → 1/0); 26.5 % de abandono |
| SHA-256 original | `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` |
| SHA-256 limpio (v1.1.0) | `7e1b2bdab10d2c738616bdadad3bfd0c33622649033a3348dd60303895c67348` |

El diccionario de datos y los supuestos están en el reporte técnico y en los comentarios de `fuentes/datos_prep.py`.

## Reproducción

### Opción A · Google Colab (recomendada)
1. Abrir `fuentes/entrena.ipynb` en Colab.
2. `Entorno de ejecución → Ejecutar todo` (CPU basta; 10–15 minutos).
3. El cuaderno instala `mlflow==3.17.1`, recrea la estructura completa en `/content/Actividad5`, descarga los datos, limpia, entrena, registra en MLflow, abre el panel de MLflow en una pestaña nueva y descarga `Actividad5.zip`.

### Opción B · Local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python fuentes/datos_prep.py        # genera datos/datos_limp/ (v1.1.0)
python fuentes/train.py             # entrena, compara y registra en MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Abrir http://127.0.0.1:5000 → experimento **Actividad5-churn-telco**.

### Verificar que los datos son la misma versión
```bash
sha256sum datos/datos_ini/Telco-Customer-Churn.csv datos/datos_limp/telco_churn_limpio.csv
```
Los valores deben coincidir con la tabla de arriba y con las etiquetas `dataset_ini_sha256` y `dataset_sha256` de cada run en MLflow.

## Inicializar Git y publicar en GitHub

```bash
git init
git add .
git commit -m "Actividad 5: datos v1.1.0, entrenamiento y registro en MLflow"
git tag -a datos-v1.0.0 -m "Datos originales IBM Telco Churn"
git tag -a datos-v1.1.0 -m "Datos limpios (reglas R1-R11)"
git branch -M main
git remote add origin https://github.com/<usuario>/<repositorio>.git
git push -u origin main --tags
```
Cada run de MLflow guarda el commit en la etiqueta `git_commit`; si se ejecuta antes del primer commit aparece `sin-git`.

## Decisiones principales

| Decisión | Justificación |
|---|---|
| `TotalCharges` en blanco → 0 cuando `tenure = 0` | Los 11 blancos son clientes nuevos sin ciclo facturado: es una regla de negocio, no un faltante aleatorio; imputar la mediana inventaría cargos. |
| "No internet service" / "No phone service" → "No" | La ausencia del servicio base ya está en `InternetService` y `PhoneService`; evita colinealidad perfecta tras el one-hot. |
| Eliminar 22 duplicados exactos (sin `customerID`) | Perfiles idénticos con la misma etiqueta podrían caer en entrenamiento y prueba a la vez e inflar las métricas. |
| Marcar (no eliminar) 27 registros con cargos inconsistentes | Desviación > 25 % entre `TotalCharges` y `tenure × MonthlyCharges`; son clientes recientes con cambio de tarifa, no errores evidentes. |
| Escalado y one-hot dentro del `Pipeline` | Se ajustan sólo con el entrenamiento de cada pliegue: no hay fuga de información hacia validación o prueba. |
| Selección por ROC-AUC | Métrica independiente del umbral y robusta al desbalance 74/26; el umbral operativo se elige aparte, maximizando F1 con predicciones fuera de pliegue del entrenamiento. |
| Partición 80/20 estratificada, semilla 42 | El conjunto de prueba (1 405 clientes) no interviene en ninguna decisión de ajuste. |

## Resultados (resumen)

| Modelo | Mejores hiperparámetros | ROC-AUC CV (5 pliegues) | ROC-AUC prueba | F1 prueba (umbral elegido) |
|---|---|---|---|---|
| Referencia (clase mayoritaria) | — | 0.500 | 0.500 | 0.000 |
| Regresión logística | C = 10, L2, sin ponderar clases | 0.846 ± 0.016 | 0.840 | 0.616 (umbral 0.31) |
| Random forest | 300 árboles, sin límite de profundidad, `min_samples_leaf` = 30, `max_features` = sqrt, `balanced_subsample` | 0.849 ± 0.018 | 0.842 | 0.637 (umbral 0.54) |

La diferencia de ROC-AUC no es estadísticamente significativa (CV repetida 10 × 3: *t* corregida p = 0.42; bootstrap en prueba: IC 95 % de la diferencia −0.010 a 0.005; McNemar p = 0.13). Detalle en el reporte técnico y en `CHANGELOG.md`.

## Entorno de la corrida documentada

Python 3.13.16 · numpy 2.5.3 · pandas 3.0.5 · scikit-learn 1.9.1 · mlflow 3.17.1. En Colab las versiones de numpy, pandas y scikit-learn pueden ser otras; las métricas pueden variar en el tercer decimal. `train.py` se adapta a scikit-learn < 1.8 (usa `penalty`) y ≥ 1.8 (usa `l1_ratio`).
