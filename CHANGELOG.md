# CHANGELOG · Actividad 5

Formato basado en *Keep a Changelog*. Se versionan por separado los **datos** (`datos-vX.Y.Z`) y el **proyecto** (`vX.Y.Z`). Cambio mayor = rompe el esquema; menor = transformación nueva; parche = corrección sin cambio de esquema.

---

## Reporte final (v1.0.0 · 18-oct-2026)

**Problema:** clasificación binaria de abandono de clientes (IBM Telco Customer Churn).
**Datos usados:** `datos-v1.1.0` · 7 021 filas · 26.45 % de abandono · SHA-256 `7e1b2bda…67348`.
**Partición:** 5 616 entrenamiento / 1 405 prueba, estratificada, semilla 42.
**Experimento MLflow:** `Actividad5-churn-telco` · 76 runs (1 referencia, 2 GridSearch padres con 24 + 48 candidatos hijos, 1 comparación estadística) · 2 modelos en el Model Registry (`churn_regresion_logistica` v1, `churn_random_forest` v1).

| Métrica (prueba) | Referencia | Regresión logística (umbral 0.31) | Random forest (umbral 0.54) |
|---|---|---|---|
| Accuracy | 0.735 | 0.762 | 0.773 |
| Precision | 0.000 | 0.537 | 0.552 |
| Recall | 0.000 | 0.723 | 0.753 |
| F1 | 0.000 | 0.616 | 0.637 |
| ROC-AUC | 0.500 | 0.840 | 0.842 |
| PR-AUC | 0.265 | 0.637 | 0.657 |

**Comparación estadística:** ROC-AUC sin diferencia significativa (CV 10 × 3: diferencia −0.002, *t* corregida p = 0.42, Wilcoxon p = 0.10; bootstrap en prueba IC 95 % [−0.010, 0.005]; McNemar p = 0.13). F1 con umbral 0.50 favorece al random forest (p = 0.005), pero se explica por la ponderación de clases, no por mejor capacidad de discriminación.

**Decisión:** se recomienda **random forest** para continuar (mejor recall y F1 con umbral calibrado, mismo nivel de discriminación) y se conserva **regresión logística** como alternativa interpretable de igual desempeño en ROC-AUC.

**Deuda técnica pendiente:** versionado de datos sólo con SHA-256 y etiquetas de Git (sin DVC), MLflow local sin servidor compartido, sin pruebas unitarias en CI, sin monitoreo de deriva ni calibración de probabilidades. Acciones propuestas en el reporte técnico (apartado 4).

---

## [v1.0.0] · 18-oct-2026 · Entrega final
### Agregado
- Comparación estadística: CV repetida estratificada 10 × 3 con *t* de remuestreo corregida (Nadeau y Bengio) y Wilcoxon; bootstrap pareado (2 000 réplicas) del ROC-AUC en prueba; McNemar.
- Registro de los modelos ganadores en el Model Registry con firma (`signature`) y ejemplo de entrada.
- Verificación de reproducibilidad: el modelo recargado desde el registro reproduce el ROC-AUC registrado.
- Exportación de `resultados/mlflow_runs.csv`, tablas de métricas y figuras.

## [v0.3.0] · 18-oct-2026 · Ajuste de hiperparámetros
### Cambiado
- Rejilla de regresión logística ampliada de C ∈ {0.01 … 10} a C ∈ {0.001 … 100} (24 combinaciones): en la primera corrida el mejor C quedó en el borde de la rejilla.
- Rejilla de random forest: `min_samples_leaf` de {1, 5, 20} a {1, 10, 30, 60} (48 combinaciones), por la misma razón.
- Umbral de decisión elegido maximizando F1 con predicciones fuera de pliegue del entrenamiento.

## [v0.2.0] · 18-oct-2026 · Entrenamiento y MLflow
### Agregado
- `fuentes/train.py`: Pipeline (StandardScaler + OneHotEncoder + clasificador), GridSearchCV con StratifiedKFold (k = 5) y seis métricas; runs padre/hijo en MLflow; etiquetas de versión de datos, commit, librerías y semilla.
- Línea base `DummyClassifier` (clase mayoritaria).
### Corregido
- MLflow 3.x serializa con *skops*: se declaró `sklearn.tree._tree.Tree` como tipo de confianza para poder registrar el random forest.

## [v0.1.0] · 18-oct-2026 · Estructura y preparación de datos
### Agregado
- Estructura `Actividad5/` (datos, fuentes, README, CHANGELOG).
- `fuentes/datos_prep.py`: descarga, perfil de calidad, reglas R1–R11, validaciones automáticas y `.meta.json`.

---

## Datos

### [datos-v1.1.0] · 18-oct-2026 · Dataset limpio
Archivo `datos/datos_limp/telco_churn_limpio.csv` · SHA-256 `7e1b2bdab10d2c738616bdadad3bfd0c33622649033a3348dd60303895c67348`.
- R1 recorte de espacios (11 celdas de `TotalCharges` con un espacio).
- R3–R4 `TotalCharges` a numérico; 11 blancos con `tenure = 0` → 0 por regla de negocio.
- R5 `SeniorCitizen` 0/1 → No/Yes.
- R6 "No internet service"/"No phone service" → "No" (9 838 celdas en 7 columnas).
- R7 eliminación de 22 duplicados exactos sin `customerID` (7 043 → 7 021 filas).
- R8 marca `anomalia_cargos` en 27 registros (no se eliminan; no se usan como predictor).
- R9 nueva variable `n_servicios` (0–7).
- R10–R11 objetivo a 1/0, tipos finales y eliminación de `customerID`.

### [datos-v1.0.0] · 18-oct-2026 · Dataset original
Archivo `datos/datos_ini/Telco-Customer-Churn.csv` · SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91` · 7 043 × 21 · copia íntegra de la fuente de IBM.
