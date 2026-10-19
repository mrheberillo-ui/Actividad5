# Fuente de los datos originales (v1.0.0)

- Dataset: IBM Telco Customer Churn (datos de ejemplo publicados por IBM).
- Repositorio: https://github.com/IBM/telco-customer-churn-on-icp4d
- URL de descarga: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
- Fecha de obtención: 18 de octubre de 2026
- SHA-256: 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91
- Regla: este archivo no se modifica. Toda transformación vive en fuentes/datos_prep.py y produce datos/datos_limp/.
- Para volver a obtenerlo: `python fuentes/datos_prep.py` (lo descarga si falta).
