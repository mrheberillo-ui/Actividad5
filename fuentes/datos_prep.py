"""
datos_prep.py — Obtención, validación, limpieza y versionado del dataset
IBM Telco Customer Churn.

Uso desde la terminal (raíz de Actividad5/):
    python fuentes/datos_prep.py

Genera:
    datos/datos_ini/Telco-Customer-Churn.csv      (copia íntegra de la fuente)
    datos/datos_limp/telco_churn_limpio.csv       (dataset limpio, v1.1.0)
    datos/datos_limp/telco_churn_limpio.meta.json (versión, huellas SHA-256 y bitácora)

El escalado y la codificación one-hot NO se guardan en el CSV limpio: se
aplican dentro del Pipeline de scikit-learn (train.py) para que se ajusten
sólo con los datos de entrenamiento de cada pliegue y no haya fuga de
información.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constantes del dataset
# ---------------------------------------------------------------------------
URL_FUENTE = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
NOMBRE_INI = "Telco-Customer-Churn.csv"
NOMBRE_LIMP = "telco_churn_limpio.csv"
VERSION_DATOS = "1.1.0"

ID = "customerID"
OBJETIVO = "Churn"

SERVICIOS_INTERNET = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
CATEGORICAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", *SERVICIOS_INTERNET,
    "Contract", "PaperlessBilling", "PaymentMethod",
]
NUMERICAS = ["tenure", "MonthlyCharges", "TotalCharges", "n_servicios"]

# Dominio válido de cada columna categórica en los datos originales
DOMINIOS = {
    "gender": {"Female", "Male"},
    "SeniorCitizen": {0, 1},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    **{c: {"Yes", "No", "No internet service"} for c in SERVICIOS_INTERNET},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    },
    "Churn": {"Yes", "No"},
}

# Tolerancia para la regla de consistencia TotalCharges ≈ tenure × MonthlyCharges
TOLERANCIA_CARGOS = 0.25


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def sha256(ruta: str | Path) -> str:
    """Huella SHA-256 de un archivo (identifica la versión exacta de los datos)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar_datos(destino: str | Path, url: str = URL_FUENTE) -> Path:
    """Descarga el CSV original si no existe ya en `destino`."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def cargar_datos_ini(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV original sin conversiones implícitas (todo como texto salvo lo numérico limpio)."""
    return pd.read_csv(ruta, dtype={ID: str, "TotalCharges": str})


# ---------------------------------------------------------------------------
# Diagnóstico de calidad
# ---------------------------------------------------------------------------
def perfil_calidad(df: pd.DataFrame) -> dict:
    """Resumen de calidad del dataset original: tipos, nulos, blancos,
    duplicados, valores fuera de dominio y desbalance de la variable objetivo."""
    tc_texto = df["TotalCharges"].astype(str).str.strip()
    tc = pd.to_numeric(tc_texto.replace("", np.nan), errors="coerce")
    sin_id = df.drop(columns=[ID])
    fuera_dominio = {}
    for col, dominio in DOMINIOS.items():
        n = int((~df[col].isin(dominio)).sum())
        if n:
            fuera_dominio[col] = n
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "nulos_explicitos": int(df.isna().sum().sum()),
        "totalcharges_en_blanco": int((tc_texto == "").sum()),
        "totalcharges_no_numerico": int(tc.isna().sum()),
        "ids_duplicados": int(df[ID].duplicated().sum()),
        "duplicados_exactos_sin_id": int(sin_id.duplicated().sum()),
        "duplicados_atributos_sin_objetivo": int(sin_id.drop(columns=[OBJETIVO]).duplicated().sum()),
        "valores_fuera_de_dominio": fuera_dominio,
        "tasa_churn": round(float((df[OBJETIVO] == "Yes").mean()), 4),
    }


# ---------------------------------------------------------------------------
# Limpieza y estandarización
# ---------------------------------------------------------------------------
def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica las reglas de limpieza en orden y devuelve (df_limpio, bitácora).

    Reglas:
      R1  Recortar espacios en todas las columnas de texto.
      R2  Validar dominios de las categóricas (falla si aparece un valor no esperado).
      R3  TotalCharges de texto a numérico; los blancos se convierten en nulos.
      R4  Regla de negocio: si tenure == 0 el cliente aún no tiene ciclo facturado,
          por lo que TotalCharges = 0 (imputación por regla, no por la mediana).
      R5  SeniorCitizen 0/1 -> "No"/"Yes" para tratarla igual que las demás binarias.
      R6  "No internet service" / "No phone service" -> "No": la ausencia del servicio
          base ya está en InternetService / PhoneService (evita colinealidad perfecta).
      R7  Eliminar duplicados exactos (todas las columnas salvo customerID), conservando
          el primero; evita que un mismo perfil quede en entrenamiento y prueba.
      R8  Marcar (sin eliminar) registros cuyo TotalCharges se aleja más de ±25 % de
          tenure × MonthlyCharges: pueden ser cambios de tarifa legítimos.
      R9  Variable derivada n_servicios (servicios adicionales contratados, 0-7).
      R10 Objetivo Churn "Yes"/"No" -> 1/0; tipos finales: category / int / float.
      R11 Eliminar customerID (identificador sin valor predictivo).
    """
    bitacora: list[dict] = []
    d = df.copy()

    def registrar(regla: str, descripcion: str, filas_afectadas: int) -> None:
        bitacora.append({"regla": regla, "descripcion": descripcion,
                         "filas_afectadas": int(filas_afectadas), "filas_resultantes": int(len(d))})

    # R1
    texto = [c for c in d.columns if d[c].dtype == object or str(d[c].dtype) in ("string", "str")]
    cambios = 0
    for c in texto:
        s = d[c].astype(str)
        cambios += int((s != s.str.strip()).sum())
        d[c] = s.str.strip()
    registrar("R1", "Recorte de espacios en columnas de texto", cambios)

    # R2
    for col, dominio in DOMINIOS.items():
        valores = d[col] if col != "SeniorCitizen" else d[col].astype(int)
        invalidos = set(valores.unique()) - dominio
        if invalidos:
            raise ValueError(f"Valores fuera de dominio en {col}: {invalidos}")
    registrar("R2", "Validación de dominios categóricos (sin valores inválidos)", 0)

    # R3
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"].replace("", np.nan), errors="coerce")
    nulos_tc = int(d["TotalCharges"].isna().sum())
    registrar("R3", "TotalCharges convertido a numérico; blancos -> nulo", nulos_tc)

    # R4
    mascara = d["TotalCharges"].isna() & (d["tenure"] == 0)
    d.loc[mascara, "TotalCharges"] = 0.0
    if d["TotalCharges"].isna().any():
        raise ValueError("Quedan nulos en TotalCharges con tenure > 0: revisar la fuente.")
    registrar("R4", "Regla de negocio: tenure = 0 => TotalCharges = 0", int(mascara.sum()))

    # R5
    d["SeniorCitizen"] = d["SeniorCitizen"].astype(int).map({0: "No", 1: "Yes"})
    registrar("R5", "SeniorCitizen 0/1 -> No/Yes", len(d))

    # R6
    n6 = 0
    for c in SERVICIOS_INTERNET + ["MultipleLines"]:
        m = d[c].isin(["No internet service", "No phone service"])
        n6 += int(m.sum())
        d.loc[m, c] = "No"
    registrar("R6", "'No internet/phone service' -> 'No' (7 columnas)", n6)

    # R7
    antes = len(d)
    d = d.loc[~d.drop(columns=[ID]).duplicated(keep="first")].reset_index(drop=True)
    registrar("R7", "Eliminación de duplicados exactos (sin considerar customerID)", antes - len(d))

    # R8
    esperado = d["tenure"] * d["MonthlyCharges"]
    razon = np.where(esperado > 0, d["TotalCharges"] / esperado.replace(0, np.nan), 1.0)
    d["anomalia_cargos"] = (np.abs(razon - 1) > TOLERANCIA_CARGOS).astype(int)
    registrar("R8", f"Marca de inconsistencia de cargos (|razón-1| > {TOLERANCIA_CARGOS})",
              int(d["anomalia_cargos"].sum()))

    # R9
    adicionales = SERVICIOS_INTERNET + ["MultipleLines"]
    d["n_servicios"] = (d[adicionales] == "Yes").sum(axis=1).astype(int)
    registrar("R9", "Variable derivada n_servicios (0-7)", len(d))

    # R10
    d[OBJETIVO] = (d[OBJETIVO] == "Yes").astype(int)
    for c in CATEGORICAS:
        d[c] = d[c].astype("category")
    d["tenure"] = d["tenure"].astype(int)
    d["MonthlyCharges"] = d["MonthlyCharges"].astype(float)
    d["TotalCharges"] = d["TotalCharges"].astype(float)
    registrar("R10", "Objetivo a 1/0 y tipos finales (category, int, float)", len(d))

    # R11
    d = d.drop(columns=[ID])
    registrar("R11", "Eliminación de customerID", 0)

    columnas = CATEGORICAS + NUMERICAS + ["anomalia_cargos", OBJETIVO]
    return d[columnas], bitacora


def validar_limpio(d: pd.DataFrame) -> None:
    """Pruebas mínimas de calidad sobre el dataset limpio (fallan con AssertionError)."""
    assert d.isna().sum().sum() == 0, "Hay nulos en el dataset limpio"
    assert d.duplicated().sum() == 0, "Hay duplicados exactos en el dataset limpio"
    assert set(d[OBJETIVO].unique()) <= {0, 1}, "Objetivo no binario"
    assert (d["tenure"] >= 0).all() and (d["MonthlyCharges"] > 0).all()
    assert (d["TotalCharges"] >= 0).all()
    for c in SERVICIOS_INTERNET + ["MultipleLines"]:
        assert set(d[c].astype(str).unique()) <= {"Yes", "No"}, f"Categoría residual en {c}"


# ---------------------------------------------------------------------------
# Versionado
# ---------------------------------------------------------------------------
def guardar_limpio(d: pd.DataFrame, bitacora: list[dict], ruta_ini: Path,
                   dir_limp: Path, perfil: dict) -> dict:
    """Guarda el CSV limpio y un archivo .meta.json con versión, huellas y bitácora."""
    dir_limp.mkdir(parents=True, exist_ok=True)
    ruta_csv = dir_limp / NOMBRE_LIMP
    d.to_csv(ruta_csv, index=False, lineterminator="\n")
    meta = {
        "dataset": "IBM Telco Customer Churn",
        "version": VERSION_DATOS,
        "fuente": URL_FUENTE,
        "archivo_ini": f"datos/datos_ini/{ruta_ini.name}",
        "sha256_ini": sha256(ruta_ini),
        "archivo_limpio": f"datos/datos_limp/{NOMBRE_LIMP}",
        "sha256_limpio": sha256(ruta_csv),
        "filas": int(len(d)),
        "columnas": list(d.columns),
        "tasa_churn": round(float(d[OBJETIVO].mean()), 4),
        "perfil_calidad_ini": perfil,
        "bitacora_limpieza": bitacora,
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(dir_limp / (NOMBRE_LIMP.replace(".csv", ".meta.json")), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def cargar_limpio(ruta: str | Path) -> pd.DataFrame:
    """Lee el CSV limpio restaurando los tipos categóricos."""
    d = pd.read_csv(ruta)
    for c in CATEGORICAS:
        d[c] = d[c].astype(str).astype("category")
    return d


def preparar(raiz: str | Path = ".") -> dict:
    """Flujo completo: descarga (si hace falta) -> perfil -> limpieza -> validación -> versionado."""
    raiz = Path(raiz)
    ruta_ini = descargar_datos(raiz / "datos" / "datos_ini" / NOMBRE_INI)
    df = cargar_datos_ini(ruta_ini)
    perfil = perfil_calidad(df)
    limpio, bitacora = limpiar(df)
    validar_limpio(limpio)
    return guardar_limpio(limpio, bitacora, ruta_ini, raiz / "datos" / "datos_limp", perfil)


if __name__ == "__main__":
    m = preparar(Path(__file__).resolve().parents[1])
    print(f"Dataset limpio v{m['version']}: {m['filas']} filas · churn {m['tasa_churn']:.2%}")
    print(f"SHA-256 original: {m['sha256_ini']}")
    print(f"SHA-256 limpio:   {m['sha256_limpio']}")
    for paso in m["bitacora_limpieza"]:
        print(f"  {paso['regla']:>3}  {paso['descripcion']:<70} {paso['filas_afectadas']:>6}")
