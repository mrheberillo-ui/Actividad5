"""
train.py — Entrenamiento, ajuste de hiperparámetros (GridSearchCV + k-fold
estratificado), evaluación, comparación estadística y registro en MLflow.

Uso desde la terminal (raíz de Actividad5/), después de datos_prep.py:
    python fuentes/train.py
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Experimento MLflow: "Actividad5-churn-telco"
    · 1 run de referencia (DummyClassifier)
    · 1 run padre por algoritmo con sus candidatos del grid como runs hijos
    · 1 run de comparación estadística entre los dos mejores modelos
Los modelos ganadores se registran en el Model Registry.
"""
from __future__ import annotations

import inspect
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*ill-defined.*")
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")

# ---------------------------------------------------------------------------
# Configuración del experimento (todo lo que afecta el resultado vive aquí)
# ---------------------------------------------------------------------------
SEMILLA = 42
PROPORCION_PRUEBA = 0.20
K_PLIEGUES = 5
METRICA_SELECCION = "roc_auc"
NOMBRE_EXPERIMENTO = "Actividad5-churn-telco"
COLORES = {"regresion_logistica": "#2a78d6", "random_forest": "#eb6834", "referencia": "#8a8984"}
ETIQUETAS = {"regresion_logistica": "Regresión logística", "random_forest": "Random forest",
             "referencia": "Referencia (clase mayoritaria)"}

ESCORERS = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "average_precision": "average_precision",
}

_SK_NUEVO = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def espacio_busqueda() -> dict:
    """Modelos base y rejillas de hiperparámetros.

    Regresión logística: C (inverso de la regularización) en escala logarítmica,
    tipo de regularización (L1 hace selección de variables; L2 encoge) y
    ponderación de clases para el desbalance 74/26.  24 combinaciones.
    Random forest: profundidad máxima y mínimo de muestras por hoja controlan el
    sobreajuste; max_features la decorrelación entre árboles; class_weight el
    desbalance. 300 árboles fijos (más árboles no cambian el sesgo, sólo la
    varianza y el costo).  48 combinaciones.
    """
    rejilla_lr = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "clf__class_weight": [None, "balanced"]}
    if _SK_NUEVO:
        rejilla_lr["clf__l1_ratio"] = [0.0, 1.0]          # 0 = L2, 1 = L1 (scikit-learn ≥ 1.8)
    else:
        rejilla_lr["clf__penalty"] = ["l2", "l1"]
    return {
        "regresion_logistica": (
            LogisticRegression(solver="liblinear", max_iter=5000, random_state=SEMILLA),
            rejilla_lr,
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=300, random_state=SEMILLA, n_jobs=-1),
            {
                "clf__max_depth": [6, 10, None],
                "clf__min_samples_leaf": [1, 10, 30, 60],
                "clf__max_features": ["sqrt", 0.3],
                "clf__class_weight": [None, "balanced_subsample"],
            },
        ),
    }


def construir_pipeline(estimador) -> Pipeline:
    """Escalado de numéricas + one-hot de categóricas + clasificador.
    Se ajusta dentro de cada pliegue, así que no hay fuga de información."""
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), dp.NUMERICAS),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), dp.CATEGORICAS),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", estimador)])


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_particion(raiz: Path):
    """Lee el dataset limpio versionado y lo divide 80/20 estratificado."""
    ruta = raiz / "datos" / "datos_limp" / dp.NOMBRE_LIMP
    meta = json.loads((raiz / "datos" / "datos_limp" /
                       dp.NOMBRE_LIMP.replace(".csv", ".meta.json")).read_text(encoding="utf-8"))
    d = dp.cargar_limpio(ruta)
    X = d[dp.CATEGORICAS + dp.NUMERICAS]
    y = d[dp.OBJETIVO].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=PROPORCION_PRUEBA, stratify=y, random_state=SEMILLA)
    return d, X_tr, X_te, y_tr, y_te, meta, ruta


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metricas(y, proba, umbral: float = 0.5) -> dict:
    pred = (proba >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "average_precision": average_precision_score(y, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def umbral_optimo_f1(y, proba) -> float:
    """Umbral que maximiza F1 sobre probabilidades fuera de pliegue (sólo entrenamiento)."""
    p, r, u = precision_recall_curve(y, proba)
    f1 = np.where(p[:-1] + r[:-1] > 0, 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1]), 0)
    return float(u[int(np.argmax(f1))])


def nombres_hiperparametros(params: dict) -> dict:
    """Quita el prefijo clf__ y normaliza l1_ratio/penalty a 'regularizacion'."""
    limpio = {}
    for k, v in params.items():
        k = k.replace("clf__", "")
        if k == "l1_ratio":
            k, v = "regularizacion", "L1" if v == 1.0 else "L2"
        elif k == "penalty":
            k, v = "regularizacion", v.upper()
        limpio[k] = "None" if v is None else v
    return limpio


# ---------------------------------------------------------------------------
# Gráficas que se adjuntan como artefactos
# ---------------------------------------------------------------------------
def _guardar(fig, ruta: Path) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return ruta


def grafica_confusion(y, proba, umbral, titulo, ruta) -> Path:
    cm = confusion_matrix(y, (proba >= umbral).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1b1b1b", fontsize=12)
    ax.set_xticks([0, 1], ["Permanece", "Abandona"])
    ax.set_yticks([0, 1], ["Permanece", "Abandona"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)
    return _guardar(fig, ruta)


def grafica_roc(y, proba, titulo, ruta) -> Path:
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y, proba):.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="#999")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(titulo, fontsize=10)
    ax.legend(loc="lower right", frameon=False)
    return _guardar(fig, ruta)


# ---------------------------------------------------------------------------
# Entrenamiento con registro en MLflow
# ---------------------------------------------------------------------------
def configurar_mlflow(raiz: Path, tracking_uri: str | None = None) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{(raiz / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    if mlflow.get_experiment_by_name(NOMBRE_EXPERIMENTO) is None:
        mlflow.create_experiment(
            NOMBRE_EXPERIMENTO,
            artifact_location=(raiz / "mlartifacts").resolve().as_uri(),
            tags={"proyecto": "MSCA2102 Actividad 5", "problema": "clasificacion binaria",
                  "dataset": "IBM Telco Customer Churn",
                  # Hace que la interfaz de MLflow 3 abra el experimento en modo "Model training"
                  "mlflow.experimentKind": "custom_model_development"},
        )
    mlflow.set_experiment(NOMBRE_EXPERIMENTO)
    return uri


def _commit_git(raiz: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(raiz), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sin-git"


def etiquetas_comunes(raiz: Path, meta: dict) -> dict:
    return {
        "dataset_version": meta["version"],
        "dataset_sha256": meta["sha256_limpio"][:16],
        "dataset_ini_sha256": meta["sha256_ini"][:16],
        "git_commit": _commit_git(raiz),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "semilla": str(SEMILLA),
    }


def registrar_dataset(d: pd.DataFrame, ruta: Path, meta: dict) -> None:
    ds = mlflow.data.from_pandas(d, source=str(ruta), targets=dp.OBJETIVO,
                                 name=f"telco_churn_limpio_v{meta['version']}")
    mlflow.log_input(ds, context="entrenamiento")


def entrenar_referencia(X_tr, X_te, y_tr, y_te, tags, carpeta: Path) -> dict:
    """Línea base: predice siempre la clase mayoritaria (piso que todo modelo debe superar)."""
    with mlflow.start_run(run_name="referencia_clase_mayoritaria") as run:
        mlflow.set_tags({**tags, "tipo_run": "referencia", "modelo": "DummyClassifier"})
        modelo = construir_pipeline(DummyClassifier(strategy="most_frequent"))
        cv = StratifiedKFold(K_PLIEGUES, shuffle=True, random_state=SEMILLA)
        res = cross_validate(modelo, X_tr, y_tr, cv=cv, scoring=ESCORERS)
        modelo.fit(X_tr, y_tr)
        proba = modelo.predict_proba(X_te)[:, 1]
        m_te = metricas(y_te, proba)
        mlflow.log_params({"estrategia": "most_frequent", "k_pliegues": K_PLIEGUES})
        mlflow.log_metrics({f"cv_{k}_media": float(np.mean(res[f"test_{k}"])) for k in ESCORERS})
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m_te.items()})
        return {"run_id": run.info.run_id, "test": m_te, "proba": proba}


def entrenar_con_grid(nombre: str, estimador, rejilla: dict, X_tr, X_te, y_tr, y_te,
                      d: pd.DataFrame, ruta_datos: Path, meta: dict, tags: dict,
                      carpeta: Path, registrar_modelo: bool = True) -> dict:
    """GridSearchCV (k-fold estratificado) dentro de un run padre; cada candidato
    del grid se registra como run hijo con sus parámetros y métricas de CV."""
    cv = StratifiedKFold(K_PLIEGUES, shuffle=True, random_state=SEMILLA)
    grid = GridSearchCV(construir_pipeline(estimador), rejilla, scoring=ESCORERS,
                        refit=METRICA_SELECCION, cv=cv, n_jobs=-1, return_train_score=True)
    with mlflow.start_run(run_name=f"{nombre}_gridsearch") as padre:
        mlflow.set_tags({**tags, "tipo_run": "gridsearch", "modelo": type(estimador).__name__,
                         "metrica_seleccion": METRICA_SELECCION})
        registrar_dataset(d, ruta_datos, meta)
        grid.fit(X_tr, y_tr)
        cvr = pd.DataFrame(grid.cv_results_)

        # --- runs hijos: un candidato del grid por run ---
        for i, fila in cvr.iterrows():
            with mlflow.start_run(run_name=f"{nombre}_cand_{i:02d}", nested=True):
                mlflow.set_tags({"tipo_run": "candidato", "modelo": type(estimador).__name__,
                                 "rank_roc_auc": int(fila["rank_test_roc_auc"])})
                mlflow.log_params(nombres_hiperparametros(fila["params"]))
                mlflow.log_metrics({f"cv_{k}_media": float(fila[f"mean_test_{k}"]) for k in ESCORERS})
                mlflow.log_metrics({f"cv_{k}_desv": float(fila[f"std_test_{k}"]) for k in ESCORERS})
                mlflow.log_metric("cv_roc_auc_train_media", float(fila["mean_train_roc_auc"]))
                mlflow.log_metric("tiempo_ajuste_s", float(fila["mean_fit_time"]))

        mejor = grid.best_estimator_
        i_mejor = grid.best_index_
        # Umbral que maximiza F1 con probabilidades fuera de pliegue (sin tocar prueba)
        proba_oof = cross_val_predict(clone(mejor), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        umbral = umbral_optimo_f1(y_tr, proba_oof)
        proba_te = mejor.predict_proba(X_te)[:, 1]
        m_05 = metricas(y_te, proba_te, 0.5)
        m_u = metricas(y_te, proba_te, umbral)

        mlflow.log_params({
            **{f"mejor_{k}": v for k, v in nombres_hiperparametros(grid.best_params_).items()},
            "k_pliegues": K_PLIEGUES, "n_candidatos": len(cvr), "proporcion_prueba": PROPORCION_PRUEBA,
            "metrica_seleccion": METRICA_SELECCION, "umbral_f1_oof": round(umbral, 4),
            "escalado": "StandardScaler", "codificacion": "OneHotEncoder(drop=if_binary)",
            "n_entrenamiento": len(X_tr), "n_prueba": len(X_te),
            **{f"base_{k}": v for k, v in estimador.get_params().items()
               if k in ("solver", "max_iter", "n_estimators", "random_state")},
        })
        mlflow.log_metrics({f"cv_{k}_media": float(cvr.loc[i_mejor, f"mean_test_{k}"]) for k in ESCORERS})
        mlflow.log_metrics({f"cv_{k}_desv": float(cvr.loc[i_mejor, f"std_test_{k}"]) for k in ESCORERS})
        mlflow.log_metric("cv_roc_auc_train_media", float(cvr.loc[i_mejor, "mean_train_roc_auc"]))
        mlflow.log_metrics({f"test_{k}": float(v) for k, v in m_05.items()})
        mlflow.log_metrics({f"test_umbral_{k}": float(v) for k, v in m_u.items()
                            if k in ("accuracy", "precision", "recall", "f1")})

        # --- artefactos ---
        sub = carpeta / nombre
        sub.mkdir(parents=True, exist_ok=True)
        cvr_out = cvr.drop(columns=[c for c in cvr.columns if c.startswith("split")])
        cvr_out.to_csv(sub / "cv_results.csv", index=False)
        grafica_confusion(y_te, proba_te, 0.5, f"{ETIQUETAS[nombre]} · umbral 0.50", sub / "confusion_050.png")
        grafica_confusion(y_te, proba_te, umbral, f"{ETIQUETAS[nombre]} · umbral {umbral:.2f}",
                          sub / "confusion_umbral.png")
        grafica_roc(y_te, proba_te, f"ROC en prueba · {ETIQUETAS[nombre]}", sub / "roc_prueba.png")
        (sub / "metricas_prueba.json").write_text(json.dumps(
            {"umbral_050": m_05, "umbral_f1_oof": {"umbral": umbral, **m_u}}, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(sub), artifact_path="evaluacion")

        firma = infer_signature(X_tr.astype({c: str for c in dp.CATEGORICAS}), proba_te[:5])
        ejemplo = X_tr.head(3).astype({c: str for c in dp.CATEGORICAS})
        extra = {}
        if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
            # MLflow ≥ 3.x serializa con skops: se declaran los tipos propios del modelo
            # (creado por nosotros mismos, por eso es seguro confiar en ellos).
            extra["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
        info = mlflow.sklearn.log_model(
            mejor, name="modelo", signature=firma, input_example=ejemplo,
            registered_model_name=f"churn_{nombre}" if registrar_modelo else None, **extra,
        )
        return {"run_id": padre.info.run_id, "grid": grid, "cv_results": cvr, "mejor": mejor,
                "umbral": umbral, "proba_te": proba_te, "proba_oof": proba_oof,
                "test_050": m_05, "test_umbral": m_u, "model_uri": info.model_uri}


# ---------------------------------------------------------------------------
# Comparación estadística
# ---------------------------------------------------------------------------
def ttest_corregido(diferencias: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """t de remuestreo corregida (Nadeau y Bengio, 2003): corrige la varianza por el
    traslape entre conjuntos de entrenamiento de distintos pliegues."""
    k = len(diferencias)
    media, var = diferencias.mean(), diferencias.var(ddof=1)
    t = media / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def comparar_modelos(res_a: dict, res_b: dict, X_tr, y_tr, y_te, tags: dict, carpeta: Path,
                     n_bootstrap: int = 2000) -> dict:
    """Compara los dos mejores modelos con: (1) CV repetida 10×3 sobre los mismos
    pliegues + t corregida y Wilcoxon; (2) bootstrap pareado del AUC en prueba;
    (3) McNemar sobre las predicciones en prueba con el umbral de cada modelo."""
    rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEMILLA)
    pliegues = {}
    for nombre, res in (("regresion_logistica", res_a), ("random_forest", res_b)):
        r = cross_validate(clone(res["mejor"]), X_tr, y_tr, cv=rcv,
                           scoring={"roc_auc": "roc_auc", "average_precision": "average_precision", "f1": "f1"},
                           n_jobs=-1)
        pliegues[nombre] = r
    n_test_f = len(y_tr) // 10
    n_train_f = len(y_tr) - n_test_f
    filas = []
    for met in ("roc_auc", "average_precision", "f1"):
        a = pliegues["regresion_logistica"][f"test_{met}"]
        b = pliegues["random_forest"][f"test_{met}"]
        dif = a - b
        t, p_t = ttest_corregido(dif, n_train_f, n_test_f)
        w = stats.wilcoxon(a, b)
        filas.append({"metrica": met, "media_lr": a.mean(), "desv_lr": a.std(ddof=1),
                      "media_rf": b.mean(), "desv_rf": b.std(ddof=1), "dif_media": dif.mean(),
                      "lr_gana_pliegues": int((dif > 0).sum()), "n_pliegues": len(dif),
                      "t_corregida": t, "p_t_corregida": p_t,
                      "wilcoxon_W": float(w.statistic), "p_wilcoxon": float(w.pvalue)})
    tabla_cv = pd.DataFrame(filas)

    # Bootstrap pareado del AUC en prueba
    rng = np.random.default_rng(SEMILLA)
    ya = np.asarray(y_te)
    pa, pb = res_a["proba_te"], res_b["proba_te"]
    difs, aucs_a, aucs_b = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(ya), len(ya))
        if ya[idx].min() == ya[idx].max():
            continue
        aa, bb = roc_auc_score(ya[idx], pa[idx]), roc_auc_score(ya[idx], pb[idx])
        aucs_a.append(aa); aucs_b.append(bb); difs.append(aa - bb)
    difs = np.array(difs)
    boot = {
        "auc_lr": roc_auc_score(ya, pa), "auc_lr_ic95": np.percentile(aucs_a, [2.5, 97.5]).tolist(),
        "auc_rf": roc_auc_score(ya, pb), "auc_rf_ic95": np.percentile(aucs_b, [2.5, 97.5]).tolist(),
        "dif_auc": float(roc_auc_score(ya, pa) - roc_auc_score(ya, pb)),
        "dif_auc_ic95": np.percentile(difs, [2.5, 97.5]).tolist(),
        "p_bootstrap_bilateral": float(min(1.0, 2 * min((difs <= 0).mean(), (difs >= 0).mean()))),
        "n_bootstrap": int(len(difs)),
    }

    # McNemar (con corrección de continuidad) sobre aciertos en prueba
    pred_a = (pa >= res_a["umbral"]).astype(int)
    pred_b = (pb >= res_b["umbral"]).astype(int)
    ok_a, ok_b = pred_a == ya, pred_b == ya
    n01, n10 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10) if (n01 + n10) else 0.0
    mcnemar = {"lr_acierta_rf_falla": n01, "rf_acierta_lr_falla": n10,
               "chi2": float(chi2), "p_mcnemar": float(stats.chi2.sf(chi2, 1))}

    # Gráfica de distribución por pliegue
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, met, titulo in zip(axes, ("roc_auc", "average_precision", "f1"),
                               ("ROC-AUC", "Precisión promedio (PR-AUC)", "F1 (umbral 0.50)")):
        datos = [pliegues["regresion_logistica"][f"test_{met}"], pliegues["random_forest"][f"test_{met}"]]
        bp = ax.boxplot(datos, widths=0.5, patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORES["regresion_logistica"] + "55")
        bp["boxes"][1].set_facecolor(COLORES["random_forest"] + "55")
        for k, serie in enumerate(datos, start=1):
            ax.scatter(np.full(len(serie), k) + np.random.default_rng(k).uniform(-.12, .12, len(serie)),
                       serie, s=9, c=list(COLORES.values())[k - 1], alpha=.7, zorder=3)
        ax.set_xticks([1, 2], ["Reg. logística", "Random forest"])
        ax.set_title(f"{titulo} · 30 pliegues", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    sub = carpeta / "comparacion"
    sub.mkdir(parents=True, exist_ok=True)
    _guardar(fig, sub / "pliegues_cv_repetida.png")
    tabla_cv.to_csv(sub / "pruebas_cv_repetida.csv", index=False)
    pd.DataFrame({"lr_roc_auc": pliegues["regresion_logistica"]["test_roc_auc"],
                  "rf_roc_auc": pliegues["random_forest"]["test_roc_auc"],
                  "lr_average_precision": pliegues["regresion_logistica"]["test_average_precision"],
                  "rf_average_precision": pliegues["random_forest"]["test_average_precision"],
                  "lr_f1": pliegues["regresion_logistica"]["test_f1"],
                  "rf_f1": pliegues["random_forest"]["test_f1"]}).to_csv(sub / "pliegues.csv", index=False)
    (sub / "bootstrap_mcnemar.json").write_text(json.dumps({"bootstrap_auc": boot, "mcnemar": mcnemar},
                                                           indent=2), encoding="utf-8")

    with mlflow.start_run(run_name="comparacion_estadistica"):
        mlflow.set_tags({**tags, "tipo_run": "comparacion",
                         "run_lr": res_a["run_id"], "run_rf": res_b["run_id"]})
        mlflow.log_params({"cv_repetida": "RepeatedStratifiedKFold(10x3)", "n_bootstrap": n_bootstrap,
                           "prueba_pareada": "t corregida Nadeau-Bengio + Wilcoxon"})
        for _, f in tabla_cv.iterrows():
            m = f["metrica"]
            mlflow.log_metrics({f"{m}_dif_media": f["dif_media"], f"{m}_p_t_corregida": f["p_t_corregida"],
                                f"{m}_p_wilcoxon": f["p_wilcoxon"]})
        mlflow.log_metrics({"auc_test_dif": boot["dif_auc"], "auc_test_dif_ic_inf": boot["dif_auc_ic95"][0],
                            "auc_test_dif_ic_sup": boot["dif_auc_ic95"][1],
                            "p_bootstrap": boot["p_bootstrap_bilateral"], "p_mcnemar": mcnemar["p_mcnemar"]})
        mlflow.log_artifacts(str(sub), artifact_path="comparacion")
    return {"tabla_cv": tabla_cv, "pliegues": pliegues, "bootstrap": boot, "mcnemar": mcnemar}


# ---------------------------------------------------------------------------
# Flujo completo
# ---------------------------------------------------------------------------
def ejecutar(raiz: str | Path = ".", tracking_uri: str | None = None) -> dict:
    raiz = Path(raiz).resolve()
    np.random.seed(SEMILLA)
    uri = configurar_mlflow(raiz, tracking_uri)
    d, X_tr, X_te, y_tr, y_te, meta, ruta = cargar_particion(raiz)
    tags = etiquetas_comunes(raiz, meta)
    carpeta = raiz / "resultados"
    salida = {"tracking_uri": uri, "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te, "datos": d}
    salida["referencia"] = entrenar_referencia(X_tr, X_te, y_tr, y_te, tags, carpeta)
    for nombre, (est, rejilla) in espacio_busqueda().items():
        print(f"GridSearchCV · {nombre} · {np.prod([len(v) for v in rejilla.values()])} candidatos × {K_PLIEGUES} pliegues")
        salida[nombre] = entrenar_con_grid(nombre, est, rejilla, X_tr, X_te, y_tr, y_te,
                                           d, ruta, meta, tags, carpeta)
    salida["comparacion"] = comparar_modelos(salida["regresion_logistica"], salida["random_forest"],
                                             X_tr, y_tr, y_te, tags, carpeta)
    # Exporta la tabla de runs (respaldo legible de lo registrado en MLflow)
    exp = mlflow.get_experiment_by_name(NOMBRE_EXPERIMENTO)
    runs = mlflow.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    runs.to_csv(carpeta / "mlflow_runs.csv", index=False)
    salida["runs"] = runs
    return salida


if __name__ == "__main__":
    s = ejecutar(Path(__file__).resolve().parents[1])
    for n in ("regresion_logistica", "random_forest"):
        r = s[n]
        print(f"{n:<22} mejores={nombres_hiperparametros(r['grid'].best_params_)}  "
              f"AUC cv={r['grid'].best_score_:.4f}  AUC prueba={r['test_050']['roc_auc']:.4f}  "
              f"F1 prueba={r['test_050']['f1']:.4f}")
    print(s["comparacion"]["tabla_cv"].round(4).to_string(index=False))
    print(json.dumps(s["comparacion"]["bootstrap"], indent=1))
    print(s["comparacion"]["mcnemar"])
    print(f"Runs registrados en {s['tracking_uri']}: {len(s['runs'])}")
