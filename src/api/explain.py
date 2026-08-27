"""Explicabilidad SHAP punto a punto para el modelo multi-output de energia
especifica / throughput.

El mejor modelo se elige dinamicamente en `train_multioutput.run_training_pipeline`
(puede terminar siendo un `MultiOutputRegressor` envolviendo un estimador de
arbol por target, o un estimador nativamente multi-output como RandomForest o
regresion lineal), asi que este modulo construye el explainer apropiado segun
el tipo de modelo cargado en vez de asumir uno fijo: `shap.TreeExplainer`
cuando el estimador es un arbol (rapido, exacto), con fallback a un
explainer model-agnostico (`shap.Explainer` sobre `predict`) en cualquier
otro caso.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor


def _tree_or_generic_explainer(estimator, background: pd.DataFrame):
    try:
        return shap.TreeExplainer(estimator, background)
    except Exception:
        return shap.Explainer(estimator.predict, background)


def build_target_explainers(model, background: pd.DataFrame) -> list:
    """Devuelve un explainer por target (mismo orden que `TARGET_COLS`)."""
    if isinstance(model, MultiOutputRegressor):
        return [_tree_or_generic_explainer(est, background) for est in model.estimators_]
    shared = _tree_or_generic_explainer(model, background)
    return [shared, shared]


def explain_point(explainers: list, target_cols: list[str], x_row: pd.DataFrame) -> dict:
    """Calcula la contribucion SHAP de cada feature a cada prediccion
    (energia especifica, throughput) para una sola fila de entrada."""
    feature_cols = list(x_row.columns)
    result = {}

    for i, (target, explainer) in enumerate(zip(target_cols, explainers)):
        shap_values = explainer(x_row)
        values = np.asarray(shap_values.values)

        if values.ndim == 3:
            contribs = values[0, :, i]
        elif values.ndim == 2:
            contribs = values[0]
        else:
            contribs = values

        result[target] = {f: float(v) for f, v in zip(feature_cols, contribs)}

    return result
