"""Blocos de modelagem compartilhados entre o notebook e treinar.py."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TAXA_JUROS_MES = 0.02  # taxa de referência para estimar a parcela (Tabela Price)


def adicionar_comprometimento(X: pd.DataFrame) -> pd.DataFrame:
    """Parcela estimada / renda. Operação linha a linha: não aprende nada, não vaza."""
    X = X.copy()
    i = TAXA_JUROS_MES
    parcela = X["valor_emprestimo"] * i / (1 - (1 + i) ** -X["prazo_meses"])
    X["comprometimento_renda"] = parcela / X["renda_mensal"].replace(0, np.nan)
    return X


def criar_preprocessador(numericas, categoricas, escalar=True):
    passos_num = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        passos_num.append(("escalar", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(passos_num), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ]
    )


def algoritmos(class_weight=None):
    return {
        "regressao_logistica": LogisticRegression(max_iter=2000, class_weight=class_weight),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=10, n_jobs=-1,
            class_weight=class_weight, random_state=RANDOM_STATE,
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
            class_weight=class_weight, random_state=RANDOM_STATE,
        ),
    }


def criar_pipeline(modelo, numericas, categoricas, comprometimento=False):
    numericas = list(numericas)
    passos = []
    if comprometimento:
        passos.append(("comprometimento", FunctionTransformer(adicionar_comprometimento)))
        numericas = numericas + ["comprometimento_renda"]
    passos += [("preproc", criar_preprocessador(numericas, categoricas)), ("modelo", modelo)]
    return Pipeline(passos)


def custo_total(y_true, y_prob, limiar, custo_fn=8000, custo_fp=1500):
    """FN = inadimplente aprovado; FP = bom cliente recusado."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_prob) >= limiar).astype(int)
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return fn * custo_fn + fp * custo_fp, fn, fp


def melhor_limiar(y_true, y_prob, custo_fn=8000, custo_fp=1500, grade=None):
    grade = np.round(np.arange(0.05, 0.96, 0.01), 2) if grade is None else grade
    custos = [custo_total(y_true, y_prob, t, custo_fn, custo_fp)[0] for t in grade]
    return float(grade[int(np.argmin(custos))])
