"""Treina e salva um modelo para cada problema de config.PROBLEMAS.

Uso: python treinar.py            (todos os problemas)
     python treinar.py credito    (só um)

Para cada problema: separa treino/teste (80/20 estratificado, random_state=42),
escolhe o limiar de menor custo com cross_val_predict NO TREINO, treina o pipeline
no treino, mede no teste e salva modelo + limiar + métricas em modelos/.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict, train_test_split

from config import PROBLEMAS
from modelagem import algoritmos, criar_pipeline, custo_total, melhor_limiar

RAIZ = Path(__file__).parent


def metricas(y, prob, limiar, custos):
    pred = (prob >= limiar).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    custo, _, _ = custo_total(y, prob, limiar, custos["falso_negativo"], custos["falso_positivo"])
    return {
        "acuracia": accuracy_score(y, pred), "precisao": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob),
        "matriz_confusao": {"vn": int(tn), "fp": int(fp), "fn": int(fn), "vp": int(tp)},
        "custo_total": custo,
    }


def treinar(nome, cfg):
    print(f"\n=== {nome}: {cfg['titulo']} ===")
    df = pd.read_csv(RAIZ / cfg["dados"])
    numericas = [f["nome"] for f in cfg["features"] if f["tipo"] == "numero"]
    categoricas = [f["nome"] for f in cfg["features"] if f["tipo"] == "categoria"]
    X = df[numericas + categoricas]  # só o que o formulário fornece; 'descartar' fica de fora
    y = df[cfg["alvo"]]

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    modelo = algoritmos()[cfg["algoritmo"]].set_params(**cfg.get("hiperparametros", {}))
    pipeline = criar_pipeline(modelo, numericas, categoricas, cfg.get("comprometimento_renda", False))

    custos = cfg["custos"]
    prob_oof = cross_val_predict(pipeline, X_treino, y_treino, cv=5, method="predict_proba")[:, 1]
    limiar = melhor_limiar(y_treino, prob_oof, custos["falso_negativo"], custos["falso_positivo"])
    print(f"limiar de menor custo (validação cruzada no treino): {limiar:.2f}")

    pipeline.fit(X_treino, y_treino)
    prob_teste = pipeline.predict_proba(X_teste)[:, 1]
    resultado = {
        "problema": nome, "algoritmo": cfg["algoritmo"], "limiar": limiar,
        "treinado_em": datetime.now().isoformat(timespec="seconds"),
        "n_treino": len(X_treino), "n_teste": len(X_teste),
        "cv_treino": metricas(y_treino, prob_oof, limiar, custos),
        "teste": metricas(y_teste, prob_teste, limiar, custos),
        "teste_limiar_05": metricas(y_teste, prob_teste, 0.5, custos),
    }

    destino = RAIZ / cfg["modelo"]
    destino.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipeline, "limiar": limiar, "colunas": list(X.columns),
                 "resultado": resultado}, destino)
    destino.with_suffix(".json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False),
                                            encoding="utf-8")

    t = resultado["teste"]
    print(f"teste | ROC AUC {t['roc_auc']:.4f} | precisão {t['precisao']:.3f} | recall {t['recall']:.3f} "
          f"| F1 {t['f1']:.3f} | acurácia {t['acuracia']:.3f}")
    print(f"teste | custo com limiar {limiar:.2f}: R$ {t['custo_total']:,.0f} "
          f"(limiar 0,5: R$ {resultado['teste_limiar_05']['custo_total']:,.0f})")
    print(f"modelo salvo em {destino}")


if __name__ == "__main__":
    escolhidos = sys.argv[1:] or list(PROBLEMAS)
    for nome in escolhidos:
        treinar(nome, PROBLEMAS[nome])
