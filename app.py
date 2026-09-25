"""Sistema web: uma aba por problema de config.PROBLEMAS.

Uso: python treinar.py  (uma vez)  ->  python app.py  ->  http://localhost:5000
"""
import math
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request

from config import PROBLEMAS

RAIZ = Path(__file__).parent
app = Flask(__name__)
_modelos = {}


def carregar(nome):
    if nome not in _modelos:
        caminho = RAIZ / PROBLEMAS[nome]["modelo"]
        if not caminho.exists():
            abort(503, f"Modelo de '{nome}' não encontrado. Rode: python treinar.py {nome}")
        _modelos[nome] = joblib.load(caminho)
    return _modelos[nome]


def validar(cfg, dados):
    """Converte e valida a entrada conforme limites e opções do config. Devolve (linha, erros)."""
    linha, erros = {}, []
    for f in cfg["features"]:
        bruto = dados.get(f["nome"])
        vazio = bruto is None or str(bruto).strip() == ""
        if f["tipo"] == "numero":
            if vazio:
                if f.get("opcional"):
                    linha[f["nome"]] = math.nan  # o pipeline imputa
                    continue
                erros.append(f"{f['rotulo']} é obrigatório.")
                continue
            try:
                valor = float(str(bruto).replace(",", "."))
            except ValueError:
                erros.append(f"{f['rotulo']} deve ser numérico.")
                continue
            if not f["min"] <= valor <= f["max"]:
                erros.append(f"{f['rotulo']} deve estar entre {f['min']} e {f['max']}.")
            linha[f["nome"]] = valor
        else:
            if bruto not in f["opcoes"]:
                erros.append(f"{f['rotulo']}: escolha entre {', '.join(f['opcoes'])}.")
            linha[f["nome"]] = bruto
    return linha, erros


@app.get("/")
def index():
    return render_template("index.html", problemas=PROBLEMAS)


@app.get("/api/<nome>/info")
def info(nome):
    if nome not in PROBLEMAS:
        abort(404)
    return jsonify(carregar(nome)["resultado"])


@app.post("/api/<nome>/prever")
def prever(nome):
    if nome not in PROBLEMAS:
        abort(404)
    cfg = PROBLEMAS[nome]
    linha, erros = validar(cfg, request.get_json(silent=True) or {})
    if erros:
        return jsonify({"erros": erros}), 400

    art = carregar(nome)
    X = pd.DataFrame([linha])[art["colunas"]]
    prob = float(art["pipeline"].predict_proba(X)[0, 1])
    classe = int(prob >= art["limiar"])
    custos = cfg.get("custos", {})
    return jsonify({
        "probabilidade": prob,
        "limiar": art["limiar"],
        "classe": classe,
        **cfg["classes"][classe],
        # perda esperada de cada decisão para este pedido (ajuda a equipe a revisar)
        "perda_esperada_aprovar": prob * custos.get("falso_negativo", 0),
        "perda_esperada_recusar": (1 - prob) * custos.get("falso_positivo", 0),
    })


@app.errorhandler(503)
def sem_modelo(e):
    return jsonify({"erros": [e.description]}), 503


if __name__ == "__main__":
    app.run(debug=False, port=5000)
