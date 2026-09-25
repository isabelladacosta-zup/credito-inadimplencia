"""Configuração dos problemas servidos pelo sistema.

Cada entrada de PROBLEMAS vira uma aba na interface (index.html) e um modelo
treinado por treinar.py. As features descrevem o formulário: tipo, limites e opções.
"""

PROBLEMAS = {
    "credito": {
        "titulo": "Risco de crédito",
        "descricao": "Probabilidade de inadimplência de um pedido de empréstimo.",
        "dados": "data/credito.csv",
        "alvo": "inadimplente",
        "descartar": ["id_contrato"],  # identificador, sem valor preditivo
        "modelo": "modelos/credito.joblib",
        "algoritmo": "regressao_logistica",  # escolhido por validação cruzada (notebook, parte 2)
        "hiperparametros": {"C": 0.03},  # GridSearchCV (notebook, desafio extra 2)
        "comprometimento_renda": True,  # feature derivada (desafio extra 1)
        "custos": {"falso_negativo": 8000, "falso_positivo": 1500},  # R$
        "classes": {
            0: {"rotulo": "Risco baixo", "acao": "Aprovar"},
            1: {"rotulo": "Risco alto", "acao": "Recusar ou enviar para análise manual"},
        },
        "features": [
            {"nome": "idade", "rotulo": "Idade", "tipo": "numero",
             "min": 18, "max": 80, "passo": 1, "padrao": 35, "unidade": "anos"},
            {"nome": "renda_mensal", "rotulo": "Renda mensal", "tipo": "numero",
             "min": 0, "max": 100000, "passo": 100, "padrao": 4000, "unidade": "R$",
             "opcional": True},
            {"nome": "tempo_emprego_anos", "rotulo": "Tempo de emprego", "tipo": "numero",
             "min": 0, "max": 60, "passo": 0.5, "padrao": 5, "unidade": "anos",
             "opcional": True},
            {"nome": "score_credito", "rotulo": "Score de crédito", "tipo": "numero",
             "min": 300, "max": 1000, "passo": 1, "padrao": 650},
            {"nome": "dividas_ativas", "rotulo": "Dívidas ativas", "tipo": "numero",
             "min": 0, "max": 20, "passo": 1, "padrao": 1},
            {"nome": "possui_imovel", "rotulo": "Possui imóvel", "tipo": "categoria",
             "opcoes": ["sim", "nao"], "padrao": "nao"},
            {"nome": "finalidade", "rotulo": "Finalidade", "tipo": "categoria",
             "opcoes": ["pessoal", "veiculo", "reforma", "educacao", "negocio"],
             "padrao": "pessoal"},
            {"nome": "valor_emprestimo", "rotulo": "Valor do empréstimo", "tipo": "numero",
             "min": 1000, "max": 150000, "passo": 500, "padrao": 15000, "unidade": "R$"},
            {"nome": "prazo_meses", "rotulo": "Prazo", "tipo": "numero",
             "min": 12, "max": 60, "passo": 12, "padrao": 36, "unidade": "meses"},
        ],
    },
}
