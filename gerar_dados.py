"""Gera data/credito.csv: 6.000 contratos sintéticos, ~21% inadimplentes.

Uso: python gerar_dados.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

N = 6000
TAXA_ALVO = 0.21
SEED = 42
TAXA_JUROS_MES = 0.025  # usada só para simular a parcela "real" do contrato

rng = np.random.default_rng(SEED)

idade = np.clip(rng.normal(40, 12, N), 18, 80).round().astype(int)

# renda cresce com a idade até ~50 anos
fator_idade = 1 + 0.015 * np.clip(idade - 18, 0, 32)
renda_mensal = np.round(rng.lognormal(np.log(2800), 0.55, N) * fator_idade, 2)
renda_mensal = np.clip(renda_mensal, 1412, 60000)

tempo_emprego_anos = np.minimum(rng.exponential(6, N), idade - 16).clip(0).round(1)

score_credito = np.clip(rng.normal(640, 130, N), 300, 1000).round().astype(int)

dividas_ativas = np.minimum(rng.poisson(1.1, N), 8)

p_imovel = 1 / (1 + np.exp(-(idade - 42) / 8))
possui_imovel = np.where(rng.random(N) < p_imovel, "sim", "nao")

finalidades = np.array(["pessoal", "veiculo", "reforma", "educacao", "negocio"])
finalidade = rng.choice(finalidades, N, p=[0.35, 0.25, 0.15, 0.10, 0.15])

prazo_meses = rng.choice([12, 24, 36, 48, 60], N, p=[0.15, 0.25, 0.25, 0.15, 0.20])
valor_emprestimo = rng.lognormal(np.log(renda_mensal * 3.5), 0.6)
valor_emprestimo = np.clip(np.round(valor_emprestimo, -2), 1000, 150000)

# parcela real (Price) e comprometimento de renda: principal driver "escondido"
i = TAXA_JUROS_MES
parcela = valor_emprestimo * i / (1 - (1 + i) ** -prazo_meses)
comprometimento = parcela / renda_mensal

efeito_finalidade = pd.Series(
    {"pessoal": 0.30, "veiculo": -0.25, "reforma": -0.10, "educacao": 0.05, "negocio": 0.55}
)

logit_sem_intercepto = (
    -0.0095 * (score_credito - 640)
    + 0.38 * dividas_ativas
    - 0.55 * (possui_imovel == "sim")
    + efeito_finalidade[finalidade].to_numpy()
    + 3.0 * np.minimum(comprometimento, 1.2)
    - 0.045 * tempo_emprego_anos
    - 0.012 * (idade - 40)
    + 0.004 * (prazo_meses - 36)
    + rng.normal(0, 0.9, N)  # heterogeneidade não observada
)

# calibra o intercepto por bisseção para obter ~21% de inadimplência esperada
lo, hi = -10.0, 10.0
for _ in range(60):
    b0 = (lo + hi) / 2
    if (1 / (1 + np.exp(-(b0 + logit_sem_intercepto)))).mean() > TAXA_ALVO:
        hi = b0
    else:
        lo = b0
prob = 1 / (1 + np.exp(-(b0 + logit_sem_intercepto)))
inadimplente = (rng.random(N) < prob).astype(int)

df = pd.DataFrame(
    {
        "id_contrato": np.arange(100001, 100001 + N),
        "idade": idade,
        "renda_mensal": renda_mensal,
        "tempo_emprego_anos": tempo_emprego_anos,
        "score_credito": score_credito,
        "dividas_ativas": dividas_ativas,
        "possui_imovel": possui_imovel,
        "finalidade": finalidade,
        "valor_emprestimo": valor_emprestimo,
        "prazo_meses": prazo_meses,
        "inadimplente": inadimplente,
    }
)

# ausentes: renda (não informada) e tempo de emprego (autônomos / não informado)
df.loc[rng.random(N) < 0.06, "renda_mensal"] = np.nan
df.loc[rng.random(N) < 0.08, "tempo_emprego_anos"] = np.nan

saida = Path(__file__).parent / "data" / "credito.csv"
saida.parent.mkdir(exist_ok=True)
df.to_csv(saida, index=False)
print(f"{len(df)} contratos salvos em {saida} | inadimplência: {df.inadimplente.mean():.1%}")
