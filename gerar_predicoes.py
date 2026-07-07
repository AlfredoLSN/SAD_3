import pandas as pd
import joblib


# ============================================================
# 1. Carregar base e modelo
# ============================================================

base = pd.read_csv("base_classificacao_beat_mes.csv")
base["MES_REF"] = pd.to_datetime(base["MES_REF"], errors="coerce")

artefatos = joblib.load("modelo_prioridade_beat.pkl")

modelo = artefatos["modelo"]
features = artefatos["features"]

# ============================================================
# 2. Selecionar o mês mais recente disponível
# ============================================================

ultimo_mes = base["MES_REF"].max()

dados_dashboard = base[base["MES_REF"] == ultimo_mes].copy()

# Garantir features sem nulos
dados_dashboard[features] = dados_dashboard[features].fillna(0)

# ============================================================
# 3. Gerar previsões
# ============================================================

X_dash = dados_dashboard[features]

pred = modelo.predict(X_dash)
probas = modelo.predict_proba(X_dash)
classes = modelo.classes_

dados_dashboard["prioridade_prevista"] = pred

for i, classe in enumerate(classes):
    dados_dashboard[f"prob_{classe.lower()}"] = probas[:, i]

# ============================================================
# 4. Criar coluna de probabilidade da classe prevista
# ============================================================

def obter_prob_prevista(row):
    classe = row["prioridade_prevista"].lower()
    return row[f"prob_{classe}"]

dados_dashboard["prob_prioridade_prevista"] = dados_dashboard.apply(
    obter_prob_prevista,
    axis=1
)

# ============================================================
# 5. Ordenar para ranking
# ============================================================

if "prob_alta" in dados_dashboard.columns:
    dados_dashboard = dados_dashboard.sort_values(
        ["prioridade_prevista", "prob_alta"],
        ascending=[True, False]
    )

# ============================================================
# 6. Salvar arquivo para o dashboard
# ============================================================

dados_dashboard.to_csv("predicoes_dashboard_ultimo_mes.csv", index=False)

print(f"Predições geradas para o mês: {ultimo_mes}")
print("Arquivo salvo em: predicoes_dashboard_ultimo_mes.csv")
print(dados_dashboard[[
    "BEAT_OF_OCCURRENCE",
    "MES_REF",
    "prioridade_prevista",
    "prob_prioridade_prevista"
]].head())