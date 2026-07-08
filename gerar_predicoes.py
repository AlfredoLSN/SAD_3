import pandas as pd
import numpy as np
import joblib


BASE_MENSAL = "base_beat_mes.csv"
ARTEFATO_MODELO = "modelo_prioridade_beat.pkl"
SAIDA_DASHBOARD = "predicoes_dashboard_ultimo_mes.csv"
SAIDA_OPERACIONAL = "predicoes_dashboard_operacional.csv"


def safe_div(num, den):
    return np.where(den > 0, num / den, 0)


def preparar_features_operacionais(base_mensal, features):
    base = base_mensal.copy()
    base["MES_REF"] = pd.to_datetime(base["MES_REF"], errors="coerce")
    base["BEAT_OF_OCCURRENCE"] = base["BEAT_OF_OCCURRENCE"].astype(str)
    base = base.dropna(subset=["MES_REF"])
    base = base.sort_values(["BEAT_OF_OCCURRENCE", "MES_REF"])

    ultimo_mes_historico = base["MES_REF"].max()
    mes_previsao_data = ultimo_mes_historico + pd.DateOffset(months=1)

    ultimos_6m = base[
        (base["MES_REF"] > ultimo_mes_historico - pd.DateOffset(months=6))
        & (base["MES_REF"] <= ultimo_mes_historico)
    ].copy()

    agregada = (
        ultimos_6m.groupby("BEAT_OF_OCCURRENCE")
        .agg(
            total_acidentes_ult_6m=("total_acidentes_mes", "sum"),
            acidentes_noite_ult_6m=("acidentes_noite_mes", "sum"),
            acidentes_fim_semana_ult_6m=("acidentes_fim_semana_mes", "sum"),
            iluminacao_risco_ult_6m=("iluminacao_risco_mes", "sum"),
            pista_risco_ult_6m=("pista_risco_mes", "sum"),
            defeito_via_ult_6m=("defeito_via_mes", "sum"),
            dispositivo_problema_ult_6m=("dispositivo_problema_mes", "sum"),
            clima_risco_ult_6m=("clima_risco_mes", "sum"),
            velocidade_soma_ult_6m=("velocidade_soma_mes", "sum"),
            num_units_soma_ult_6m=("num_units_soma_mes", "sum"),
            lat_media_beat=("lat_media_beat", "first"),
            lon_media_beat=("lon_media_beat", "first"),
        )
        .reset_index()
    )

    total = agregada["total_acidentes_ult_6m"]
    agregada["media_acidentes_mensal_ult_6m"] = total / 6
    agregada["perc_acidentes_noite_ult_6m"] = safe_div(
        agregada["acidentes_noite_ult_6m"], total
    )
    agregada["perc_acidentes_fim_semana_ult_6m"] = safe_div(
        agregada["acidentes_fim_semana_ult_6m"], total
    )
    agregada["perc_iluminacao_risco_ult_6m"] = safe_div(
        agregada["iluminacao_risco_ult_6m"], total
    )
    agregada["perc_pista_risco_ult_6m"] = safe_div(
        agregada["pista_risco_ult_6m"], total
    )
    agregada["perc_defeito_via_ult_6m"] = safe_div(
        agregada["defeito_via_ult_6m"], total
    )
    agregada["perc_dispositivo_problema_ult_6m"] = safe_div(
        agregada["dispositivo_problema_ult_6m"], total
    )
    agregada["perc_clima_risco_ult_6m"] = safe_div(
        agregada["clima_risco_ult_6m"], total
    )
    agregada["velocidade_media_ult_6m"] = safe_div(
        agregada["velocidade_soma_ult_6m"], total
    )
    agregada["num_units_medio_ult_6m"] = safe_div(
        agregada["num_units_soma_ult_6m"], total
    )

    agregada["MES_REF"] = mes_previsao_data
    agregada["ultimo_mes_historico"] = ultimo_mes_historico
    agregada["mes_previsao"] = mes_previsao_data.month

    agregada[features] = agregada[features].fillna(0)
    return agregada, ultimo_mes_historico, mes_previsao_data


def main():
    base_mensal = pd.read_csv(BASE_MENSAL)
    artefatos = joblib.load(ARTEFATO_MODELO)

    modelo = artefatos["modelo"]
    features = artefatos["features"]

    dados_dashboard, ultimo_mes_historico, mes_previsao_data = (
        preparar_features_operacionais(base_mensal, features)
    )

    x_dash = dados_dashboard[features]
    pred = modelo.predict(x_dash)
    probas = modelo.predict_proba(x_dash)
    classes = modelo.classes_

    dados_dashboard["prioridade_prevista"] = pred

    for i, classe in enumerate(classes):
        dados_dashboard[f"prob_{classe.lower()}"] = probas[:, i]

    def obter_prob_prevista(row):
        return row[f"prob_{row['prioridade_prevista'].lower()}"]

    dados_dashboard["prob_prioridade_prevista"] = dados_dashboard.apply(
        obter_prob_prevista, axis=1
    )

    if "prob_alta" in dados_dashboard.columns:
        dados_dashboard = dados_dashboard.sort_values(
            ["prioridade_prevista", "prob_alta"],
            ascending=[True, False],
        )

    saida = SAIDA_DASHBOARD
    try:
        dados_dashboard.to_csv(saida, index=False)
    except PermissionError:
        saida = SAIDA_OPERACIONAL
        dados_dashboard.to_csv(saida, index=False)

    print(f"Historico usado ate: {ultimo_mes_historico.date()}")
    fim_janela_previsao = mes_previsao_data + pd.DateOffset(months=2)
    inicio_janela_historica = ultimo_mes_historico - pd.DateOffset(months=5)
    print(
        "Indicadores historicos usados: "
        f"{inicio_janela_historica.date()} a {ultimo_mes_historico.date()}"
    )
    print(
        "Prioridade prevista para janela de 3 meses: "
        f"{mes_previsao_data.date()} a {fim_janela_previsao.date()}"
    )
    print(f"Arquivo salvo em: {saida}")
    print(
        dados_dashboard[
            [
                "BEAT_OF_OCCURRENCE",
                "MES_REF",
                "prioridade_prevista",
                "prob_prioridade_prevista",
            ]
        ].head()
    )


if __name__ == "__main__":
    main()
