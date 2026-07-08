import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).parent
PREDICOES_OPERACIONAL_PATH = BASE_DIR / "predicoes_dashboard_operacional.csv"
PREDICOES_PATH = BASE_DIR / "predicoes_dashboard_ultimo_mes.csv"
BEATS_PATH = BASE_DIR / "PoliceBeatDec2012_20260707.csv"
BASE_MENSAL_PATH = BASE_DIR / "base_beat_mes.csv"
IMPORTANCIA_PATH = BASE_DIR / "importancia_features.csv"
MODELO_PATH = BASE_DIR / "modelo_prioridade_beat.pkl"

CORES_PRIORIDADE = {
    "ALTA": "#c1121f",
    "MEDIA": "#f4a261",
    "BAIXA": "#6a994e",
    "SEM DADO": "#adb5bd",
}

ORDEM_PRIORIDADE = {"ALTA": 0, "MEDIA": 1, "BAIXA": 2, "SEM DADO": 3}

NOMES_FEATURES = {
    "total_acidentes_ult_6m": "Total de acidentes nos ultimos 6 meses",
    "media_acidentes_mensal_ult_6m": "Media mensal de acidentes",
    "perc_acidentes_noite_ult_6m": "% acidentes a noite",
    "perc_acidentes_fim_semana_ult_6m": "% acidentes no fim de semana",
    "perc_iluminacao_risco_ult_6m": "% com iluminacao de risco",
    "perc_pista_risco_ult_6m": "% com pista de risco",
    "perc_defeito_via_ult_6m": "% com defeito na via",
    "perc_dispositivo_problema_ult_6m": "% com problema em dispositivo",
    "perc_clima_risco_ult_6m": "% com clima de risco",
    "velocidade_media_ult_6m": "Velocidade media registrada",
    "num_units_medio_ult_6m": "Media de unidades envolvidas",
    "mes_previsao": "Mes da previsao",
}


st.set_page_config(
    page_title="SAD - Priorizacao de Police Beats",
    page_icon=":material/traffic:",
    layout="wide",
)


def formatar_percentual(valor):
    if pd.isna(valor):
        return "-"
    return f"{valor * 100:.1f}%"


def formatar_numero(valor, casas=0):
    if pd.isna(valor):
        return "-"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def split_top_level(text):
    grupos = []
    nivel = 0
    inicio = 0

    for idx, char in enumerate(text):
        if char == "(":
            nivel += 1
        elif char == ")":
            nivel -= 1
        elif char == "," and nivel == 0:
            grupos.append(text[inicio:idx].strip())
            inicio = idx + 1

    grupos.append(text[inicio:].strip())
    return [grupo for grupo in grupos if grupo]


def remover_parenteses_externos(text):
    texto = text.strip()
    if texto.startswith("(") and texto.endswith(")"):
        return texto[1:-1].strip()
    return texto


def parse_ring(text):
    pontos = []
    for ponto in text.split(","):
        valores = ponto.strip().split()
        if len(valores) >= 2:
            pontos.append([float(valores[0]), float(valores[1])])
    return pontos


def wkt_multipolygon_to_coordinates(wkt):
    texto = str(wkt).strip()

    if texto.startswith("MULTIPOLYGON"):
        conteudo = remover_parenteses_externos(texto[len("MULTIPOLYGON") :].strip())
        polygons = []
        for polygon_text in split_top_level(conteudo):
            polygon_inner = remover_parenteses_externos(polygon_text)
            rings = []
            for ring_text in split_top_level(polygon_inner):
                ring_inner = remover_parenteses_externos(ring_text)
                rings.append(parse_ring(ring_inner))
            polygons.append(rings)
        return polygons

    if texto.startswith("POLYGON"):
        conteudo = remover_parenteses_externos(texto[len("POLYGON") :].strip())
        rings = []
        for ring_text in split_top_level(conteudo):
            rings.append(parse_ring(remover_parenteses_externos(ring_text)))
        return [rings]

    raise ValueError("Geometria WKT nao suportada.")


@st.cache_data(show_spinner=False)
def carregar_predicoes():
    caminho = PREDICOES_OPERACIONAL_PATH if PREDICOES_OPERACIONAL_PATH.exists() else PREDICOES_PATH
    pred = pd.read_csv(caminho)
    pred["BEAT_OF_OCCURRENCE"] = pred["BEAT_OF_OCCURRENCE"].astype(str)
    pred["MES_REF"] = pd.to_datetime(pred["MES_REF"], errors="coerce")

    if "ultimo_mes_historico" in pred.columns:
        pred["ultimo_mes_historico"] = pd.to_datetime(
            pred["ultimo_mes_historico"], errors="coerce"
        )

    for coluna in ["prob_alta", "prob_media", "prob_baixa"]:
        if coluna not in pred.columns:
            pred[coluna] = 0.0

    pred["prioridade_prevista"] = pred["prioridade_prevista"].fillna("SEM DADO")
    pred["ordem_prioridade"] = pred["prioridade_prevista"].map(ORDEM_PRIORIDADE).fillna(9)
    return pred.sort_values(["ordem_prioridade", "prob_alta"], ascending=[True, False])


@st.cache_data(show_spinner=False)
def carregar_base_mensal():
    mensal = pd.read_csv(BASE_MENSAL_PATH)
    mensal["BEAT_OF_OCCURRENCE"] = mensal["BEAT_OF_OCCURRENCE"].astype(str)
    mensal["MES_REF"] = pd.to_datetime(mensal["MES_REF"], errors="coerce")
    return mensal


@st.cache_data(show_spinner=False)
def carregar_importancias():
    if not IMPORTANCIA_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])

    imp = pd.read_csv(IMPORTANCIA_PATH)
    imp["feature_label"] = imp["feature"].map(NOMES_FEATURES).fillna(imp["feature"])
    return imp.sort_values("importance", ascending=True)


@st.cache_data(show_spinner=False)
def carregar_geojson():
    beats = pd.read_csv(BEATS_PATH)
    beats["BEAT_NUM"] = beats["BEAT_NUM"].astype(str)

    features = []
    for row in beats.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "id": str(row.BEAT_NUM),
                "properties": {
                    "DISTRICT": str(row.DISTRICT),
                    "SECTOR": str(row.SECTOR),
                    "BEAT": str(row.BEAT),
                    "BEAT_NUM": str(row.BEAT_NUM),
                },
                "geometry": {
                    "type": "MultiPolygon",
                    "coordinates": wkt_multipolygon_to_coordinates(row.the_geom),
                },
            }
        )

    return {"type": "FeatureCollection", "features": features}


@st.cache_data(show_spinner=False)
def carregar_dimensao_beats():
    beats = pd.read_csv(BEATS_PATH, usecols=["DISTRICT", "SECTOR", "BEAT", "BEAT_NUM"])
    beats["BEAT_NUM"] = beats["BEAT_NUM"].astype(str)
    beats["DISTRICT"] = beats["DISTRICT"].astype(str)
    beats["SECTOR"] = beats["SECTOR"].astype(str)
    return beats.rename(columns={"BEAT_NUM": "BEAT_OF_OCCURRENCE"})


@st.cache_resource(show_spinner=False)
def carregar_modelo():
    artefatos = joblib.load(MODELO_PATH)
    return artefatos["modelo"], artefatos["features"]


def prever_prioridade_manual(modelo, features, entrada):
    linha = pd.DataFrame([entrada])
    linha = linha.reindex(columns=features, fill_value=0).fillna(0)

    classe = modelo.predict(linha)[0]
    probas = modelo.predict_proba(linha)[0]

    resultado = entrada.copy()
    resultado["prioridade_prevista"] = classe

    for classe_modelo, probabilidade in zip(modelo.classes_, probas):
        resultado[f"prob_{classe_modelo.lower()}"] = probabilidade

    resultado["prob_prioridade_prevista"] = resultado[f"prob_{classe.lower()}"]
    return pd.Series(resultado)


def aplicar_filtros(pred, dimensao):
    dados = pred.merge(dimensao, on="BEAT_OF_OCCURRENCE", how="left")

    with st.sidebar:
        st.header("Filtros")

        prioridades = st.multiselect(
            "Prioridade prevista",
            ["ALTA", "MEDIA", "BAIXA"],
            default=["ALTA", "MEDIA", "BAIXA"],
        )

        distritos = sorted(dados["DISTRICT"].dropna().unique())
        distrito_sel = st.multiselect("Distrito", distritos, default=distritos)

        setores = sorted(dados["SECTOR"].dropna().unique())
        setor_sel = st.multiselect("Setor", setores, default=setores)

        limite_ranking = st.slider("Itens no ranking", 5, 30, 15)

    filtrado = dados[
        dados["prioridade_prevista"].isin(prioridades)
        & dados["DISTRICT"].isin(distrito_sel)
        & dados["SECTOR"].isin(setor_sel)
    ].copy()

    return filtrado, limite_ranking


def card_metrica(label, valor, ajuda=None):
    st.metric(label, valor, help=ajuda)


def construir_mapa(dados, geojson):
    if dados.empty:
        st.info("Nenhum beat atende aos filtros selecionados.")
        return

    mapa_df = dados.copy()
    mapa_df["prob_alta_pct"] = mapa_df["prob_alta"] * 100
    mapa_df["media_mensal"] = mapa_df["media_acidentes_mensal_ult_6m"]

    fig = px.choropleth_map(
        mapa_df,
        geojson=geojson,
        locations="BEAT_OF_OCCURRENCE",
        featureidkey="properties.BEAT_NUM",
        color="prioridade_prevista",
        color_discrete_map=CORES_PRIORIDADE,
        category_orders={"prioridade_prevista": ["ALTA", "MEDIA", "BAIXA"]},
        hover_name="BEAT_OF_OCCURRENCE",
        hover_data={
            "BEAT_OF_OCCURRENCE": False,
            "DISTRICT": True,
            "SECTOR": True,
            "prioridade_prevista": True,
            "prob_alta_pct": ":.1f",
            "total_acidentes_ult_6m": ":.0f",
            "media_mensal": ":.1f",
            "perc_acidentes_noite_ult_6m": ":.1%",
            "perc_pista_risco_ult_6m": ":.1%",
        },
        center={"lat": 41.84, "lon": -87.68},
        zoom=9.2,
        opacity=0.78,
        map_style="carto-positron",
    )

    fig.update_layout(
        height=670,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend_title_text="Prioridade",
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, width="stretch")


def construir_ranking(dados, limite):
    ranking = dados.sort_values("prob_alta", ascending=False).head(limite).copy()
    ranking["Prob. alta"] = ranking["prob_alta"].map(formatar_percentual)
    ranking["Acidentes 6m"] = ranking["total_acidentes_ult_6m"].map(
        lambda v: formatar_numero(v, 0)
    )
    ranking["Media mensal"] = ranking["media_acidentes_mensal_ult_6m"].map(
        lambda v: formatar_numero(v, 1)
    )

    st.dataframe(
        ranking[
            [
                "BEAT_OF_OCCURRENCE",
                "DISTRICT",
                "SECTOR",
                "prioridade_prevista",
                "Prob. alta",
                "Acidentes 6m",
                "Media mensal",
            ]
        ].rename(
            columns={
                "BEAT_OF_OCCURRENCE": "Beat",
                "DISTRICT": "Distrito",
                "SECTOR": "Setor",
                "prioridade_prevista": "Prioridade",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def principais_fatores(row):
    fatores = {
        "Noite": row.get("perc_acidentes_noite_ult_6m", 0),
        "Fim de semana": row.get("perc_acidentes_fim_semana_ult_6m", 0),
        "Iluminacao": row.get("perc_iluminacao_risco_ult_6m", 0),
        "Pista": row.get("perc_pista_risco_ult_6m", 0),
        "Defeito viario": row.get("perc_defeito_via_ult_6m", 0),
        "Dispositivo": row.get("perc_dispositivo_problema_ult_6m", 0),
        "Clima": row.get("perc_clima_risco_ult_6m", 0),
    }
    return sorted(fatores.items(), key=lambda item: item[1], reverse=True)[:3]


def recomendacao_operacional(row):
    prioridade = row["prioridade_prevista"]
    fatores = [nome for nome, valor in principais_fatores(row) if valor > 0]

    if prioridade == "ALTA":
        prefixo = "Priorizar intervencao preventiva"
    elif prioridade == "MEDIA":
        prefixo = "Monitorar e planejar acao focalizada"
    else:
        prefixo = "Manter acompanhamento"

    if not fatores:
        return f"{prefixo}, com foco inicial no volume historico de acidentes."

    return f"{prefixo}, observando principalmente: {', '.join(fatores).lower()}."


def aba_visao_geral(dados, geojson, limite_ranking):
    total_beats = dados["BEAT_OF_OCCURRENCE"].nunique()
    alta = (dados["prioridade_prevista"] == "ALTA").sum()
    media = (dados["prioridade_prevista"] == "MEDIA").sum()
    baixa = (dados["prioridade_prevista"] == "BAIXA").sum()
    prob_max = dados["prob_alta"].max() if not dados.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        card_metrica("Beats analisados", formatar_numero(total_beats))
    with c2:
        card_metrica("Prioridade alta", formatar_numero(alta))
    with c3:
        card_metrica("Prioridade media", formatar_numero(media))
    with c4:
        card_metrica("Prioridade baixa", formatar_numero(baixa))
    with c5:
        card_metrica("Maior prob. alta", formatar_percentual(prob_max))

    mapa_col, ranking_col = st.columns([2.2, 1])
    with mapa_col:
        construir_mapa(dados, geojson)
    with ranking_col:
        st.subheader("Ranking de atencao")
        construir_ranking(dados, limite_ranking)


def aba_priorizacao(dados):
    tabela = dados.sort_values(["ordem_prioridade", "prob_alta"], ascending=[True, False])
    tabela = tabela.copy()
    tabela["prob_alta_fmt"] = tabela["prob_alta"].map(formatar_percentual)
    tabela["prob_prevista_fmt"] = tabela["prob_prioridade_prevista"].map(formatar_percentual)
    tabela["recomendacao"] = tabela.apply(recomendacao_operacional, axis=1)

    st.dataframe(
        tabela[
            [
                "BEAT_OF_OCCURRENCE",
                "DISTRICT",
                "SECTOR",
                "prioridade_prevista",
                "prob_alta_fmt",
                "prob_prevista_fmt",
                "total_acidentes_ult_6m",
                "media_acidentes_mensal_ult_6m",
                "recomendacao",
            ]
        ].rename(
            columns={
                "BEAT_OF_OCCURRENCE": "Beat",
                "DISTRICT": "Distrito",
                "SECTOR": "Setor",
                "prioridade_prevista": "Prioridade",
                "prob_alta_fmt": "Prob. alta",
                "prob_prevista_fmt": "Confianca da classe",
                "total_acidentes_ult_6m": "Acidentes 6m",
                "media_acidentes_mensal_ult_6m": "Media mensal",
                "recomendacao": "Apoio a decisao",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def aba_diagnostico(dados, mensal):
    if dados.empty:
        st.info("Nenhum beat disponivel para diagnostico com os filtros atuais.")
        return

    beats = dados.sort_values("prob_alta", ascending=False)["BEAT_OF_OCCURRENCE"].tolist()
    beat = st.selectbox("Police beat", beats)
    row = dados[dados["BEAT_OF_OCCURRENCE"] == beat].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_metrica("Prioridade", row["prioridade_prevista"])
    with c2:
        card_metrica("Prob. alta", formatar_percentual(row["prob_alta"]))
    with c3:
        card_metrica("Acidentes 6m", formatar_numero(row["total_acidentes_ult_6m"]))
    with c4:
        card_metrica("Media mensal", formatar_numero(row["media_acidentes_mensal_ult_6m"], 1))

    st.info(recomendacao_operacional(row))

    hist = mensal[mensal["BEAT_OF_OCCURRENCE"] == beat].sort_values("MES_REF").tail(24)
    fig_linha = px.line(
        hist,
        x="MES_REF",
        y="total_acidentes_mes",
        markers=True,
        labels={"MES_REF": "Mes", "total_acidentes_mes": "Acidentes"},
    )
    fig_linha.update_layout(height=320, margin={"r": 10, "t": 20, "l": 10, "b": 10})

    fatores_df = pd.DataFrame(
        principais_fatores(row),
        columns=["Fator", "Percentual"],
    ).sort_values("Percentual")
    fig_fatores = px.bar(
        fatores_df,
        x="Percentual",
        y="Fator",
        orientation="h",
        text=fatores_df["Percentual"].map(formatar_percentual),
        labels={"Percentual": "Percentual nos ultimos 6 meses", "Fator": ""},
        color="Percentual",
        color_continuous_scale=["#d8f3dc", "#f4a261", "#c1121f"],
    )
    fig_fatores.update_layout(
        height=320,
        margin={"r": 10, "t": 20, "l": 10, "b": 10},
        coloraxis_showscale=False,
    )

    col_hist, col_fatores = st.columns(2)
    with col_hist:
        st.subheader("Historico recente")
        st.plotly_chart(fig_linha, width="stretch")
    with col_fatores:
        st.subheader("Fatores dominantes")
        st.plotly_chart(fig_fatores, width="stretch")

    probs = pd.DataFrame(
        {
            "Classe": ["ALTA", "MEDIA", "BAIXA"],
            "Probabilidade": [row["prob_alta"], row["prob_media"], row["prob_baixa"]],
        }
    )
    fig_probs = px.bar(
        probs,
        x="Classe",
        y="Probabilidade",
        color="Classe",
        color_discrete_map=CORES_PRIORIDADE,
        text=probs["Probabilidade"].map(formatar_percentual),
    )
    fig_probs.update_layout(
        height=300,
        yaxis_tickformat=".0%",
        showlegend=False,
        margin={"r": 10, "t": 20, "l": 10, "b": 10},
    )
    st.subheader("Confianca da classificacao")
    st.plotly_chart(fig_probs, width="stretch")


def aba_modelo(importancias):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_metrica("Acuracia teste", "78,3%")
    with c2:
        card_metrica("Precision ALTA", "74,7%")
    with c3:
        card_metrica("Recall ALTA", "78,5%")
    with c4:
        card_metrica("F1 ALTA", "76,5%")

    if not importancias.empty:
        fig = px.bar(
            importancias,
            x="importance",
            y="feature_label",
            orientation="h",
            labels={"importance": "Importancia", "feature_label": ""},
            color="importance",
            color_continuous_scale=["#cad2c5", "#f4a261", "#c1121f"],
        )
        fig.update_layout(
            height=470,
            margin={"r": 10, "t": 20, "l": 10, "b": 10},
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, width="stretch")

    st.caption(
        "O modelo usa o historico dos ultimos 6 meses por police beat para estimar "
        "a prioridade dos proximos 3 meses. A previsao deve apoiar triagem e "
        "priorizacao, nao substituir avaliacao tecnica."
    )


def aba_simulador():
    modelo, features = carregar_modelo()

    st.subheader("Simulador de prioridade")

    col_base, col_contexto, col_operacao = st.columns(3)

    with col_base:
        total_acidentes = st.number_input(
            "Acidentes nos ultimos 6 meses",
            min_value=0,
            value=180,
            step=1,
        )
        velocidade_media = st.number_input(
            "Velocidade media",
            min_value=0.0,
            value=30.0,
            step=1.0,
        )
        num_units_medio = st.number_input(
            "Media de unidades envolvidas",
            min_value=0.0,
            value=2.0,
            step=0.1,
        )
        mes_previsao = st.selectbox(
            "Mes inicial da janela prevista",
            list(range(1, 13)),
            index=6,
            format_func=lambda mes: f"{mes:02d}",
        )

    with col_contexto:
        perc_noite = st.slider("% acidentes a noite", 0, 100, 30)
        perc_fim_semana = st.slider("% acidentes no fim de semana", 0, 100, 25)
        perc_clima = st.slider("% com clima de risco", 0, 100, 15)
        perc_iluminacao = st.slider("% com iluminacao de risco", 0, 100, 35)

    with col_operacao:
        perc_pista = st.slider("% com pista de risco", 0, 100, 20)
        perc_defeito = st.slider("% com defeito na via", 0, 100, 3)
        perc_dispositivo = st.slider("% com problema em dispositivo", 0, 100, 2)

    entrada = {
        "total_acidentes_ult_6m": float(total_acidentes),
        "media_acidentes_mensal_ult_6m": float(total_acidentes) / 6,
        "perc_acidentes_noite_ult_6m": perc_noite / 100,
        "perc_acidentes_fim_semana_ult_6m": perc_fim_semana / 100,
        "perc_iluminacao_risco_ult_6m": perc_iluminacao / 100,
        "perc_pista_risco_ult_6m": perc_pista / 100,
        "perc_defeito_via_ult_6m": perc_defeito / 100,
        "perc_dispositivo_problema_ult_6m": perc_dispositivo / 100,
        "perc_clima_risco_ult_6m": perc_clima / 100,
        "velocidade_media_ult_6m": float(velocidade_media),
        "num_units_medio_ult_6m": float(num_units_medio),
        "mes_previsao": int(mes_previsao),
    }

    resultado = prever_prioridade_manual(modelo, features, entrada)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card_metrica("Prioridade simulada", resultado["prioridade_prevista"])
    with c2:
        card_metrica("Prob. alta", formatar_percentual(resultado.get("prob_alta", 0)))
    with c3:
        card_metrica("Confianca da classe", formatar_percentual(resultado["prob_prioridade_prevista"]))
    with c4:
        card_metrica("Media mensal", formatar_numero(resultado["media_acidentes_mensal_ult_6m"], 1))

    st.info(recomendacao_operacional(resultado))

    probs = pd.DataFrame(
        {
            "Classe": ["ALTA", "MEDIA", "BAIXA"],
            "Probabilidade": [
                resultado.get("prob_alta", 0),
                resultado.get("prob_media", 0),
                resultado.get("prob_baixa", 0),
            ],
        }
    )
    fig_probs = px.bar(
        probs,
        x="Classe",
        y="Probabilidade",
        color="Classe",
        color_discrete_map=CORES_PRIORIDADE,
        text=probs["Probabilidade"].map(formatar_percentual),
    )
    fig_probs.update_layout(
        height=320,
        yaxis_tickformat=".0%",
        showlegend=False,
        margin={"r": 10, "t": 20, "l": 10, "b": 10},
    )
    st.plotly_chart(fig_probs, width="stretch")


def main():
    st.title("Sistema de Apoio a Decisao para Priorizacao de Police Beats")

    pred = carregar_predicoes()
    dimensao = carregar_dimensao_beats()
    geojson = carregar_geojson()
    mensal = carregar_base_mensal()
    importancias = carregar_importancias()

    dados, limite_ranking = aplicar_filtros(pred, dimensao)

    mes_previsao = pred["MES_REF"].max()
    fim_janela_previsao = mes_previsao + pd.DateOffset(months=2)
    if "ultimo_mes_historico" in pred.columns:
        ultimo_hist = pred["ultimo_mes_historico"].max()
        inicio_janela_historica = ultimo_hist - pd.DateOffset(months=5)
        st.caption(
            f"Prioridade prevista para {mes_previsao:%m/%Y} a "
            f"{fim_janela_previsao:%m/%Y}, usando indicadores dos ultimos "
            f"6 meses: {inicio_janela_historica:%m/%Y} a {ultimo_hist:%m/%Y}."
        )
    else:
        st.caption(
            f"Prioridade prevista para {mes_previsao:%m/%Y} a "
            f"{fim_janela_previsao:%m/%Y}."
        )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Visao geral", "Priorizacao", "Diagnostico do beat", "Simulador", "Modelo"]
    )

    with tab1:
        aba_visao_geral(dados, geojson, limite_ranking)
    with tab2:
        aba_priorizacao(dados)
    with tab3:
        aba_diagnostico(dados, mensal)
    with tab4:
        aba_simulador()
    with tab5:
        aba_modelo(importancias)


if __name__ == "__main__":
    main()
