# SAD - Priorizacao de Police Beats

Dashboard em Streamlit para priorizar `police beats` de Chicago com base no historico recente de acidentes de transito.

A aplicacao usa indicadores dos ultimos 6 meses para prever a prioridade dos proximos 3 meses:

```text
ALTA | MEDIA | BAIXA
```

Para entender a logica temporal e metodologica, veja:

```text
doc/DOCUMENTACAO_APLICACAO.md
```

## Requisitos

- Python 3.12

## Instalar dependencias

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Executar dashboard

```powershell
streamlit run app.py
```

Acesse:

```text
http://localhost:8501
```

## Abas da aplicacao

- `Visao geral`: mapa, filtros e ranking.
- `Priorizacao`: tabela de beats priorizados.
- `Diagnostico do beat`: detalhes de um beat especifico.
- `Simulador`: classificacao manual a partir de indicadores informados.
- `Modelo`: metricas, matriz de confusao e importancia das variaveis.
