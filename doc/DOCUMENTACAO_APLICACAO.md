# Logica da Aplicacao SAD de Priorizacao de Police Beats

## Objetivo

Esta aplicacao e um Sistema de Apoio a Decisao (SAD) para priorizar areas policiais de Chicago, chamadas de `police beats`, com base no historico recente de acidentes de transito.

A pergunta central da aplicacao e:

> quais police beats devem receber maior atencao nos proximos meses?

O resultado do modelo e uma classificacao de prioridade:

- `ALTA`;
- `MEDIA`;
- `BAIXA`.

Essa classificacao nao representa um acidente individual. Ela representa o nivel de prioridade de um `police beat` considerando seu comportamento recente e a expectativa para uma janela futura.

## Unidade de analise

A unidade de analise da aplicacao e:

```text
police beat + mes
```

Ou seja, o modelo nao olha para cada acidente separadamente. Primeiro, os acidentes sao agrupados por area policial e por mes. A partir desse agrupamento, sao criados indicadores mensais para cada police beat.

Exemplos de indicadores mensais:

- total de acidentes;
- acidentes a noite;
- acidentes em fim de semana;
- acidentes com pista de risco;
- acidentes com clima de risco;
- acidentes com iluminacao de risco;
- acidentes com lesao;
- acidentes graves;
- mortes.

Essa escolha faz sentido porque a decisao que a dashboard apoia tambem e territorial: decidir quais areas devem ser priorizadas.

## Logica temporal da aplicacao

A parte mais importante da aplicacao e a logica temporal.

O modelo foi construido para usar o passado recente e estimar a prioridade futura.

Em termos simples:

```text
ultimos 6 meses de historico -> prioridade dos proximos 3 meses
```

Isso significa que, para cada police beat, o modelo recebe indicadores calculados a partir dos 6 meses anteriores e tenta prever se aquele beat tera prioridade `ALTA`, `MEDIA` ou `BAIXA` na janela seguinte de 3 meses.

No estado atual da aplicacao:

```text
Historico usado: 01/2026 a 06/2026
Periodo previsto: 07/2026 a 09/2026
```

Portanto, quando a dashboard mostra a previsao para `07/2026 a 09/2026`, ela esta usando os indicadores dos 6 meses imediatamente anteriores.

## Como o passado vira entrada do modelo

Depois que os acidentes sao agregados por police beat e mes, a aplicacao calcula indicadores acumulados dos ultimos 6 meses.

Exemplos:

- total de acidentes nos ultimos 6 meses;
- media mensal de acidentes nos ultimos 6 meses;
- percentual de acidentes a noite;
- percentual de acidentes em fim de semana;
- percentual de acidentes com pista de risco;
- percentual de acidentes com clima de risco;
- velocidade media;
- media de unidades envolvidas.

Essas variaveis formam a entrada do modelo.

Um cuidado importante foi evitar usar informacao do proprio futuro como entrada. Para isso, no treinamento, os indicadores historicos sao calculados olhando para meses anteriores ao periodo previsto.

## Como a prioridade futura foi definida

Para treinar o modelo, foi necessario criar uma variavel-alvo, ou seja, uma resposta que o modelo deveria aprender a prever.

Essa prioridade futura foi criada a partir de um score de severidade. A ideia foi dar pesos maiores para eventos mais graves:

```text
score =
    acidentes
    + peso para acidentes com lesao
    + peso maior para lesoes graves
    + peso ainda maior para mortes
```

Depois, esse score foi somado em uma janela futura de 3 meses.

Assim, para cada police beat em determinado mes, o alvo representa a severidade observada nos 3 meses seguintes.

Em seguida, esse score futuro foi transformado em tres classes:

- `BAIXA`: menor prioridade relativa;
- `MEDIA`: prioridade intermediaria;
- `ALTA`: maior prioridade relativa.

Essa classificacao e relativa ao comportamento historico da propria base.

## Treinamento do modelo

O modelo treinado foi um classificador Random Forest.

A separacao entre treino e teste foi feita de forma temporal. Isso e importante porque a aplicacao tenta simular uma situacao real: aprender com o passado e avaliar no futuro.

Em vez de sortear linhas aleatoriamente, o projeto separou:

```text
dados antigos -> treino
dados mais recentes -> teste
```

Esse tipo de separacao evita uma avaliacao otimista demais e combina melhor com a proposta preditiva da aplicacao.

## Predicao operacional

Depois do treinamento, a aplicacao usa o modelo salvo para gerar uma previsao operacional.

Essa previsao nao usa dados futuros observados. Ela usa apenas a base mensal historica disponivel.

O processo e:

1. Identificar o ultimo mes disponivel na base.
2. Pegar os 6 meses anteriores.
3. Calcular os indicadores de entrada para cada police beat.
4. Aplicar o modelo treinado.
5. Gerar a prioridade prevista para os proximos 3 meses.

No caso atual:

```text
Ultimo mes historico: 06/2026
Entrada do modelo: 01/2026 a 06/2026
Saida prevista: 07/2026 a 09/2026
```

O arquivo usado pela dashboard e:

```text
predicoes_dashboard_operacional.csv
```

## Como o mapa entra na aplicacao

O mapa usa a geometria dos police beats do arquivo:

```text
PoliceBeatDec2012_20260707.csv
```

Essa camada geografica e combinada com as predicoes do modelo usando o identificador do police beat.

A juncao correta e:

```text
BEAT_OF_OCCURRENCE = BEAT_NUM
```

Depois da juncao, cada police beat e colorido de acordo com a prioridade prevista:

- vermelho para `ALTA`;
- laranja para `MEDIA`;
- verde para `BAIXA`.

Assim, o mapa nao mostra apenas onde aconteceram acidentes. Ele mostra quais areas foram classificadas como mais prioritarias pelo modelo.

## O que a dashboard entrega

A dashboard transforma o modelo em uma ferramenta de apoio a decisao.

Ela permite:

- visualizar a prioridade por police beat no mapa;
- ver um ranking das areas mais criticas;
- consultar uma tabela de priorizacao;
- analisar o diagnostico de um beat especifico;
- simular cenarios manuais;
- consultar metricas do modelo.

A aba de simulador segue a mesma logica do modelo: o usuario informa indicadores agregados dos ultimos 6 meses, e a aplicacao estima a prioridade para a janela futura de 3 meses.

## Como interpretar o resultado

A classificacao `ALTA` nao significa que necessariamente ocorrera um acidente grave naquele local. Ela indica que, considerando o historico recente e os padroes aprendidos pelo modelo, aquele police beat merece maior atencao relativa.

Da mesma forma, `BAIXA` nao significa ausencia de risco. Significa apenas que, comparado aos demais beats, aquele local foi classificado como menos prioritario naquele periodo.

Por isso, a aplicacao deve ser entendida como apoio a decisao, nao como substituta da avaliacao tecnica.

## Resumo da logica

O fluxo geral da aplicacao pode ser resumido assim:

```text
Acidentes individuais
        |
        v
Agrupamento por police beat e mes
        |
        v
Indicadores historicos dos ultimos 6 meses
        |
        v
Modelo de classificacao
        |
        v
Prioridade prevista para os proximos 3 meses
        |
        v
Mapa, ranking, diagnostico e simulador
```

Essa e a logica central do projeto: usar o passado recente de cada area para apoiar a priorizacao territorial dos proximos meses.

