# Desafio Técnico — Cientista de Dados Pleno / Squad WhatsApp
**Prefeitura do Rio de Janeiro**

Solução para o desafio de criação da **Inteligência de Escolha de Canais WhatsApp**: identificar quais fontes de dados são mais confiáveis ("quentes") e selecionar os **Top-2 telefones por CPF** com maior probabilidade de entrega, reduzindo custos operacionais sem perder alcance populacional.

---

## Problema

O Registro Municipal Integrado (RMI) consolida dados de múltiplos sistemas (Saúde, Educação, Assistência Social, IPTU etc.). Um mesmo cidadão pode ter vários telefones vinculados ao seu CPF — muitos antigos ou desatualizados. Disparar mensagens para todos gera custo desnecessário e compromete janelas de comunicação preciosas.

---

## Solução

Score composto que combina três sinais para priorizar os melhores contatos:

**1° Frescor do dado — peso 40 %**
Decaimento exponencial sobre `registro_data_atualizacao`. Dados com 180 dias de defasagem recebem score 0,50; com 2 anos recebem ~0,06.

**2° Confiabilidade do sistema — peso 50 %**
Wilson Lower Bound (WLB) do sistema de origem. Corrige o viés de volume da taxa bruta: sistemas com poucas tentativas recebem penalidade estatística. Diferença de ~19 p.p. entre melhor e pior sistema, significativa a p < 0,001 (Kruskal-Wallis).

**3° Bônus geográfico — peso 10 %**
DDD 21 (Rio de Janeiro), identificado via EDA pelo valor int64 mascarado dominante (~98 % das linhas).

```
S = 0,40 × exp(-λt)  +  0,50 × WLB(sistema)  +  0,10 × I[DDD = 21]
```

---

## Estrutura do Repositório

```
desafio-cientista-dados-pleno-campanhas/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── analise_whatsapp.ipynb    # notebook principal — entregável do desafio
└── src/
    ├── utils.py                  # carga, explode, merge, auditoria
    ├── scoring.py                # WLB, decaimento, score composto, Top-N
    └── plots.py                  # todas as visualizações
```

---

## Como Reproduzir

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Baixar os dados

Baixe os Parquets do bucket GCS e coloque em `data/` (na raiz do projeto):

```
https://console.cloud.google.com/storage/browser/case_vagas/whatsapp
```

Arquivos necessários:
- `base_disparo_mascarado.parquet`
- `dim_telefone_mascarado.parquet`

### 3. Executar o notebook

1. Abra `notebooks/analise_whatsapp.ipynb`
2. Ajuste os parâmetros na célula `setup` se necessário (`SCORING`, `AB_P1`, `AB_P2`)
3. `Kernel → Restart & Run All`

O arquivo `outputs/top2_por_cpf.parquet` é gerado automaticamente ao final da execução.

---

## Conteúdo do Notebook

### Parte 1 — EDA e Qualidade das Fontes

- Carga dos dois Parquets com auditoria completa de nulos e cardinalidade
- As colunas críticas (`id_disparo`, `contato_telefone`, `status_disparo`, `telefone_numero`, `telefone_aparicoes`) não possuem nulos
- Distribuição de status dos disparos
- Desestruturação de `telefone_aparicoes` (lista de dicts) via `explode + json_normalize` — vetorizado O(n)
- Inner join com histórico de disparos
- **Janela de Atualidade**: taxa de entrega por faixa etária do dado — achado crítico: todos os dados têm pelo menos 6 meses de defasagem
- Testes não-paramétricos: Mann-Whitney U e Kruskal-Wallis confirmam diferenças significativas entre sistemas
- Regressão OLS log-linear quantifica o coeficiente do decaimento temporal

### Parte 2 — Inteligência de Priorização

- Ranking de sistemas por Wilson Lower Bound com explicação matemática detalhada
- Por que o sistema X é melhor que o Y: comparação concreta WLB vs taxa bruta
- Score composto vetorizado (zero `apply`/`iterrows`): frescor → Wilson LB → DDD
- Seleção Top-2 por CPF via `groupby + rank` — O(n log k)

### Parte 3 — Desenho do Experimento A/B

- Hipótese unicaudal: H₁: p_B > p_A (taxa de entrega do algoritmo > aleatório)
- Métrica primária: Delivery Rate (`status = delivered`)
- Métricas secundárias: custo por cidadão engajado e guardrail de opt-out (< 2 %)
- Cálculo de N mínimo: 1.557 por grupo (~15 dias com 10 % do volume diário)
- Curva de poder e divisão aleatória de CPFs em grupos A/B com semente fixa
- Template de medição com Teste Z para proporções ao fim do teste

---

## Decisões

**Por que inner join no merge?**
Disparos sem telefone na dimensão não têm sistema de origem rastreável. Mantê-los adicionaria ruído sem valor preditivo para o score.

**Por que Wilson Lower Bound e não taxa bruta?**
A taxa bruta favorece sistemas com poucas tentativas (1 sucesso em 1 tentativa = 100 %). O WLB aplica penalidade proporcional à incerteza estatística, corrigindo o viés de seleção alertado no enunciado do desafio.

**Por que frescor vem antes do Wilson no score?**
A EDA revelou que todos os dados têm pelo menos 6 meses de defasagem — o frescor é o primeiro filtro natural de qualidade antes de considerar a confiabilidade do sistema.

**Por que testes não-paramétricos?**
`is_delivered` é binária (distribuição Bernoulli). Testes paramétricos pressupõem normalidade e produziriam resultados inválidos neste contexto.

---

## Output

`outputs/top2_por_cpf.parquet` — Top-2 telefones por CPF com score composto e `score_frescor` intermediário, pronto para consumo pelo motor de disparos.
