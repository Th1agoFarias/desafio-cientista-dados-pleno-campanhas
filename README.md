Routing Engine — Squad WhatsApp / Prefeitura do Rio de Janeiro
Seleciona os Top-2 telefones por CPF com maior probabilidade de entrega em campanhas de WhatsApp, reduzindo custo operacional sem perder alcance populacional.

Problema
Cada cidadão pode ter vários números de telefone cadastrados em diferentes sistemas da Prefeitura (Saúde, Educação, IPTU etc.). Disparar para todos gera custo desnecessário. O desafio é identificar — antes do disparo — os dois melhores números de cada CPF.

Solução
Score composto calculado em três etapas, refletindo a ordem de prioridade analítica estabelecida na EDA:

1° Frescor do dado (40 %) — decaimento exponencial sobre registro_data_atualizacao. A taxa λ é calibrada internamente e não é um parâmetro configurável.

2° Confiabilidade do sistema (50 %) — Wilson Lower Bound do sistema de origem, corrigindo o viés de volume da taxa bruta. Diferença de ~19 p.p. entre melhor e pior sistema, significativa a p < 0,001 (Kruskal-Wallis).

3° Bônus geográfico (10 %) — DDD 21 (Rio de Janeiro), identificado via EDA pelo valor int64 mascarado dominante (~98 % das linhas).

'_' allowed only in math mode

$$S = 0{,}40 \cdot e^{-\lambda t} + 0{,}50 \cdot WLB_{\text{sistema}} + 0{,}10 \cdot \mathbb{1}[DDD \in \text{DDD_ALVO}]$$

Estrutura do Repositório
desafio-tecnico/
├── README.md
├── requirements.txt
├── .gitignore
└── desafio-cientista-dados-pleno-campanhas/
    ├── data/
    │   ├── base_disparo_mascarado.parquet
    │   └── dim_telefone_mascarado.parquet
    ├── notebooks/
    │   └── analise_whatsapp.ipynb      # notebook principal — único entregável
    └── src/
        ├── utils.py
        ├── scoring.py
        └── plots.py
Como Reproduzir
Pré-requisitos
pip install -r requirements.txt
Dados
Baixe os Parquets do bucket GCS e coloque em desafio-cientista-dados-pleno-campanhas/data/:

https://console.cloud.google.com/storage/browser/case_vagas/whatsapp
Executar
Abra desafio-cientista-dados-pleno-campanhas/notebooks/analise_whatsapp.ipynb
Ajuste os parâmetros na célula setup se necessário (SCORING, AB_P1, AB_P2)
Kernel → Restart & Run All
O arquivo outputs/top2_por_cpf.parquet é gerado ao final da execução.

Arquitetura do Notebook
Parte 1 — EDA e Qualidade das Fontes

Carga dos dois Parquets com auditoria de nulos. As colunas críticas (id_disparo, contato_telefone, status_disparo, telefone_numero, telefone_aparicoes) não possuem nulos. Colunas com alto percentual de ausência (falha_datahora 93 %, id_sessao 39 %) não são utilizadas na análise.

Distribuição de status, desestruturação de telefone_aparicoes via explode + json_normalize e inner join com histórico de disparos. Análise de decaimento temporal: taxa de entrega por faixa etária do dado. Testes não-paramétricos (Mann-Whitney U, Kruskal-Wallis) e regressão OLS confirmam as diferenças entre sistemas e o efeito do tempo.

Parte 2 — Inteligência de Priorização

Ranking de sistemas por Wilson Lower Bound com explicação matemática. Score composto vetorizado (zero apply/iterrows): frescor calculado primeiro, depois Wilson LB, depois bônus DDD. Seleção Top-2 por CPF via groupby + rank.

Parte 3 — Desenho do Experimento A/B

Hipótese unicaudal (H₁: p_B > p_A), métricas primária (Delivery Rate) e secundárias (custo por cidadão, guardrail de opt-out). Cálculo de N mínimo, curva de poder, divisão aleatória de CPFs em grupos A/B com semente fixa. Template de medição com Teste Z para proporções ao fim do teste.

Módulos
src/utils.py
load_parquets(data_dir) — lê os dois Parquets e valida o schema obrigatório.

explode_aparicoes(df_telefones) — expande telefone_aparicoes (lista de dicts) para linhas via explode + json_normalize, vetorizado O(n).

build_merged(df_disparos, df_tels) — inner join pelo número de telefone + flag is_delivered.

resumo_eda(df_d, df_t) — auditoria de nulos, dtypes e cardinalidade por coluna.

src/scoring.py
DECAY_LAMBDA — taxa de decaimento (constante pública); score reduz à metade a cada 180 dias.

wilson_lower_bound_vectorised(sucessos, total) — WLB com IC 95 %, vetorizado O(n).

calcular_score_frescor_vectorised(datas, hoje) — decaimento exponencial vetorizado.

calcular_performance_sistemas(df_merged, min_disparos) — WLB e taxa bruta por sistema.

calcular_scores_batch(df, sistema_scores, hoje, ...) — score composto vetorizado, zero apply/iterrows.

selecionar_top_n(df, id_cidadao_col, score_col, n) — Top-N por CPF via groupby + rank.

calcular_tamanho_amostra(p1, p2, alpha, power) — N mínimo por grupo, Teste Z para proporções.

src/plots.py
configure_plots(dpi) — tema global matplotlib; chamar uma vez no início do notebook.

plot_status_bar(status_counts, palette, output_dir) — bar chart horizontal de distribuição de status com data labels.

plot_decaimento_area_ci(time_perf, output_dir) — bar chart vertical com paleta viridis mostrando taxa de entrega por faixa etária do dado.

plot_ranking_sistemas(performance, output_dir) — bar chart horizontal agrupado: taxa bruta (vermelho) vs Wilson LB (verde) por sistema, com data labels de percentual e volume.

plot_score_kde(df_tels, df_top, top_n, lift, reducao, output_dir) — KDE sobreposto: todos os telefones vs Top-N selecionados.

plot_curva_poder(ab_p1, ab_p2, n_por_grupo, ab_alpha, ab_power, output_dir) — curva de poder em log-scale: N necessário por grupo vs MDE.

plot_ab_treemap(dist_ab, cores, output_dir) — treemap proporcional dos grupos A e B.

Decisões de Design
Por que inner join no merge? Disparos sem telefone na dimensão não têm sistema de origem rastreável. Mantê-los adicionaria ruído sem valor preditivo.

Por que λ não é configurável? A taxa de decaimento é calibrada uma vez pela EDA e fixada. Expô-la como parâmetro aumentaria o risco de ajuste incorreto pelo usuário.

Por que frescor vem antes do Wilson no score? O dataset revelou que todos os dados têm pelo menos 6 meses de defasagem — o frescor é o primeiro filtro de qualidade antes de considerar o sistema.

Por que testes não-paramétricos? is_delivered é Bernoulli. Testes paramétricos pressupõem normalidade e produziriam resultados inválidos.

Por que divisão aleatória simples para A/B? A randomização por .sample(random_state=42) é reproduzível, fácil de auditar e suficiente para o contexto — sem necessidade de infraestrutura adicional de hashing.

Output
outputs/top2_por_cpf.parquet — Top-2 telefones por CPF com score composto e score_frescor intermediário, pronto para consumo pelo motor de disparos.

