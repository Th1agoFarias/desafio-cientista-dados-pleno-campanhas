"""
plots.py
========
Funções de visualização para o pipeline Squad WhatsApp — Prefeitura do Rio.

Contrato de interface (todas as funções):
    - Recebem DataFrames / Series produzidos pelos módulos de ETL e scoring.
    - Aceitam ``output_dir: Optional[Path]`` — salva PNG quando fornecido.
    - Retornam ``(fig, ax)`` para customização adicional pelo chamador.

Chamar ``configure_plots()`` uma vez no início do notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm as snorm

# Taxa de decaimento — espelha scoring.DECAY_LAMBDA; score cai à metade a cada 180 dias.
_DECAY_LAMBDA: float = np.log(2) / 180

PALETTE_STATUS_PADRAO: Dict[str, str] = {
    "delivered": "#2ecc71",
    "read":      "#3498db",
    "failed":    "#e74c3c",
    "sent":      "#f39c12",
    "processing":"#95a5a6",
}

CORES_AB_PADRAO: Dict[str, str] = {
    "A": "#3498db",
    "B": "#e74c3c",
}

# Mantém os nomes anteriores para retrocompatibilidade
DEFAULT_PALETTE_STATUS = PALETTE_STATUS_PADRAO
DEFAULT_CORES_AB       = CORES_AB_PADRAO


# ---------------------------------------------------------------------------
# Tema global
# ---------------------------------------------------------------------------

def configure_plots(dpi: int = 130) -> None:
    """Aplica o tema visual compartilhado. Chamar uma vez no início do notebook."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi":        dpi,
        "figure.facecolor":  "white",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


def _salvar(fig: plt.Figure, caminho: Optional[Path]) -> None:
    """Salva a figura em disco se o caminho for fornecido."""
    if caminho is not None:
        fig.savefig(caminho, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. Distribuição de status — Barra horizontal
# ---------------------------------------------------------------------------

def plot_status_bar(
    status_counts: pd.DataFrame,
    palette: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Gráfico de barras horizontal com rótulos de dados para a distribuição de status dos disparos.

    Parâmetros
    ----------
    status_counts : DataFrame com colunas ``status`` e ``pct`` (percentual).
    palette       : dicionário mapeando status → cor hex.
    output_dir    : diretório para salvar ``status_bar.png``.
    """
    if palette is None:
        palette = PALETTE_STATUS_PADRAO

    data   = status_counts.sort_values("pct", ascending=True)
    colors = [palette.get(str(s), "#95a5a6") for s in data["status"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(data["status"], data["pct"],
                   color=colors, alpha=0.88, edgecolor="white", linewidth=1.5)

    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{row['pct']:.1f}%", va="center", fontsize=10)

    ax.set_xlabel("Percentual (%)", fontsize=11)
    ax.set_title("Distribuição de Status dos Disparos", fontsize=13, pad=14)
    ax.set_xlim(0, data["pct"].max() * 1.18)
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "status_bar.png")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Ranking de sistemas — Barra horizontal agrupada
# ---------------------------------------------------------------------------

def plot_ranking_sistemas(
    performance: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Barras horizontais agrupadas: taxa bruta vs. Wilson LB por sistema.

    Ordenado por WLB decrescente (melhor sistema no topo). A diferença entre
    as duas barras mostra a correção de viés de volume: gap maior = menor
    volume = maior penalidade estatística pelo Wilson Lower Bound.

    Parâmetros
    ----------
    performance : DataFrame indexado por ``id_sistema`` com colunas
                  ``taxa_bruta``, ``wilson_score``, ``total_disparos``.
    output_dir  : diretório para salvar ``ranking_sistemas.png``.
    """
    data = performance.reset_index().sort_values("wilson_score", ascending=True)
    n    = len(data)
    y    = np.arange(n)
    h    = 0.35

    fig, ax = plt.subplots(figsize=(11, max(4, n * 1.3)))

    bars_raw = ax.barh(y + h / 2, data["taxa_bruta"],   h,
                       color="#e74c3c", alpha=0.75, label="Taxa bruta")
    bars_wlb = ax.barh(y - h / 2, data["wilson_score"], h,
                       color="#2ecc71", alpha=0.85, label="Wilson LB (95% CI)")

    for bar in bars_raw:
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1%}", va="center", fontsize=8, color="#c0392b")

    for bar, (_, row) in zip(bars_wlb, data.iterrows()):
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.1%}   n={int(row['total_disparos']):,}",
                va="center", fontsize=8, color="#27ae60")

    ax.set_yticks(y)
    ax.set_yticklabels(data["id_sistema"].astype(str), fontsize=9)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlim(0, data["taxa_bruta"].max() * 1.30)
    ax.set_xlabel("Taxa de Entrega", fontsize=11)
    ax.set_title(
        "Ranking de Sistemas — Taxa Bruta vs. Wilson Lower Bound\n"
        "(WLB penaliza sistemas com menor volume — diferença entre barras = correção de viés)",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "ranking_sistemas.png")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Decaimento temporal — Barra com paleta viridis
# ---------------------------------------------------------------------------

def plot_decaimento(
    time_perf: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Taxa de entrega por faixa etária do dado com paleta viridis.

    Mostra o 'calor' de cada faixa: dados mais recentes têm maior
    probabilidade de entrega. Cada barra representa uma janela de tempo
    desde a última atualização do telefone no sistema de origem.

    Parâmetros
    ----------
    time_perf  : DataFrame com colunas ``meses_cat`` e ``taxa``.
    output_dir : diretório para salvar ``decaimento.png``.
    """
    x      = np.arange(len(time_perf))
    n      = len(time_perf)
    cmap   = cm.get_cmap("viridis", max(n, 1))
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, time_perf["taxa"], color=colors, width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(time_perf["meses_cat"], fontsize=10)
    ax.set_xlabel("Tempo desde a Última Atualização do Registro", fontsize=11)
    ax.set_ylabel("Taxa de Entrega", fontsize=11)
    ax.set_title("Decaimento da Qualidade do Dado (Tempo vs. Entrega)", fontsize=13, pad=15)
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "decaimento.png")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Distribuição de scores — KDE suavizado
# ---------------------------------------------------------------------------

def plot_score_kde(
    df_tels: pd.DataFrame,
    df_top: pd.DataFrame,
    top_n: int = 2,
    lift: float = 0.0,
    reducao: float = 0.0,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """KDE sobreposto: todos os telefones vs. Top-N selecionados.

    Parâmetros
    ----------
    df_tels   : DataFrame com coluna ``score`` (todos os registros).
    df_top    : DataFrame com coluna ``score`` (Top-N selecionados).
    top_n     : número de telefones selecionados por CPF (usado nos rótulos).
    lift      : lift percentual do Top-N vs. baseline (usado no título).
    reducao   : redução de volume percentual (usado no título).
    output_dir: diretório para salvar ``score_kde.png``.
    """
    scores_all = df_tels["score"].dropna()
    scores_top = df_top["score"].dropna()
    x_range    = np.linspace(scores_all.min(), scores_all.max(), 400)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_range, gaussian_kde(scores_all)(x_range), alpha=0.20, color="steelblue")
    ax.plot(x_range, gaussian_kde(scores_all)(x_range), color="steelblue", linewidth=2,
            label=f"Todos  (μ={scores_all.mean():.3f})")

    ax.fill_between(x_range, gaussian_kde(scores_top)(x_range), alpha=0.40, color="seagreen")
    ax.plot(x_range, gaussian_kde(scores_top)(x_range), color="seagreen", linewidth=2,
            label=f"Top-{top_n}  (μ={scores_top.mean():.3f})")

    ax.axvline(scores_all.mean(), color="steelblue", linestyle="--", alpha=0.6)
    ax.axvline(scores_top.mean(), color="seagreen",  linestyle="--", alpha=0.8)

    ax.set_xlabel("Score Composto", fontsize=11)
    ax.set_ylabel("Densidade", fontsize=11)
    ax.set_title(
        f"Distribuição KDE do Score Composto\n"
        f"(Lift Top-{top_n}: +{lift:.1f}%  |  Redução de volume: {reducao:.1f}%)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "score_kde.png")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Curva de poder A/B
# ---------------------------------------------------------------------------

def plot_curva_poder(
    ab_p1: float,
    ab_p2: float,
    n_por_grupo: int,
    ab_alpha: float = 0.05,
    ab_power: float = 0.80,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Curva de poder: N por grupo vs. efeito mínimo detectável (escala log).

    Parâmetros
    ----------
    ab_p1, ab_p2  : proporções baseline e alvo (MDE).
    n_por_grupo   : N necessário para o cenário base (anotado no gráfico).
    ab_alpha      : nível de significância.
    ab_power      : poder estatístico desejado.
    output_dir    : diretório para salvar ``curva_poder.png``.
    """
    lifts = np.linspace(0.01, 0.15, 300)
    z_a   = snorm.ppf(1 - ab_alpha)
    z_b   = snorm.ppf(ab_power)

    n_vals = []
    for lv in lifts:
        p2     = ab_p1 + lv
        p_pool = (ab_p1 + p2) / 2
        num    = (z_a * np.sqrt(2 * p_pool * (1 - p_pool))
                  + z_b * np.sqrt(ab_p1 * (1 - ab_p1) + p2 * (1 - p2))) ** 2
        n_vals.append(int(np.ceil(num / lv ** 2)))

    mde_pp = (ab_p2 - ab_p1) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lifts * 100, n_vals, color="steelblue", linewidth=2.5)
    ax.fill_between(lifts * 100, n_vals, alpha=0.08, color="steelblue")
    ax.axvline(mde_pp, color="#e74c3c", linestyle="--", linewidth=1.8,
               label=f"MDE = {mde_pp:.0f} p.p.")
    ax.axhline(n_por_grupo, color="#2ecc71", linestyle=":", linewidth=2,
               label=f"n = {n_por_grupo:,} por grupo")
    ax.scatter([mde_pp], [n_por_grupo], color="#e74c3c", s=120, zorder=5)
    ax.annotate(f"  n = {n_por_grupo:,}", (mde_pp, n_por_grupo),
                fontsize=10, color="#e74c3c")

    ax.set_xlabel("Lift mínimo detectável (p.p.)", fontsize=11)
    ax.set_ylabel("N por grupo necessário", fontsize=11)
    ax.set_title(
        f"Curva de Poder — N vs. MDE\n"
        f"(α={ab_alpha:.0%}, poder={ab_power:.0%}, baseline={ab_p1:.0%})",
        fontsize=12,
    )
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "curva_poder.png")
    plt.show()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Distribuição A/B — Treemap
# ---------------------------------------------------------------------------

def plot_ab_treemap(
    dist_ab: pd.Series,
    cores: Optional[Dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Treemap proporcional mostrando os tamanhos dos grupos A e B.

    Parâmetros
    ----------
    dist_ab    : Series com rótulos de grupo como índice e contagens como valores.
    cores      : dicionário mapeando rótulo → cor hex.
    output_dir : diretório para salvar ``ab_treemap.png``.
    """
    if cores is None:
        cores = CORES_AB_PADRAO

    total = dist_ab.sum()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    x_pos = 0.0
    for grupo, count in dist_ab.items():
        width = count / total
        rect  = mpatches.FancyBboxPatch(
            (x_pos + 0.005, 0.05), width - 0.010, 0.90,
            boxstyle="round,pad=0.01",
            facecolor=cores.get(grupo, "#aaaaaa"),
            edgecolor="white", linewidth=3, alpha=0.88,
        )
        ax.add_patch(rect)
        ax.text(x_pos + width / 2, 0.5,
                f"Grupo {grupo}\n{count:,}\n({count/total:.1%})",
                ha="center", va="center",
                fontsize=15, color="white", fontweight="bold")
        x_pos += width

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Distribuição A/B por Grupo", fontsize=13, pad=12)
    plt.tight_layout()
    _salvar(fig, output_dir and output_dir / "ab_treemap.png")
    plt.show()
    return fig, ax
