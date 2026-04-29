"""
scoring.py
==========
Routing Engine — Squad WhatsApp / Prefeitura do Rio.

Mathematical layers
-------------------
1. Wilson Lower Bound (Bayesian smoothing on system reliability)
2. Exponential Decay  (recency score — data freshness)
3. DDD Geolocation Feature
4. Composite Weighted Score
5. Top-N Selection per CPF
6. A/B Randomisation (SHA-256 hash of CPF)
7. Sample Size Calculator
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_N: int = 2
_W_SISTEMA: float   = 0.50
_W_FRESCOR: float   = 0.40
_W_DDD: float       = 0.10

# Exponential decay rate — score halves every 180 days (implementation detail).
# Not exposed as a parameter: calibrated from EDA and fixed.
DECAY_LAMBDA: float = np.log(2) / 180


# ---------------------------------------------------------------------------
# 1. Wilson Lower Bound
# ---------------------------------------------------------------------------

def wilson_lower_bound_vectorised(
    sucessos: pd.Series,
    total: pd.Series,
    confianca: float = 0.95,
) -> pd.Series:
    """Vectorised Wilson Lower Bound — O(n), no Python loops.

    Corrects raw delivery rate for volume bias.
    WLB = (p̂ + z²/2n - z·√((p̂(1-p̂)/n) + z²/4n²)) / (1 + z²/n)
    """
    if (total == 0).any():
        raise ValueError("total contains zeros — filter before calling.")

    z   = norm.ppf(1 - (1 - confianca) / 2)
    z2  = z ** 2
    n   = total.astype(float)
    p   = sucessos / n
    num = p + z2 / (2 * n) - z * np.sqrt((p * (1 - p) / n) + z2 / (4 * n ** 2))
    den = 1 + z2 / n
    return num / den


def wilson_lower_bound(sucessos: float, total: float, confianca: float = 0.95) -> float:
    """Scalar Wilson LB — for single-pair lookups."""
    if total == 0:
        return 0.0
    return float(wilson_lower_bound_vectorised(
        pd.Series([sucessos]), pd.Series([total]), confianca
    ).iloc[0])


# ---------------------------------------------------------------------------
# 2. Exponential Decay
# ---------------------------------------------------------------------------

def calcular_score_frescor_vectorised(
    registro_data_atualizacao: pd.Series,
    hoje: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """Vectorised exponential decay — O(n).

    Missing dates (NaT) → score = 0.0 (maximally stale).
    score = exp(−DECAY_LAMBDA · t),  t in days.
    """
    if hoje is None:
        hoje = pd.Timestamp.now()

    dates = pd.to_datetime(registro_data_atualizacao, errors="coerce")
    dias  = (hoje - dates).dt.days.fillna(np.inf).clip(lower=0).astype(float)
    return np.exp(-DECAY_LAMBDA * dias)


# ---------------------------------------------------------------------------
# 3. DDD Geolocation Feature
# ---------------------------------------------------------------------------

def calcular_score_ddd_vectorised(
    telefone_ddd: pd.Series,
    ddds_alvo: frozenset = frozenset(),
) -> pd.Series:
    """Vectorised DDD bonus — O(n). Returns 1.0 if DDD in target, else 0.0.

    Works with both string and int64 (masked) DDD values.
    Pass an empty frozenset to disable the bonus.
    """
    if not ddds_alvo:
        return pd.Series(0.0, index=telefone_ddd.index)
    return telefone_ddd.isin(ddds_alvo).astype(float)


# ---------------------------------------------------------------------------
# 4. System-level performance
# ---------------------------------------------------------------------------

def calcular_performance_sistemas(
    df_merged: pd.DataFrame,
    id_sistema_col: str = "id_sistema",
    is_delivered_col: str = "is_delivered",
    id_disparo_col: str = "id_disparo",
    min_disparos: int = 1,
) -> pd.DataFrame:
    """Aggregate per-system delivery performance and compute WLB. O(n) via groupby."""
    perf = df_merged.groupby(id_sistema_col).agg(
        total_disparos=(id_disparo_col, "count"),
        sucessos=(is_delivered_col, "sum"),
    )
    perf = perf[perf["total_disparos"] >= min_disparos].copy()
    perf["taxa_bruta"]    = perf["sucessos"] / perf["total_disparos"]
    perf["wilson_score"]  = wilson_lower_bound_vectorised(
        perf["sucessos"], perf["total_disparos"]
    )
    return perf.sort_values("wilson_score", ascending=False)


# ---------------------------------------------------------------------------
# 5. Composite Score (fully vectorised batch)
# ---------------------------------------------------------------------------

def calcular_scores_batch(
    df: pd.DataFrame,
    sistema_scores: pd.Series,
    hoje: Optional[pd.Timestamp] = None,
    id_sistema_col: str = "id_sistema",
    data_col: str = "registro_data_atualizacao",
    ddd_col: str = "telefone_ddd",
    ddds_alvo: frozenset = frozenset(),
    w_sistema: float = _W_SISTEMA,
    w_frescor: float = _W_FRESCOR,
    w_ddd: float = _W_DDD,
) -> pd.Series:
  
    assert abs(w_sistema + w_frescor + w_ddd - 1.0) < 1e-9, "Weights must sum to 1"

    # 1. Freshness (decay)
    score_frescor = calcular_score_frescor_vectorised(df[data_col], hoje)

    # 2. System reliability (Wilson LB)
    score_sistema = df[id_sistema_col].map(sistema_scores).fillna(0.0)

    # 3. Geographic bonus (DDD)
    score_ddd = calcular_score_ddd_vectorised(df[ddd_col], ddds_alvo)

    return w_frescor * score_frescor + w_sistema * score_sistema + w_ddd * score_ddd


# ---------------------------------------------------------------------------
# 6. Top-N selection per CPF
# ---------------------------------------------------------------------------

def selecionar_top_n(
    df: pd.DataFrame,
    id_cidadao_col: str = "cpf",
    score_col: str = "score",
    n: int = _DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Select top-N phones per citizen, ranked by score. O(n log k)."""
    df = df.copy()
    df["_rank"] = (
        df.groupby(id_cidadao_col)[score_col]
        .rank(method="first", ascending=False)
    )
    return (
        df[df["_rank"] <= n]
        .drop(columns=["_rank"])
        .sort_values([id_cidadao_col, score_col], ascending=[True, False])
    )


# ---------------------------------------------------------------------------
# 7. A/B Randomisation
# ---------------------------------------------------------------------------

def assign_ab_group(
    df: pd.DataFrame,
    cpf_col: str = "cpf",
    salt: str = "squad_whatsapp_v1",
) -> pd.Series:
   
    def _hash_group(cpf_val: str) -> str:
        digest = hashlib.sha256(f"{cpf_val}{salt}".encode()).hexdigest()
        return "A" if int(digest, 16) % 2 == 0 else "B"

    hasher = np.frompyfunc(_hash_group, 1, 1)
    return pd.Series(
        hasher(df[cpf_col].astype(str).values),
        index=df.index,
        name="ab_grupo",
    )


# ---------------------------------------------------------------------------
# 8. Sample Size Calculator
# ---------------------------------------------------------------------------

def calcular_tamanho_amostra(
    p1: float = 0.26,
    p2: float = 0.30,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Minimum sample size per group for a one-tailed Z-test on proportions."""
    from scipy.stats import norm as _norm
    z_alpha = _norm.ppf(1 - alpha)
    z_beta  = _norm.ppf(power)
    p_pool  = (p1 + p2) / 2
    num = (
        z_alpha * np.sqrt(2 * p_pool * (1 - p_pool))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))
