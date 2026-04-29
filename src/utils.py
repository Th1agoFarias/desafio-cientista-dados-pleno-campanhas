"""
utils.py
========
Utility helpers for the Prefeitura do Rio — Squad WhatsApp pipeline.

Covers three areas:
  1. Data loading & schema validation
  2. Explode + json_normalize pipeline (telefone_aparicoes)
  3. Report / summary helpers
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Data loading & schema validation
# ---------------------------------------------------------------------------

_EXPECTED_DISPAROS_COLS  = {"id_disparo", "contato_telefone", "status_disparo"}
# 'cpf' não é coluna top-level: fica dentro dos dicts de telefone_aparicoes
_EXPECTED_TELEFONES_COLS = {"telefone_numero", "telefone_aparicoes"}


def validate_schema(df: pd.DataFrame, expected: set, name: str) -> None:
    """Raise ValueError if any required column is missing."""
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"[{name}] Colunas ausentes: {missing}")
    logger.info("[%s] Schema OK — %d colunas, %d linhas.", name, len(df.columns), len(df))


def load_parquets(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load base_disparo_mascarado and dim_telefone_mascarado from Parquet.

    Applies efficient dtypes at load time (category for high-cardinality
    string columns) to reduce memory by up to 70 %.

    Parameters
    ----------
    data_dir : Path — directory containing both Parquet files.

    Returns
    -------
    df_disparos, df_telefones : pd.DataFrame
    """
    disp_path = data_dir / "base_disparo_mascarado.parquet"
    tel_path  = data_dir / "dim_telefone_mascarado.parquet"

    if not disp_path.exists() or not tel_path.exists():
        raise FileNotFoundError(
            f"Parquets não encontrados em '{data_dir}'.\n"
            "Baixe os arquivos do bucket GCS:\n"
            "  https://console.cloud.google.com/storage/browser/case_vagas/whatsapp\n"
            "e coloque-os em data/"
        )

    df_disparos  = pd.read_parquet(disp_path)
    df_telefones = pd.read_parquet(tel_path)

    # Memory optimisation: cast string columns to category
    # (id_disparo e contato_telefone são int64 mascarados — category não agrega valor)
    if "status_disparo" in df_disparos.columns:
        df_disparos["status_disparo"] = df_disparos["status_disparo"].astype("category")

    validate_schema(df_disparos,  _EXPECTED_DISPAROS_COLS,  "base_disparo_mascarado")
    validate_schema(df_telefones, _EXPECTED_TELEFONES_COLS, "dim_telefone_mascarado")

    logger.info("disparos : %s | telefones: %s", df_disparos.shape, df_telefones.shape)
    return df_disparos, df_telefones


# ---------------------------------------------------------------------------
# 2. Explode pipeline
# ---------------------------------------------------------------------------

def explode_aparicoes(df_telefones: pd.DataFrame) -> pd.DataFrame:
    """
    Explode ``telefone_aparicoes`` list-of-dicts into individual columns.

    All vectorised — O(n · max_aparicoes).

    Parameters
    ----------
    df_telefones : raw dimension table with ``telefone_aparicoes`` column.

    Returns
    -------
    pd.DataFrame  — one row per (telefone × sistema). Columns:
        telefone_numero, cpf, id_sistema,
        registro_data_atualizacao, telefone_ddd.
    """
    exploded = (
        df_telefones
        .dropna(subset=["telefone_aparicoes"])
        .explode("telefone_aparicoes")
        .reset_index(drop=True)
    )
    sistemas = pd.json_normalize(exploded["telefone_aparicoes"])
    result = pd.concat(
        [exploded.drop(columns=["telefone_aparicoes"]), sistemas], axis=1
    )
    result["registro_data_atualizacao"] = pd.to_datetime(
        result["registro_data_atualizacao"], errors="coerce"
    )
    logger.info(
        "Explode concluído — %d linhas (de %d telefones)", len(result), len(df_telefones)
    )
    return result


def build_merged(
    df_disparos: pd.DataFrame,
    df_tels_sistemas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner-join dispatches with phone-system dimension.

    Key: contato_telefone (disparos) ↔ telefone_numero (dim).

    Parameters
    ----------
    df_disparos      : must contain id_disparo, contato_telefone, status_disparo.
    df_tels_sistemas : output of ``explode_aparicoes``.

    Returns
    -------
    pd.DataFrame  — merged table with ``is_delivered`` boolean column.
    """
    slim_d = (
        df_disparos[["id_disparo", "contato_telefone", "status_disparo"]]
        .dropna(subset=["contato_telefone"])
    )
    slim_t = (
        df_tels_sistemas[
            ["telefone_numero", "cpf", "id_sistema",
             "registro_data_atualizacao", "telefone_ddd"]
        ]
        .dropna(subset=["id_sistema", "telefone_numero"])
    )

    merged = slim_d.merge(
        slim_t,
        left_on="contato_telefone",
        right_on="telefone_numero",
        how="inner",
    )
    merged["is_delivered"] = merged["status_disparo"] == "delivered"
    logger.info("Merge concluído — %d linhas", len(merged))
    return merged


# ---------------------------------------------------------------------------
# 3. Report helpers
# ---------------------------------------------------------------------------

def resumo_eda(
    df_disparos: pd.DataFrame,
    df_telefones: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quick data-quality summary: nulls, dtypes, cardinality.

    Returns
    -------
    pd.DataFrame  — one row per column from both tables combined.
    """
    rows = []
    for tabela, df in [("base_disparo", df_disparos), ("dim_telefone", df_telefones)]:
        for col in df.columns:
            try:
                card = int(df[col].nunique())
            except TypeError:
                card = -1  # unhashable (e.g. list column)
            rows.append({
                "tabela"        : tabela,
                "coluna"        : col,
                "dtype"         : str(df[col].dtype),
                "nulos"         : int(df[col].isna().sum()),
                "pct_nulo"      : round(df[col].isna().mean() * 100, 2),
                "cardinalidade" : card,
            })
    return pd.DataFrame(rows)
