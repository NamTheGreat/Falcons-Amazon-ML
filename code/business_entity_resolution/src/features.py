"""Pairwise features for (query = S2/S3 record, candidate = S1 record)."""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from token_features import TOKEN_FEATS, token_features

STR_COLS = ["name", "name_core", "name_alt", "name_compact", "name_skel", "addr", "addr_nums"]
LOAD_COLS = ["entity_id", "country", "nonlatin_name", "src"] + STR_COLS


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _num_overlap(qn: pl.Series, sn: pl.Series):
    df = pl.DataFrame({"q": qn.str.split(" "), "s": sn.str.split(" ")}).with_columns(
        pl.col("q").list.eval(pl.element().filter(pl.element() != "")),
        pl.col("s").list.eval(pl.element().filter(pl.element() != "")),
    )
    inter = df.select(pl.col("q").list.set_intersection("s").list.len()).to_series()
    nq = df["q"].list.len()
    ns = df["s"].list.len()
    union = nq + ns - inter
    jac = (inter / union.cast(pl.Float32)).fill_nan(0).fill_null(0)
    # first (lowest-sorted is arbitrary) -> use the longest number as house/plot id
    return inter.cast(pl.Float32), jac.cast(pl.Float32), nq.cast(pl.Float32), ns.cast(pl.Float32)


FREQ_COLS_S1 = ["s_namefreq", "s_addrfreq"]
FREQ_COLS_Q = ["q_namefreq_s1", "q_namefreq_q", "q_addrfreq_s1"]


def add_freq_columns(s1: pl.DataFrame, q: pl.DataFrame):
    """Label-free frequency stats (per country) used as match-strength priors.

    A shared name or a shared building weakens the evidence of agreeing on it.
    """
    n1 = s1.group_by("country", "name_core").agg(pl.len().alias("_n"))
    a1 = s1.filter(pl.col("addr") != "").group_by("country", "addr").agg(pl.len().alias("_a"))
    nq = q.group_by("country", "name_core").agg(pl.len().alias("_nq"))
    s1 = (
        s1.join(n1, on=["country", "name_core"], how="left")
        .join(a1, on=["country", "addr"], how="left")
        .with_columns(
            pl.col("_n").fill_null(0).cast(pl.Float32).alias("s_namefreq"),
            pl.col("_a").fill_null(0).cast(pl.Float32).alias("s_addrfreq"),
        ).drop("_n", "_a")
    )
    q = (
        q.join(n1, on=["country", "name_core"], how="left")
        .join(nq, on=["country", "name_core"], how="left")
        .join(a1, on=["country", "addr"], how="left")
        .with_columns(
            pl.col("_n").fill_null(0).cast(pl.Float32).alias("q_namefreq_s1"),
            pl.col("_nq").fill_null(0).cast(pl.Float32).alias("q_namefreq_q"),
            pl.col("_a").fill_null(0).cast(pl.Float32).alias("q_addrfreq_s1"),
        ).drop("_n", "_nq", "_a")
    )
    return s1.sort("rid"), q.sort("rid")


def pair_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """pairs: qid, sid (+ blocking columns). q/s1: normalized frames indexed by rid."""
    qi = pairs["qid"].to_numpy()
    si = pairs["sid"].to_numpy()
    Q = {c: q[c].gather(qi) for c in STR_COLS}
    S = {c: s1[c].gather(si) for c in STR_COLS}
    ql = {c: Q[c].to_list() for c in STR_COLS}
    sl = {c: S[c].to_list() for c in STR_COLS}

    f = {}
    f["n_ratio"] = _cp(ql["name_core"], sl["name_core"], fuzz.ratio)
    f["n_tset"] = _cp(ql["name_core"], sl["name_core"], fuzz.token_set_ratio)
    f["n_tsort"] = _cp(ql["name_core"], sl["name_core"], fuzz.token_sort_ratio)
    f["n_partial"] = _cp(ql["name_core"], sl["name_core"], fuzz.partial_ratio)
    f["n_jw"] = _cp(ql["name_core"], sl["name_core"], JaroWinkler.normalized_similarity)
    f["nfull_ratio"] = _cp(ql["name"], sl["name"], fuzz.ratio)
    f["nfull_tset"] = _cp(ql["name"], sl["name"], fuzz.token_set_ratio)
    f["alt_tset"] = _cp(ql["name_alt"], sl["name_core"], fuzz.token_set_ratio)
    f["comp_ratio"] = _cp(ql["name_compact"], sl["name_compact"], fuzz.ratio)
    f["comp_partial"] = _cp(ql["name_compact"], sl["name_compact"], fuzz.partial_ratio)
    f["skel_ratio"] = _cp(ql["name_skel"], sl["name_skel"], fuzz.ratio)
    f["skel_tset"] = _cp(ql["name_skel"], sl["name_skel"], fuzz.token_set_ratio)
    f["a_ratio"] = _cp(ql["addr"], sl["addr"], fuzz.ratio)
    f["a_tset"] = _cp(ql["addr"], sl["addr"], fuzz.token_set_ratio)
    f["a_tsort"] = _cp(ql["addr"], sl["addr"], fuzz.token_sort_ratio)
    f["a_partial"] = _cp(ql["addr"], sl["addr"], fuzz.partial_token_set_ratio)
    f["a_psort"] = _cp(ql["addr"], sl["addr"], fuzz.partial_token_sort_ratio)
    inter, jac, nq, ns = _num_overlap(Q["addr_nums"], S["addr_nums"])

    qlen = Q["name_core"].str.len_chars().cast(pl.Float32)
    slen = S["name_core"].str.len_chars().cast(pl.Float32)
    qalen = Q["addr"].str.len_chars().cast(pl.Float32)
    salen = S["addr"].str.len_chars().cast(pl.Float32)
    out = pairs.with_columns(
        [pl.Series(k, v) for k, v in f.items()]
        + [
            inter.alias("num_inter"), jac.alias("num_jac"), nq.alias("q_nnums"), ns.alias("s_nnums"),
            qlen.alias("q_nlen"), slen.alias("s_nlen"), qalen.alias("q_alen"), salen.alias("s_alen"),
            (Q["name_alt"].str.len_chars() > 0).cast(pl.Float32).alias("q_has_alt"),
            pl.Series("q_nonlatin", q["nonlatin_name"].gather(qi).cast(pl.Float32)),
            pl.Series("q_src", q["src"].gather(qi).cast(pl.Float32)),
        ]
        + [pl.Series(c, s1[c].gather(si)) for c in FREQ_COLS_S1 if c in s1.columns]
        + [pl.Series(c, q[c].gather(qi)) for c in FREQ_COLS_Q if c in q.columns]
    )
    out = out.with_columns(
        pl.max_horizontal("n_tset", "alt_tset", "comp_partial", "skel_tset").alias("name_best"),
        (pl.min_horizontal("q_nlen", "s_nlen") / pl.max_horizontal("q_nlen", "s_nlen", pl.lit(1.0))).alias("nlen_ratio"),
    )
    return out


def add_token_features(pairs: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    qi = pairs["qid"].to_numpy()
    si = pairs["sid"].to_numpy()
    g = lambda df, c, idx: df[c].gather(idx).to_list()
    tf = token_features(
        g(q, "name_core", qi), g(s1, "name_core", si), g(q, "name_alt", qi), g(q, "name_compact", qi),
        g(q, "name", qi), g(s1, "name", si), g(q, "addr_nums", qi), g(s1, "addr_nums", si), g(q, "addr", qi),
    )
    return pairs.with_columns([pl.Series(k, tf[:, j]) for j, k in enumerate(TOKEN_FEATS)])


def context_features(df: pl.DataFrame, score_col: str = "p1") -> pl.DataFrame:
    """Per-query competition features from a first-stage score."""
    df = df.with_columns(
        pl.col(score_col).rank("ordinal", descending=True).over("qid").cast(pl.Float32).alias("q_rank"),
    )
    return df.with_columns(
        pl.col(score_col).max().over("qid").alias("q_max"),
        pl.len().over("qid").cast(pl.Float32).alias("q_ncand"),
        pl.col(score_col).filter(pl.col("q_rank") == 2).max().over("qid").fill_null(0).alias("q_second"),
        pl.col(score_col).max().over("sid").alias("s_max"),
        (pl.col(score_col) > 0.5).sum().over("sid").cast(pl.Float32).alias("s_nstrong"),
    ).with_columns(
        (pl.col(score_col) - pl.when(pl.col("q_rank") == 1).then(pl.col("q_second")).otherwise(pl.col("q_max")))
        .alias("q_margin"),
    ).drop("q_second")


BLOCK_FEATS = ["bscore_name", "bscore_addr", "bscore_combo", "bscore_apair", "n_keys", "brank"]
PAIR_FEATS = BLOCK_FEATS + [
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw", "nfull_ratio", "nfull_tset", "alt_tset",
    "comp_ratio", "comp_partial", "skel_ratio", "skel_tset", "a_ratio", "a_tset", "a_tsort",
    "a_partial", "a_psort", "num_inter", "num_jac", "q_nnums", "s_nnums", "q_nlen", "s_nlen",
    "q_alen", "s_alen", "q_has_alt", "q_nonlatin", "q_src", "name_best", "nlen_ratio",
] + FREQ_COLS_S1 + FREQ_COLS_Q
STAGE2_FEATS_EXTRA = TOKEN_FEATS
CTX_FEATS = ["p1", "q_rank", "q_max", "q_ncand", "s_max", "s_nstrong", "q_margin"]
STAGE2_FEATS = PAIR_FEATS + TOKEN_FEATS + CTX_FEATS
