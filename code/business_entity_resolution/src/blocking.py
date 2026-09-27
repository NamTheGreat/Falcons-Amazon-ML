"""Candidate generation (blocking).

Every Source 2/3 record matches at most one Source 1 entity, so blocking is run
from the S2/S3 side: each S2/S3 record ("query") is matched against the Source 1
index with an inverted index over hashed keys:

* name tokens, phonetic skeletons of name tokens, name token bigrams, prefixes/suffixes of the space-free name
  (catches URL-style names like ``kempgloba.com``)
* address tokens, and (number, neighbouring token) pairs
* (name token or phonetic token) x (address word or number) combos, which stay
  selective for generic and transliterated names
* unordered address-word pairs (for DBA / other-script names at busy streets)

Keys are scoped by country (an open set -- whatever labels appear), weighted by
IDF over Source 1, and each query only probes its rarest keys, which keeps the
join small. Every S1 candidate is scored by the summed IDF of shared name and
address keys, and the top candidates per query are kept.
"""
from __future__ import annotations

import math

import polars as pl

from normalize import ADDR_STOP, ADDR_MAP

_STATE_CODES = {t for t, _ in ADDR_MAP if len(t) == 2}
ADDR_SKIP = ADDR_STOP | _STATE_CODES | {
    "st", "rd", "ave", "dr", "ln", "ct", "blvd", "pl", "cir", "ter", "hwy", "pkwy",
    "city", "road", "main", "cross", "sector", "phase", "colony", "nagar", "r",
    "n", "s", "e", "w", "twp", "block", "street", "marg", "lane", "layout", "area",
}

KIND_NAME, KIND_ADDR, KIND_COMBO, KIND_APAIR = 0, 1, 2, 3


def _explode(df: pl.DataFrame, expr: pl.Expr, kind: int, tag: str) -> pl.DataFrame:
    """Build (rid, key, kind) rows from a list[str] expression."""
    return (
        df.select("rid", "country", expr.alias("tok"))
        .explode("tok")
        .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
        .select(
            "rid",
            (pl.lit(tag) + pl.col("country") + pl.lit("|") + pl.col("tok")).hash(seed=7).alias("key"),
            pl.lit(kind, dtype=pl.UInt8).alias("kind"),
        )
    )


def _cross(df: pl.DataFrame, left: pl.Expr, right: pl.Expr, tag: str) -> pl.DataFrame:
    return (
        df.select("rid", "country", left.alias("n"), right.alias("a"))
        .explode("n").explode("a")
        .filter(pl.col("n").is_not_null() & pl.col("a").is_not_null())
        .select(
            "rid",
            (pl.lit(tag) + pl.col("country") + pl.lit("|") + pl.col("n") + pl.lit("|") + pl.col("a"))
            .hash(seed=7).alias("key"),
            pl.lit(KIND_COMBO, dtype=pl.UInt8).alias("kind"),
        )
    )


def _build_keys_slice(df: pl.DataFrame, use_alt: bool) -> pl.DataFrame:
    """df needs rid (UInt32), country, name_core, name_alt, name_compact, name_skel, addr, addr_nums."""
    toks = pl.col("name_core").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
    if use_alt:
        alt = pl.col("name_alt").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
        toks = pl.concat_list(toks, alt).list.unique()
    bigr = pl.col("name_core").str.split(" ").list.eval(
        pl.element() + pl.lit("_") + pl.element().shift(-1)
    ).list.eval(pl.element().filter(pl.element().is_not_null()))
    comp = pl.col("name_compact")
    comp_keys = pl.concat_list(
        pl.when(comp.str.len_chars() >= 6).then(comp.str.slice(0, 6)).otherwise(None),
        pl.when(comp.str.len_chars() >= 9).then(comp.str.slice(-6, 6)).otherwise(None),
        pl.when(comp.str.len_chars() >= 4).then(pl.lit("full:") + comp).otherwise(None),
    )
    skel = pl.concat_list(
        pl.col("name_skel").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)),
        pl.when(pl.col("name_skel").str.len_chars() >= 4).then(pl.lit("full:") + pl.col("name_skel")).otherwise(None),
    )
    atoks = pl.col("addr").str.split(" ")
    aword = atoks.list.eval(
        pl.element().filter(
            (pl.element().str.len_chars() >= 3)
            & ~pl.element().is_in(list(ADDR_SKIP))
            & ~pl.element().str.contains(r"^\d+$")
        )
    ).list.unique()
    # (number, next token) and (prev token, number) pairs
    anum = atoks.list.eval(
        pl.when(pl.element().str.contains(r"^\d+$"))
        .then(pl.element().str.strip_chars_start("0") + pl.lit("_") + pl.element().shift(-1))
        .otherwise(None)
    ).list.eval(pl.element().filter(pl.element().is_not_null()))
    anum2 = atoks.list.eval(
        pl.when(pl.element().str.contains(r"^\d+$"))
        .then(pl.element().shift(1) + pl.lit("_") + pl.element().str.strip_chars_start("0"))
        .otherwise(None)
    ).list.eval(pl.element().filter(pl.element().is_not_null()))
    # name x address combos: selective even when both parts are common.
    # Plain and phonetic name tokens are crossed with address words and numbers.
    ntok = pl.col("name_core").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 3)
    ).list.head(3)
    stok = pl.col("name_skel").str.split(" ").list.eval(
        pl.element().filter(pl.element().str.len_chars() >= 2)
    ).list.head(3)
    nums = pl.col("addr_nums").str.split(" ").list.eval(
        pl.element().filter(pl.element() != "")
    ).list.head(4)
    combos = [
        _cross(df, ntok, aword.list.head(6), "x:"),
        _cross(df, ntok, nums, "y:"),
        _cross(df, stok, aword.list.head(6), "z:"),
        _cross(df, stok, nums, "w:"),
    ]
    # address word pairs (unordered): selective when each word is common
    aw = aword.list.head(7)
    apair = (
        df.select("rid", "country", aw.alias("a"), aw.alias("b"))
        .explode("a").explode("b")
        .filter(pl.col("a").is_not_null() & pl.col("b").is_not_null() & (pl.col("a") < pl.col("b")))
        .select(
            "rid",
            (pl.lit("p:") + pl.col("country") + pl.lit("|") + pl.col("a") + pl.lit("|") + pl.col("b"))
            .hash(seed=7).alias("key"),
            pl.lit(KIND_APAIR, dtype=pl.UInt8).alias("kind"),
        )
    )
    parts = combos + [
        apair,
        _explode(df, toks, KIND_NAME, "n:"),
        _explode(df, bigr, KIND_NAME, "b:"),
        _explode(df, comp_keys, KIND_NAME, "c:"),
        _explode(df, skel, KIND_NAME, "k:"),
        _explode(df, aword, KIND_ADDR, "a:"),
        _explode(df, anum, KIND_ADDR, "d:"),
        _explode(df, anum2, KIND_ADDR, "d:"),
    ]
    return pl.concat(parts).unique()


def build_keys(df: pl.DataFrame, use_alt: bool, slice_size: int = 400_000) -> pl.DataFrame:
    """Build keys in slices to cap peak RAM to ~1.5 GB instead of 11.5 GB."""
    if df.height <= slice_size:
        return _build_keys_slice(df, use_alt)
    slices = [
        _build_keys_slice(df.slice(i, slice_size), use_alt)
        for i in range(0, df.height, slice_size)
    ]
    return pl.concat(slices)


def generate_candidates(
    s1: pl.DataFrame,
    q: pl.DataFrame,
    k_name: int = 5,
    k_addr: int = 4,
    k_combo: int = 6,
    k_apair: int = 4,
    df_cap: int = 400,
    top_m: int = 10,
    top_m_wide: int = 24,
    chunk: int = 600_000,
    verbose: bool = True,
) -> pl.DataFrame:
    """Return candidate pairs (qid, sid, bscore_name, bscore_addr, bscore_combo, bscore_apair, n_keys, brank).

    ``s1`` / ``q`` must carry a UInt32 ``rid`` column (row ids used as sid/qid).
    Queries with no address or a non-Latin name get ``top_m_wide`` candidates,
    because their blocking scores are least reliable.
    """
    n1 = s1.height
    k1 = build_keys(s1, use_alt=False)
    dfreq = k1.group_by("key").agg(pl.len().alias("df"))
    dfreq = dfreq.filter(pl.col("df") <= df_cap).with_columns(
        (pl.lit(math.log(n1 + 1)) - (pl.col("df").cast(pl.Float64) + 1).log()).cast(pl.Float32).alias("idf")
    )
    k1 = k1.join(dfreq.select("key"), on="key", how="semi").rename({"rid": "sid"}).drop("kind")
    if verbose:
        print(f"  s1 keys: {k1.height:,}  distinct usable: {dfreq.height:,}", flush=True)

    out = []
    for start in range(0, q.height, chunk):
        qc = q.slice(start, chunk)
        kq = build_keys(qc, use_alt=True).join(dfreq, on="key", how="inner")
        # probe only the rarest keys of each kind per query
        kq = (
            kq.sort("df")
            .with_columns(pl.int_range(pl.len()).over("rid", "kind").alias("_r"))
            .filter(
                ((pl.col("kind") == KIND_NAME) & (pl.col("_r") < k_name))
                | ((pl.col("kind") == KIND_ADDR) & (pl.col("_r") < k_addr))
                | ((pl.col("kind") == KIND_COMBO) & (pl.col("_r") < k_combo))
                | ((pl.col("kind") == KIND_APAIR) & (pl.col("_r") < k_apair))
            )
            .drop("_r", "df")
        )
        pairs = kq.join(k1, on="key", how="inner")
        agg = pairs.group_by("rid", "sid").agg(
            pl.col("idf").filter(pl.col("kind") == KIND_NAME).sum().alias("bscore_name"),
            pl.col("idf").filter(pl.col("kind") == KIND_ADDR).sum().alias("bscore_addr"),
            pl.col("idf").filter(pl.col("kind") == KIND_COMBO).sum().alias("bscore_combo"),
            pl.col("idf").filter(pl.col("kind") == KIND_APAIR).sum().alias("bscore_apair"),
            pl.len().cast(pl.UInt16).alias("n_keys"),
        ).with_columns((pl.col("bscore_name") + pl.col("bscore_addr") + pl.col("bscore_combo") + pl.col("bscore_apair")).alias("_tot"))
        wide = qc.select(
            "rid", ((pl.col("addr") == "") | pl.col("nonlatin_name")).alias("_wide")
        )
        agg = (
            agg.sort(["_tot", "n_keys"], descending=True)
            .with_columns(pl.int_range(1, pl.len() + 1, dtype=pl.UInt16).over("rid").alias("brank"))
            .join(wide, on="rid", how="left")
            .filter(pl.col("brank") <= pl.when(pl.col("_wide")).then(top_m_wide).otherwise(top_m))
            .drop("_tot", "_wide")
            .rename({"rid": "qid"})
        )
        out.append(agg)
        if verbose:
            print(f"  queries {start:,}-{start + qc.height:,}: probe rows {pairs.height:,} -> {agg.height:,} cands", flush=True)
    return pl.concat(out)
