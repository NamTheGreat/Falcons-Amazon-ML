"""Decision rule and the challenge metric (macro F0.5 over Source 1 entities)."""
import numpy as np
import polars as pl


def decide(df: pl.DataFrame, thr, score: str = "p2") -> pl.DataFrame:
    """Each S2/S3 record goes to its best-scoring S1 entity if the score >= thr.

    ``thr`` is either a single cutoff applied to every row (float/np.floating,
    as tune_threshold uses it), or the name of a column already present in
    ``df`` holding a per-row cutoff -- e.g. a per-country threshold the caller
    joined in beforehand. This keeps any country-specific decision logic out
    of the core rule itself, so it never hardcodes a closed set of countries.
    """
    cutoff = pl.col(thr) if isinstance(thr, str) else pl.lit(thr)
    return (
        df.filter(pl.col(score) >= cutoff)
        .sort(score, descending=True)
        .unique("qid", keep="first")
        .select("qid", "sid")
    )


def macro_f05(pred: pl.DataFrame, gt: pl.DataFrame, sids: pl.Series, beta: float = 0.5) -> dict:
    """pred / gt: (qid, sid) pairs; sids: the S1 entities to average over."""
    base = pl.DataFrame({"sid": sids})
    pred = pred.join(base, on="sid", how="semi")
    tp = pred.join(gt, on=["qid", "sid"], how="semi").group_by("sid").agg(pl.len().alias("tp"))
    npred = pred.group_by("sid").agg(pl.len().alias("np"))
    ntrue = gt.join(base, on="sid", how="semi").group_by("sid").agg(pl.len().alias("nt"))
    d = (
        base.join(tp, on="sid", how="left").join(npred, on="sid", how="left").join(ntrue, on="sid", how="left")
        .fill_null(0)
    )
    b2 = beta * beta
    d = d.with_columns(
        pl.when((pl.col("np") == 0) & (pl.col("nt") == 0)).then(1.0)
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise(
            (1 + b2) * pl.col("tp") / ((1 + b2) * pl.col("tp") + b2 * (pl.col("nt") - pl.col("tp")) + (pl.col("np") - pl.col("tp")))
        ).alias("f")
    )
    tot_tp, tot_np, tot_nt = d["tp"].sum(), d["np"].sum(), d["nt"].sum()
    return {
        "f05": round(float(d["f"].mean()), 5),
        "micro_p": round(tot_tp / max(tot_np, 1), 5),
        "micro_r": round(tot_tp / max(tot_nt, 1), 5),
    }


def _poibin(P):
    """Poisson-binomial pmf for each row of P [m, n] -> [m, n+1]."""
    m, n = P.shape
    d = np.zeros((m, n + 1))
    d[:, 0] = 1.0
    for j in range(n):
        p = P[:, j:j + 1]
        d[:, 1:j + 2] = d[:, 1:j + 2] * (1 - p) + d[:, 0:j + 1] * p
        d[:, 0] *= (1 - P[:, j])
    return d


def decide_expected_f(df: pl.DataFrame, score: str = "p2", min_p: float = 0.05,
                      miss_rate: float = 0.0, beta: float = 0.5, max_n: int = 16) -> pl.DataFrame:
    """Per S1 entity, pick the top-k candidates that maximize expected F-beta.

    Each query is first assigned to its best S1. For every S1 the candidate
    scores are treated as independent match probabilities and E[F] is computed
    exactly (Poisson-binomial over selected and unselected candidates) for
    every k. ``miss_rate`` adds an expected number of true matches that were
    never scored (blocking misses), which makes the rule slightly bolder.
    """
    b2 = beta * beta
    best = (
        df.filter(pl.col(score) >= min_p)
        .sort(score, descending=True)
        .unique("qid", keep="first")
        .select("qid", "sid", score)
        .sort(["sid", score], descending=[False, True])
        .with_columns(pl.int_range(pl.len()).over("sid").alias("_r"))
        .filter(pl.col("_r") < max_n)
    )
    g = best.group_by("sid", maintain_order=True).agg(pl.col(score).alias("ps"), pl.col("qid").alias("qs"))
    g = g.with_columns(pl.col("ps").list.len().alias("n"))
    chosen = []
    for n, part in g.group_by("n"):
        n = int(n[0])
        P = np.array(part["ps"].to_list(), dtype=np.float64).reshape(-1, n)
        m = P.shape[0]
        # optional pseudo-candidate for unseen matches (never selectable)
        extra = np.full((m, 1), miss_rate) if miss_rate > 0 else np.zeros((m, 0))
        best_ef = np.full(m, -1.0)
        best_k = np.zeros(m, dtype=np.int64)
        for k in range(0, n + 1):
            sel = P[:, :k]
            rest = np.concatenate([P[:, k:], extra], axis=1)
            dt = _poibin(sel)          # tp distribution
            df_ = _poibin(rest)        # fn distribution
            tp = np.arange(k + 1)[None, :, None]
            fn = np.arange(rest.shape[1] + 1)[None, None, :]
            num = (1 + b2) * tp
            den = (1 + b2) * tp + b2 * fn + (k - tp)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(den == 0, 1.0, num / np.where(den == 0, 1, den))
            ef = np.einsum("mi,mj,xij->m", dt, df_, f)
            upd = ef > best_ef + 1e-12
            best_ef[upd] = ef[upd]
            best_k[upd] = k
        qs = part["qs"].to_list()
        sids = part["sid"].to_list()
        for i in np.nonzero(best_k)[0]:
            for qq in qs[i][: best_k[i]]:
                chosen.append((qq, sids[i]))
    if not chosen:
        return pl.DataFrame({"qid": [], "sid": []}, schema={"qid": pl.UInt32, "sid": pl.UInt32})
    out = np.array(chosen, dtype=np.int64)
    return pl.DataFrame({"qid": out[:, 0], "sid": out[:, 1]}).with_columns(pl.all().cast(pl.UInt32))
