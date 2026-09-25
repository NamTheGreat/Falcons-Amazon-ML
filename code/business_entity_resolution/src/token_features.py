"""Token-alignment, legal-form and house-number features.

The negatives that survive blocking are mostly near-copies of a Source 1
record: one name word swapped for another real word ("John Summit Integrated"
vs "John Summit Dental"), a changed legal form (Pvt Ltd -> Ltd, LLC -> Co), or
a perturbed house number (13045 -> 13054). True matches have noise too, but
it's typo-like. These features describe *how* two records differ, not just
how similar they are overall.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np
from rapidfuzz.distance import JaroWinkler, Levenshtein

LEGAL_FORMS = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "plc", "pllc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv", "md", "dds",
}
FILLER = {
    "svc", "center", "centre", "group", "groupe", "holdings", "holding", "partners",
    "and", "the", "of", "de", "du", "des", "la", "le", "les", "dba", "mr", "ms", "m", "s",
    "ets", "etablissements", "overseas", "trading", "ent", "solutions", "associates", "assoc",
}
TOKEN_FEATS = [
    "s_unm", "q_unm", "s_minbest", "q_minbest", "s_short_mis", "s_ntok", "q_ntok",
    "leg_q", "leg_s", "leg_eq", "leg_conflict", "leg_pvt_mis",
    "num_lev_min", "num_absdiff_min", "num_perm", "s_main_in_q", "q_main_in_s", "num_range_ok",
]
_SIM = 0.85


def _best(tok, others):
    b = 0.0
    for o in others:
        if tok == o:
            return 1.0
        s = JaroWinkler.normalized_similarity(tok, o)
        if s > b:
            b = s
    return b


def _one(qn, sn, qalt, qcomp, qfull, sfull, qnums, snums, qaddr):
    qt = [t for t in qn.split() if t]
    if qalt:
        qt += [t for t in qalt.split() if t]
    st = [t for t in sn.split() if t]
    s_unm = 0
    s_min = 1.0
    s_short = 0
    for t in st:
        b = _best(t, qt) if qt else 0.0
        if b < 0.99 and len(t) >= 4 and t in qcomp:
            b = 0.95  # glued / URL-style names
        s_min = min(s_min, b)
        if b < _SIM:
            s_unm += 1
        if len(t) <= 3 and t not in qt:
            s_short += 1
    q_unm = 0
    q_min = 1.0
    for t in qt:
        if t in FILLER:
            continue
        b = _best(t, st) if st else 0.0
        q_min = min(q_min, b)
        if b < _SIM:
            q_unm += 1
    lq = {t for t in qfull.split() if t in LEGAL_FORMS}
    ls = {t for t in sfull.split() if t in LEGAL_FORMS}
    leg_eq = float(lq == ls)
    leg_conf = float(bool(lq) and bool(ls) and not (lq & ls))
    leg_pvt = float(("pvt" in lq) != ("pvt" in ls))

    qn_ = [x for x in qnums.split() if x]
    sn_ = [x for x in snums.split() if x]
    if qn_ and sn_:
        lev = min(Levenshtein.distance(a, b) for a in qn_ for b in sn_)
        absd = min(abs(int(a[:12]) - int(b[:12])) for a in qn_ for b in sn_)
        smain = max(sn_, key=len)
        qmain = max(qn_, key=len)
        perm = float(any(sorted(a) == sorted(b) and a != b for a in qn_ for b in sn_ if len(a) >= 3))
        s_in = float(smain in qn_)
        q_in = float(qmain in sn_)
        # "1701-1705" style ranges in the query that contain the S1 number
        rng = 0.0
        try:
            for a in sn_:
                ai = int(a[:12])
                for i in range(len(qn_) - 1):
                    lo, hi = int(qn_[i][:12]), int(qn_[i + 1][:12])
                    if lo < hi and lo <= ai <= hi and (hi - lo) < 50:
                        rng = 1.0
        except ValueError:
            pass
    else:
        lev, absd, perm, s_in, q_in, rng = -1, -1, 0.0, -1.0, -1.0, 0.0
    return (
        s_unm, q_unm, s_min, q_min, s_short, len(st), len(qt),
        len(lq), len(ls), leg_eq, leg_conf, leg_pvt,
        lev, min(absd, 1e6), perm, s_in, q_in, rng,
    )


_G = {}


def _work(bounds):
    a, b = bounds
    g = _G
    return np.array(
        [_one(*(g[k][i] for k in _KEYS)) for i in range(a, b)], dtype=np.float32
    ).reshape(-1, len(TOKEN_FEATS))


_KEYS = ["qn", "sn", "qalt", "qcomp", "qfull", "sfull", "qnums", "snums", "qaddr"]


def token_features(qn, sn, qalt, qcomp, qfull, sfull, qnums, snums, qaddr, workers=10):
    """All args are equal-length lists of str. Returns float32 [n, len(TOKEN_FEATS)]."""
    global _G
    _G = dict(zip(_KEYS, [qn, sn, qalt, qcomp, qfull, sfull, qnums, snums, qaddr]))
    n = len(qn)
    step = max(1, n // (workers * 4) + 1)
    bounds = [(i, min(n, i + step)) for i in range(0, n, step)]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        parts = pool.map(_work, bounds)
    _G = {}
    return np.concatenate(parts) if parts else np.zeros((0, len(TOKEN_FEATS)), np.float32)
