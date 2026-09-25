"""Text normalization for business names and addresses.

All heavy lifting is done with vectorized polars string ops. Non-ASCII text
(accents, Devanagari, Tamil, Telugu, Kannada, ...) is romanized with
``anyascii`` (ISC license) so records in different scripts can be compared.
Nothing here depends on a fixed set of countries.
"""
from __future__ import annotations

import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------------------
# Canonical token maps (target -> variants; ADDR_MAP is a list of pairs
# because some targets repeat, e.g. "fl" = floor / Florida). Applied with word-boundary
# regexes after lowercasing / romanization / punctuation stripping.
# ---------------------------------------------------------------------------
NAME_MAP = {
    "inc": ["incorporated", "incorporation", "incorp"],
    "corp": ["corporation"],
    "co": ["company", "compagnie", "cie", "kampani", "kampni", "kmpni"],
    "ltd": ["limited", "limitd", "limted", "ltd", "lmtd", "limitemd", "limitedd", "limitet", "limirrd", "limired", "li"],
    "pvt": ["private", "pvt", "priv", "praivet", "prayivet", "praiveta", "privet", "prvt", "praibhet", "prayvet", "praivrr", "piraivet", "pra", "priveta", "praivett"],
    "llc": ["l l c", "elelsi"],
    "llp": ["l l p", "elelpi", "ailailpi", "elalpi"],
    "and": ["&", "et", "und"],
    "intl": ["international"],
    "mfg": ["manufacturing"],
    "svc": ["services", "service", "srvises", "sarvises", "servises"],
    "ent": ["enterprises", "enterprise", "entreprise", "entreprises"],
    "assoc": ["associates", "association", "associes"],
    "bros": ["brothers"],
    "tech": ["technologies", "technology", "technolojis", "teknolojij", "teknolji", "teknolojis", "tek", "teknoloji"],
    "infra": ["imphra", "inphra", "infrastructure", "inphrastrkcr"],
    "it": ["aiti"],
    "exports": ["eksports", "ekspors"],
}
# Tokens that carry little identity (legal forms, filler words).
LEGAL = {
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "pc", "plc", "pllc",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv",
    "the", "and", "of", "de", "du", "des", "la", "le", "les", "d", "l", "dba",
    "group", "groupe", "holdings", "holding", "center", "centre", "svc",
    "ets", "etablissements", "md", "a",
}

ADDR_MAP = [
    ("st", ["street", "str", "saint", "sreet", "strt"]),
    ("ave", ["avenue", "av", "aven", "avn"]),
    ("rd", ["road", "raod"]),
    ("dr", ["drive", "drv"]),
    ("blvd", ["boulevard", "bd", "boul"]),
    ("ct", ["court", "crt"]),
    ("ln", ["lane"]),
    ("pl", ["place"]),
    ("cir", ["circle"]),
    ("ter", ["terrace"]),
    ("hwy", ["highway"]),
    ("pkwy", ["parkway"]),
    ("sq", ["square"]),
    ("ste", ["suite"]),
    ("apt", ["apartment", "appt"]),
    ("fl", ["floor", "flr"]),
    ("bldg", ["building"]),
    ("n", ["north"]),
    ("s", ["south"]),
    ("e", ["east"]),
    ("w", ["west"]),
    ("r", ["rue"]),
    ("near", ["nr", "nearby"]),
    ("opp", ["opposite"]),
    ("po", ["post office", "boite postale", "bp"]),
    ("no", ["number", "num", "nos"]),
    ("mt", ["mount"]),
    ("ft", ["feet", "fort"]),
    ("twp", ["township", "townshiip"]),
    ("1", ["first"]), ("2", ["second"]), ("3", ["third"]), ("4", ["fourth"]),
    ("5", ["fifth"]), ("6", ["sixth"]), ("7", ["seventh"]), ("8", ["eighth"]),
    ("9", ["ninth"]), ("10", ["tenth"]),
    # US states
    ("al", ["alabama"]), ("ak", ["alaska"]), ("az", ["arizona"]), ("ar", ["arkansas"]),
    ("ca", ["california"]), ("co", ["colorado"]), ("ct", ["connecticut"]),
    ("de", ["delaware"]), ("fl", ["florida"]), ("ga", ["georgia"]), ("hi", ["hawaii"]),
    ("id", ["idaho"]), ("il", ["illinois"]), ("in", ["indiana"]), ("ia", ["iowa"]),
    ("ks", ["kansas"]), ("ky", ["kentucky"]), ("la", ["louisiana"]), ("me", ["maine"]),
    ("md", ["maryland"]), ("ma", ["massachusetts"]), ("mi", ["michigan"]),
    ("mn", ["minnesota"]), ("ms", ["mississippi"]), ("mo", ["missouri"]),
    ("mt", ["montana"]), ("ne", ["nebraska"]), ("nv", ["nevada"]),
    ("nh", ["new hampshire"]), ("nj", ["new jersey"]), ("nm", ["new mexico"]),
    ("ny", ["new york"]), ("nc", ["north carolina"]), ("nd", ["north dakota"]),
    ("oh", ["ohio"]), ("ok", ["oklahoma"]), ("or", ["oregon"]), ("pa", ["pennsylvania"]),
    ("ri", ["rhode island"]), ("sc", ["south carolina"]), ("sd", ["south dakota"]),
    ("tn", ["tennessee", "tamil nadu", "tamilnadu", "tmilnatu", "tmil natu"]),
    ("tx", ["texas"]), ("ut", ["utah"]), ("vt", ["vermont"]), ("va", ["virginia"]),
    ("wa", ["washington"]), ("wv", ["west virginia"]), ("wi", ["wisconsin"]),
    ("wy", ["wyoming"]), ("dc", ["district of columbia"]),
    # Indian states (English + romanized native-script spellings)
    ("mh", ["maharashtra", "mharastr", "maharastra", "maharashtr"]),
    ("ka", ["karnataka", "krnatk", "karnatak"]),
    ("kl", ["kerala", "keral", "kerlm", "keralam"]),
    ("tg", ["telangana", "telamgana", "tlmgan", "telmgana", "ts"]),
    ("ap", ["andhra pradesh", "amdhrprdes", "amdhra pradesh", "andhrapradesh"]),
    ("dl", ["delhi", "dilli", "dilhi", "new delhi"]),
    ("up", ["uttar pradesh", "uttr prdes", "uttarpradesh"]),
    ("mp", ["madhya pradesh", "mdhy prdes", "madhyapradesh"]),
    ("wb", ["west bengal", "pshcim bmgal", "pascim bamgal", "pashchim bangal"]),
    ("gj", ["gujarat", "gujrat", "gujrat", "gujart"]),
    ("rj", ["rajasthan", "rajsthan", "rajasthn"]),
    ("hr", ["haryana", "hriyana", "hariyana"]),
    ("pb", ["punjab", "pmjab", "pamjab"]),
    ("br", ["bihar", "bihaar"]),
    ("or", ["odisha", "orissa", "odisa"]),
    ("as", ["assam", "asm"]),
    ("jh", ["jharkhand", "jharkhmd"]),
    ("cg", ["chhattisgarh", "chattisgarh", "chttisgdh"]),
    ("ga", ["goa"]),
    ("uk", ["uttarakhand", "uttrakhmd"]),
    ("hp", ["himachal pradesh", "himacl prdes"]),
    ("jk", ["jammu and kashmir", "jammu kashmir"]),
    ("ch", ["chandigarh", "cmdigdh"]),
    ("py", ["puducherry", "pondicherry", "puduccheri"]),
]
# Address tokens with essentially no discriminating value.
ADDR_STOP = {
    "no", "null", "none", "na", "n/a", "the", "of", "and", "de", "du", "des",
    "la", "le", "les", "d", "l", "near", "opp", "behind", "at", "po", "dist",
    "district", "flat", "plot", "house", "h", "unit", "door", "shop", "office",
    "floor", "fl", "ste", "apt", "bldg", "pmb", "box", "old", "new",
    "cedex",  # French mail-routing suffix (e.g. "75015 Paris Cedex 15"), not address content
}

_ORDINAL = r"\b(\d+)(?:st|nd|rd|th|er|eme)\b"


def _roman(s: pl.Series) -> pl.Series:
    """Romanize the non-ASCII entries of a string Series with anyascii."""
    s = s.fill_null("")
    mask = ~s.str.contains(r"^[\x00-\x7F]*$")
    if mask.sum() == 0:
        return s
    idx = mask.arg_true()
    vals = [anyascii(v) for v in s.gather(idx).to_list()]
    return s.scatter(idx, pl.Series(vals, dtype=pl.String))


def _apply_map(e: pl.Expr, mapping) -> pl.Expr:
    items = mapping.items() if isinstance(mapping, dict) else mapping
    for tgt, variants in items:
        alts = sorted({v for v in variants if v != tgt}, key=len, reverse=True)
        if not alts:
            continue
        pat = r"\b(?:" + "|".join(a.replace(" ", r"\s+") for a in alts) + r")\b"
        e = e.str.replace_all(pat, tgt)
    return e


def _basic(e: pl.Expr) -> pl.Expr:
    """Lowercase, drop dots/apostrophes inside tokens, other punctuation -> space."""
    return (
        e.str.to_lowercase()
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"['’`\.]", "")
        .str.replace_all(r"[^a-z0-9 ]+", " ")
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
    )


def _strip_tokens(e: pl.Expr, drop: set[str]) -> pl.Expr:
    pat = r"\b(?:" + "|".join(sorted(drop, key=len, reverse=True)) + r")\b"
    return e.str.replace_all(pat, " ").str.replace_all(r"\s+", " ").str.strip_chars()


def normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Add normalized columns to a raw source frame.

    Output columns: entity_id, country, src (1/2/3), name, name_core,
    name_alt (DBA part, may be empty), name_compact, name_skel, addr, addr_nums,
    has_addr, nonlatin_name, raw_name_len.
    """
    df = df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("").str.strip_chars(),
    )
    raw_name = df["business_name"]
    nonlatin = raw_name.str.contains(r"[ऀ-෿]")  # Indic script blocks
    df = df.with_columns(
        _roman(df["business_name"]).alias("_n"),
        _roman(df["business_address"]).alias("_a"),
        nonlatin.alias("nonlatin_name"),
        raw_name.str.len_chars().alias("raw_name_len"),
        pl.col("entity_id").str.slice(1, 1).cast(pl.Int8).alias("src"),
    )

    name = _basic(pl.col("_n").str.replace_all(r"(?i)\.(com|net|org|in|fr|co)\b", " "))
    name = name.str.replace_all(r"\bwww\b", " ")
    name = _apply_map(name, NAME_MAP)
    name = name.str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(name.alias("name"))
    # remove the record's own country word, e.g. "(India)" / "(France)"
    for c in df["country"].unique().to_list():
        c_norm = _basic(pl.lit(anyascii(c or "")))
        c_str = pl.select(c_norm).item()
        if not c_str:
            continue
        df = df.with_columns(
            pl.when(pl.col("country") == c)
            .then(pl.col("name").str.replace_all(r"\b" + c_str + r"\b", " "))
            .otherwise(pl.col("name"))
            .str.replace_all(r"\s+", " ").str.strip_chars().alias("name")
        )
    # DBA / trade-name split: "x dba y" -> alt = y ; "d b a" arises from d/b/a
    dba = r"\b(?:dba|d b a|doing business as)\b"
    df = df.with_columns(
        pl.when(pl.col("name").str.contains(dba))
        .then(pl.col("name").str.split_exact(" dba ", 1).struct.field("field_1"))
        .otherwise(pl.lit(""))
        .alias("_alt")
    )
    df = df.with_columns(
        pl.col("name").str.replace_all(r"\bd b a\b", "dba").alias("name"),
    ).with_columns(
        pl.when(pl.col("name").str.contains(r"\bdba\b"))
        .then(pl.col("name").str.split(" dba ").list.last())
        .otherwise(pl.lit(""))
        .fill_null("")
        .alias("name_alt")
    ).drop("_alt")
    df = df.with_columns(_strip_tokens(pl.col("name"), LEGAL).alias("name_core"))
    df = df.with_columns(
        pl.when(pl.col("name_core") == "").then(pl.col("name")).otherwise(pl.col("name_core")).alias("name_core"),
        _strip_tokens(pl.col("name_alt"), LEGAL).alias("name_alt"),
    )
    df = df.with_columns(pl.col("name_core").str.replace_all(" ", "").alias("name_compact"))
    df = df.with_columns(
        pl.col("name_core").str.split(" ")
        .list.eval(skeleton(pl.element()))
        .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
        .list.join(" ").alias("name_skel")
    )

    addr = pl.col("_a").str.to_lowercase().str.replace_all(r"\bnull\b|\bn/a\b", " ")
    addr = _basic(addr)
    addr = addr.str.replace_all(_ORDINAL, "$1")
    # "Cedex 15" is a French postal-routing suffix, not a street number -- drop
    # it (and any trailing digits) before the number-overlap features see it.
    addr = addr.str.replace_all(r"\bcedex\b\s*\d*", " ")
    addr = _apply_map(addr, ADDR_MAP)
    addr = addr.str.replace_all(r"\s+", " ").str.strip_chars()
    df = df.with_columns(addr.alias("addr"))
    df = df.with_columns(
        pl.col("addr").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
        .list.eval(pl.element().filter(pl.element() != "")).list.unique().list.sort()
        .list.join(" ").alias("addr_nums"),
        (pl.col("addr").str.len_chars() > 0).alias("has_addr"),
    )
    return df.select(
        "entity_id", "country", "src", "name", "name_core", "name_alt", "name_compact", "name_skel",
        "addr", "addr_nums", "has_addr", "nonlatin_name", "raw_name_len",
        pl.col("business_name").alias("raw_name"), pl.col("business_address").alias("raw_addr"),
    )


def skeleton(e: pl.Expr) -> pl.Expr:
    """Crude phonetic key per token, robust to romanized Indic spellings.

    e.g. software / sophtveyr -> sftvr, healthcare / helthkeyr -> ltkr.
    """
    e = (
        e.str.replace_all("ph", "f").str.replace_all(r"c([eiy])", "s$1")
        .str.replace_all("sh", "s").str.replace_all("x", "ks")
    )
    for a, b in (("q", "k"), ("c", "k"), ("z", "k"), ("j", "k"), ("g", "k"), ("w", "v"),
                 ("d", "t"), ("b", "p"), ("m", "n")):
        e = e.str.replace_all(a, b, literal=True)
    e = e.str.replace_all(r"[aeiouyh]", "")
    for ch in "fklnprstv":  # collapse doubled consonants
        e = e.str.replace_all(ch + "{2,}", ch)
    return e


def load_source(path: str) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", infer_schema=False, quote_char=None,
        missing_utf8_is_empty_string=True,
    )
