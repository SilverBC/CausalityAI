"""Rules-based parsing of a plain-English causal claim (no language model).

`parse_claim(text, prep)` pulls out the treatment and outcome phrases, the claimed
direction, size, unit, bound, shape, population and hedges, matches the phrases to
columns with a score in [0, 1], flags claims the data can never test (mediation,
counterfactual wording, comparison with another treatment) and builds the estimand.

It never raises: problems go into `claim.notes`.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Optional

import pandas as pd

from .data import words
from .schema import Claim, Estimand, Prepared

# --------------------------------------------------------------------------- vocab
STOP = {
    "the", "a", "an", "of", "to", "on", "in", "for", "at", "by", "with", "and", "or", "their", "his",
    "her", "its", "our", "your", "my", "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "it", "they", "them", "who", "which", "s", "average", "overall", "level",
    "levels", "rate", "rates", "amount", "amounts", "people", "persons", "individuals", "units",
    "participants", "subjects", "group", "having", "getting", "receiving", "some", "any", "much",
    "many", "very", "more", "less", "also", "then", "there", "than", "do", "does", "did", "have",
    "has", "had", "get", "gets", "got",
}

SYNONYMS = [
    {"wage", "wages", "earnings", "earning", "earn", "income", "salary", "salaries", "pay", "paycheck"},
    {"test", "score", "scores", "exam", "exams", "mark", "marks", "assessment"},
    {"grade", "grades", "gpa"},
    {"treated", "treatment", "program", "programme", "intervention", "participation", "participant",
     "therapy"},
    {"health", "wellbeing", "wellness"},
    {"female", "woman", "women", "girl", "girls"},
    {"male", "man", "men", "boy", "boys"},
    {"age", "aged", "old", "older", "years"},
    {"price", "prices", "cost", "costs"},
    {"sales", "revenue", "revenues", "turnover"},
    {"consumption", "spending", "expenditure", "expenditures"},
    {"pressure", "bp"},
    {"study", "studying", "homework"},
    {"hours", "hour", "time"},
    {"job", "jobs", "employment", "employed", "work", "working"},
    {"drug", "medication", "medicine", "pill", "dose"},
    {"coffee", "caffeine"},
    {"policy", "law", "regulation", "reform", "mandate"},
    {"scholarship", "grant", "award", "aid", "bursary"},
    {"lottery", "draw"},
    {"enrolled", "enrollment", "enrolment", "enrol", "enroll", "attend", "attendance", "attending"},
    {"member", "members", "membership"},
    {"region", "area", "zone"},
]

FEMALE_WORDS = {"women", "woman", "female", "females", "girls", "girl", "mothers", "mother", "wives", "ladies"}
MALE_WORDS = {"men", "man", "male", "males", "boys", "boy", "fathers", "father", "husbands", "gentlemen"}

HEDGES = [
    "may", "might", "could", "likely", "probably", "possibly", "potentially", "suggests", "suggest",
    "associated", "tends to", "tend to", "on average", "appears to", "appear to", "seems to", "seem to",
    "perhaps", "plausibly", "linked",
]

NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10", "twenty": "20", "fifty": "50", "hundred": "100"}

# subject words dropped from the end of the treatment phrase
TRAIL_AUX = {
    "does", "do", "did", "not", "never", "may", "might", "could", "can", "cannot", "will", "would",
    "should", "must", "is", "are", "was", "were", "be", "been", "likely", "to", "tends", "tend",
    "appears", "appear", "seems", "seem", "probably", "possibly", "really", "actually", "also",
    "generally", "significantly", "strongly", "substantially", "greatly", "directly", "causally",
    "indeed", "often", "usually", "typically", "fails", "fail", "has", "have", "had", "score",
    "scores", "earn", "earns", "perform", "performs", "get", "gets", "show", "shows", "see", "sees",
    "a", "an", "the", "that", "it", "on", "average", "linearly", "only",
}
LEAD_DROP = {"the", "a", "an", "their", "his", "her", "its", "of", "on", "in", "at", "for", "to",
             "people's", "students'", "that", "than", "and", "average", "mean", "overall"}

# --------------------------------------------------------------------------- verbs
_ADV = r"(?:\w+ly\s+)?"
INC_WORDS = (r"higher|more|greater|better|larger|bigger|increased|improved|an?\s+(?:\w+\s+)?"
             r"(?:increase|rise|gain|boost|improvement|jump|growth)\s+(?:in|of)")
DEC_WORDS = (r"lower|less|fewer|worse|smaller|reduced|decreased|an?\s+(?:\w+\s+)?"
             r"(?:decrease|drop|decline|reduction|fall|loss|cut)\s+(?:in|of)|loss\s+of")
LEAD_VERBS = (r"leads?\s+to|led\s+to|results?\s+in|resulted\s+in|caus(?:es|e|ed)|produc(?:es|e|ed)|"
              r"brings?\s+about|(?:is|are|was|were)?\s*associated\s+with|(?:is|are)?\s*linked\s+to|"
              r"(?:is|are)?\s*related\s+to|correlates?\s+with|predicts?|yields?")

NO_EFFECT_RE = re.compile(
    r"(?:has|have|had|shows?|showed|makes?|made)\s+(?:no|zero)\s+(?:\w+\s+)?"
    r"(?:effect|impact|influence|bearing|difference|relationship|association)\s*(?:on|to|in|for|with)?"
    r"|\bno\s+(?:\w+\s+)?(?:effect|impact|influence)\s+on"
    r"|(?:does|do|did|will|would|can|could|may|might)?\s*(?:not|never|cannot)\s+" + _ADV +
    r"(?:affect|change|impact|influence|alter|cause|matter\s+for|make\s+a\s+difference\s+(?:to|in|for))\b"
    r"|(?:is|are|was|were)\s+(?:not\s+related|unrelated|independent)\s+(?:to|of)"
)
COMPARATIVE_VERB = (r"(?:have|has|had|get|gets|got|show|shows|showed|see|sees|earn|earns|earned|"
                    r"score|scores|scored|perform|performs|performed|are|is|were|was|do|does|did)\s+"
                    r"(?:a\s+)?(?:\w+ly\s+|much\s+)?")
VERB_TABLE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:" + LEAD_VERBS + r")\s+(?:a\s+|an\s+)?(?:\w+\s+)?(?:" + INC_WORDS + r")\b"), "increase"),
    (re.compile(r"\b(?:" + LEAD_VERBS + r")\s+(?:a\s+|an\s+)?(?:\w+\s+)?(?:" + DEC_WORDS + r")\b"), "decrease"),
    (re.compile(r"\b" + COMPARATIVE_VERB + r"(?:higher|more|greater|better|larger|bigger)\b"), "increase"),
    (re.compile(r"\b" + COMPARATIVE_VERB + r"(?:lower|less|fewer|worse|smaller)\b"), "decrease"),
    (re.compile(r"\b(?:is|are|was|were)\s+(?:good|beneficial|helpful)\s+for\b"), "increase"),
    (re.compile(r"\b(?:is|are|was|were)\s+(?:bad|harmful|detrimental)\s+(?:for|to)\b"), "decrease"),
    (re.compile(r"\bdoubl(?:es|e|ed|ing)\b"), "x2"),
    (re.compile(r"\btripl(?:es|e|ed|ing)\b"), "x3"),
    (re.compile(r"\bquadrupl(?:es|e|ed|ing)\b"), "x4"),
    (re.compile(r"\b(?:halv(?:es|e|ed|ing)|cuts?\s+in\s+half)\b"), "half"),
    (re.compile(r"\b(?:increas\w*|rais(?:e|es|ed|ing)|boost\w*|improv\w*|enhanc\w*|lift(?:s|ed|ing)?|"
                r"elevat\w*|grow(?:s|ing)?|grew|strengthen\w*|adds?\s+to|benefit(?:s|ed)?|help(?:s|ed)?|"
                r"promot\w*|stimulat\w*|accelerat\w*|expand\w*|multipl\w*|pushes\s+up|drives?\s+up)\b"), "increase"),
    (re.compile(r"\b(?:decreas\w*|reduc\w*|lower(?:s|ed|ing)?|cut(?:s|ting)?|hurt(?:s|ing)?|worsen\w*|"
                r"harm(?:s|ed)?|damag\w*|diminish\w*|declin\w*|shrink\w*|drop(?:s|ped)?|lessen\w*|"
                r"weaken\w*|undermin\w*|impair\w*|slash\w*|suppress\w*|prevent\w*|inhibit\w*|curb\w*|"
                r"depress\w*|pushes\s+down|drives?\s+down)\b"), "decrease"),
    (re.compile(r"\b(?:" + LEAD_VERBS + r")\b"), "nonzero"),
    (re.compile(r"\b(?:affect(?:s|ed|ing)?|(?:has|have|had)\s+(?:an?|some)\s+(?:\w+\s+)?(?:effect|impact|influence)\s+on|"
                r"impact(?:s|ed)?|influenc(?:es|e|ed)|chang(?:es|e|ed)|matters?\s+for|"
                r"makes?\s+a\s+(?:\w+\s+)?difference\s+(?:to|in|for)|alter(?:s|ed)?|shap(?:es|e|ed)|"
                r"modif(?:ies|y|ied)|determin(?:es|e|ed))\b"), "nonzero"),
]
NEG_BEFORE = re.compile(r"(?:\b(?:not|never|cannot|no\s+longer)\s+(?:\w+ly\s+)?|\bfails?\s+to\s+(?:\w+ly\s+)?)$")

# --------------------------------------------------------------------------- size
BOUND_PAT = (r"no\s+more\s+than|not\s+more\s+than|no\s+less\s+than|not\s+less\s+than|at\s+least|"
             r"more\s+than|greater\s+than|in\s+excess\s+of|upwards\s+of|over|above|at\s+most|up\s+to|"
             r"less\s+than|fewer\s+than|under|below|about|around|roughly|approximately|approx\.?|"
             r"some|nearly|almost|close\s+to|~|exactly|precisely|just")
BOUND_MAP = {
    "no more than": "at_most", "not more than": "at_most", "no less than": "at_least",
    "not less than": "at_least", "at least": "at_least", "more than": "at_least",
    "greater than": "at_least", "in excess of": "at_least", "upwards of": "at_least", "over": "at_least",
    "above": "at_least", "at most": "at_most", "up to": "at_most", "less than": "at_most",
    "fewer than": "at_most", "under": "at_most", "below": "at_most", "about": "about", "around": "about",
    "roughly": "about", "approximately": "about", "approx": "about", "approx.": "about", "some": "about",
    "nearly": "about", "almost": "about", "close to": "about", "~": "about", "exactly": "exactly",
    "precisely": "exactly", "just": "exactly",
}
UNIT_PAT = (r"percentage\s+points?|percent(?:age)?\s+points?|%\s*points?|pp\b|p\.p\.|"
            r"%|percent\b|per\s+cent\b|pct\b|"
            r"standard\s+deviations?|std\.?\s+devs?\b|sds?\b|σ|"
            r"times\b|x\b|-?fold\b|"
            r"points?\b|units?\b|pts?\b")
SIZE_RE = re.compile(
    r"(?:\bby\s+)?(?:(?P<bound>" + BOUND_PAT + r")\s*)?(?<![\w.])(?P<num>\d+(?:\.\d+)?|\.\d+)\s*(?P<unit>" + UNIT_PAT + r")?"
    r"(?P<after>\s+(?:higher|lower|more|less|greater|better|worse|larger|smaller|fewer))?"
)


def _unit_of(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip().lower()
    if "point" in u and ("percent" in u or "%" in u) or u in ("pp", "p.p."):
        return "pp"
    if u in ("%", "percent", "per cent", "pct"):
        return "percent"
    if u.startswith("standard") or u.startswith("std") or u in ("sd", "sds", "σ"):
        return "sd"
    if u in ("times", "x") or "fold" in u:
        return "times"
    return "absolute"


# --------------------------------------------------------------------------- shape / untestable
SHAPE_CLAUSE = [
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+)?(?:with\s+)?(?:diminishing|decreasing)\s+(?:marginal\s+)?returns?[^,;.|]*"), "diminishing"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+|which\s+|that\s+)?(?:\w+\s+)?(?:levels?|levelling|leveling)\s+off[^,;.|]*"), "diminishing"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+)?(?:\w+\s+)?(?:plateaus?|tapers?\s+off|saturates?)[^,;.|]*"), "diminishing"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+)?only\s+(?:above|beyond|after|past|once|when|over)\b[^,;.|]*"), "threshold"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+)?(?:above|beyond|past)\s+(?:a|some|certain|a\s+certain)\s+(?:\w+\s+)?(?:threshold|level|point)[^,;.|]*"), "threshold"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+|with\s+)?(?:a\s+)?threshold(?:\s+effect)?[^,;.|]*"), "threshold"),
    (re.compile(r"[,;]?\s*(?:in\s+a\s+|with\s+a\s+|following\s+a\s+)?(?:inverted[\s-]*u|u)[\s-]*shaped?(?:\s+(?:way|fashion|pattern|relationship|curve))?"), "nonmonotone"),
    (re.compile(r"[,;]?\s*(?:in\s+an?\s+)?inverted[\s-]*u\b(?:\s+(?:way|fashion|pattern|relationship|curve))?"), "nonmonotone"),
    (re.compile(r"[,;]?\s*(?:but\s+|and\s+)?(?:only\s+)?up\s+to\s+a\s+point[^,;.|]*"), "nonmonotone"),
    (re.compile(r"\bnon[\s-]?monoton\w*"), "nonmonotone"),
    (re.compile(r"[,;]?\s*(?:in\s+a\s+)?linear(?:ly)?(?:\s+(?:way|fashion|relationship))?\b"), "linear"),
    (re.compile(r"[,;]?\s*(?:in\s+)?proportion(?:al|ally)?(?:\s+to\s+[^,;.|]*)?"), "linear"),
    (re.compile(r"\b(?:for\s+)?(?:each|every)\s+(?:additional|extra|more)\b"), "linear"),
]
MORE_MORE_RE = re.compile(
    r"\bthe\s+(?P<a>more|higher|greater|bigger|larger|less|lower|fewer|smaller)\s+(?P<t>[^,;]+?),?\s+"
    r"the\s+(?P<b>more|higher|greater|better|bigger|larger|less|lower|fewer|worse|smaller)\s+(?P<o>[^,;.]+)")
MEDIATION_RE = re.compile(
    r"\b(?:through|via|mediated\s+by|by\s+way\s+of|by\s+means\s+of|because\s+(?:it|they|this)\s+\w+|"
    r"by\s+(?:increasing|raising|reducing|lowering|improving|boosting|changing|affecting|making))\b[^,;.]*")
COUNTERFACTUAL_RE = re.compile(
    r"\bwould\s+(?:not\s+)?have\b|\bwould've\b|\bhad\s+(?:they|he|she|it|we|i|you)\s+not\b|"
    r"\bif\s+(?:\w+\s+){0,4}had\b|\bhad\s+(?:\w+\s+){0,3}not\s+been\b")
CONTROL_WORDS = r"(?:the\s+)?(?:untreated|control|controls|placebo|nothing|none|no\s+\w+|not\b|non[\s-]?\w+|those|people|others|units|students|the\s+rest|without)"
COMPARISON_RE = re.compile(
    r"\b(?:(?:more|less|better|worse|greater|higher|lower|faster|slower|stronger|weaker|larger|smaller)\s+than"
    r"|compared\s+(?:with|to)|relative\s+to|versus|vs\.?|as\s+opposed\s+to|in\s+contrast\s+to)\s+"
    r"(?!\d|\.\d|" + CONTROL_WORDS + r")[^,;.|]*")

# --------------------------------------------------------------------------- population
POP_WORDS = (r"(?:people|persons|adults|individuals|patients|participants|men|women|students|workers|"
             r"users|subjects|those|ones|children|kids|employees|customers|respondents|households|firms)")
AGE_RE = re.compile(
    r"(?:\b(?:among|amongst|for|in|within)\s+)?(?:(?:the\s+)?" + POP_WORDS + r"\s+)?"
    r"(?:(?:who\s+are|that\s+are|aged?|of\s+age)\s+)?"
    r"(?P<cmp>over|above|older\s+than|under|below|younger\s+than|at\s+least|at\s+most|more\s+than|less\s+than)\s+"
    r"(?:the\s+age\s+of\s+|age\s+)?(?P<num>\d+(?:\.\d+)?)(?P<plus>\s*\+)?"
    r"(?P<yrs>\s*(?:years?|yrs?|year[\s-]olds?)(?:\s+(?:old|of\s+age))?)?")
AGED_RE = re.compile(
    r"(?:\b(?:among|amongst|for|in|within)\s+)?(?:(?:the\s+)?" + POP_WORDS + r"\s+)?"
    r"\baged?\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<tail>\+|and\s+(?:over|older|above|up)|or\s+(?:over|older|more)|"
    r"and\s+(?:under|younger|below)|or\s+(?:under|younger|less))")
NUM_COND_RE = re.compile(
    r"\b(?:among|amongst|for|in|within)?\s*(?:" + POP_WORDS + r"\s+)?(?:with|whose)\s+(?P<col>[a-z][\w ]*?)\s+"
    r"(?P<cmp>above|over|below|under|greater\s+than|less\s+than|more\s+than|at\s+least|at\s+most|>=|<=|>|<)\s+"
    r"(?P<num>-?\d+(?:\.\d+)?)")
GROUP_RE = re.compile(
    r"\b(?P<prep>among|amongst|within|for|in)\s+(?P<grp>[a-z][\w'\- ]*?)"
    r"(?=\s*(?:[,;.|]|$|\bby\b|\bwith\b|\bwhen\b|\bif\b|\bthrough\b|\bvia\b|\bcompared\b|\bthan\b|\bbut\b|"
    r"\bbecause\b|\bwhile\b|\bwho\b|\bover\b|\bunder\b|\baged?\b|\bolder\b|\byounger\b))")
CMP_OP = {"over": ">", "above": ">", "older than": ">", "more than": ">", "greater than": ">", ">": ">",
          "under": "<", "below": "<", "younger than": "<", "less than": "<", "<": "<",
          "at least": ">=", ">=": ">=", "at most": "<=", "<=": "<="}


# --------------------------------------------------------------------------- token matching
def _stem(w: str) -> str:
    w = w.lower()
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    for suf in ("ment", "ing", "ed", "er"):
        if len(w) - len(suf) >= 3 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":   # enroll / enrol
        w = w[:-1]
    return w


_SYN_INDEX: dict[str, set[int]] = {}
for _i, _grp in enumerate(SYNONYMS):
    for _w in _grp:
        _SYN_INDEX.setdefault(_stem(_w), set()).add(_i)


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    sa, sb = _stem(a), _stem(b)
    if sa == sb:
        return 0.95
    if _SYN_INDEX.get(sa, set()) & _SYN_INDEX.get(sb, set()):
        return 0.75
    if len(a) >= 4 and len(b) >= 4:
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if r >= 0.75:
            return 0.75 * r
    return 0.0


def _content_tokens(text: Optional[str]) -> list[str]:
    if not text:
        return []
    toks = [w for w in re.split(r"[^a-z0-9]+", text.lower()) if w]
    out = [w for w in toks if w not in STOP and not re.fullmatch(r"\d+(?:\.\d+)?", w)]
    return out


def _col_tokens(name: str) -> list[str]:
    toks = words(name)
    content = [t for t in toks if t not in STOP]
    return content or toks


def _phrase_score(ptoks: list[str], name: str) -> float:
    ctoks = _col_tokens(name)
    if not ptoks or not ctoks:
        return 0.0
    if ptoks == ctoks:
        return 1.0
    cov_c = sum(max(_tok_sim(c, p) for p in ptoks) for c in ctoks) / len(ctoks)
    cov_p = sum(max(_tok_sim(p, c) for c in ctoks) for p in ptoks) / len(ptoks)
    return round(min(1.0, 0.75 * cov_c + 0.25 * cov_p), 3)


def _info(prep: Prepared, name: str):
    return next((c for c in prep.columns if c.name == name), None)


def _df_name(prep: Prepared, orig: str) -> str:
    enc = prep.encoding.get(orig) or []
    base = [c for c in enc if not c.endswith("__missing")]
    if len(base) == 1:
        return base[0]
    return orig


def _rank_columns(prep: Prepared, phrase: Optional[str]) -> list[tuple[str, float]]:
    ptoks = _content_tokens(phrase)
    if not ptoks:
        return []
    out = []
    for info in prep.columns:
        if info.kind in ("constant", "text"):
            continue
        s = _phrase_score(ptoks, info.name)
        if s > 0:
            out.append((info.name, s))
    out.sort(key=lambda kv: -kv[1])
    return out


# --------------------------------------------------------------------------- population matching
def _pop_from_group(prep: Prepared, phrase: str) -> tuple[Optional[dict], float, list[str]]:
    """Match a group phrase ('women', 'the north', 'members') to a filter."""
    notes: list[str] = []
    toks = [w for w in re.split(r"[^a-z0-9\-]+", phrase.lower()) if w]
    neg = False
    clean: list[str] = []
    for i, w in enumerate(toks):
        if w in ("not", "non", "without", "no"):
            neg = True
            continue
        if w.startswith("non-"):
            neg = True
            w = w[4:]
        clean.append(w)
    content = [w for w in clean if w not in STOP and w not in {"who", "were", "live", "living", "located",
                                                                 "based", "from", "area", "areas"}]
    if not content:
        return None, 0.0, notes
    best: tuple[Optional[dict], float] = (None, 0.0)

    def consider(f: dict, s: float):
        nonlocal best
        if s > best[1]:
            best = (f, s)

    fem = bool(set(content) & FEMALE_WORDS)
    mal = bool(set(content) & MALE_WORDS)
    for info in prep.columns:
        name = info.name
        ctoks = set(_col_tokens(name))
        if info.kind == "binary":
            # gender columns
            if fem or mal:
                if ctoks & {"female", "woman", "women", "girl", "girls", "females"}:
                    val = 1 if fem else 0
                    s = 1.0 if (set(content) & ctoks) else 0.9
                    consider({"column": _df_name(prep, name), "op": "==", "value": val ^ int(neg)}, s)
                elif ctoks & {"male", "man", "men", "boy", "boys", "males"}:
                    val = 1 if mal else 0
                    s = 1.0 if (set(content) & ctoks) else 0.9
                    consider({"column": _df_name(prep, name), "op": "==", "value": val ^ int(neg)}, s)
            # binary column named in the phrase ('members', 'untreated')
            s = _phrase_score(content, name)
            val, s_un = 1, 0.0
            stripped = [w[2:] if w.startswith("un") and len(w) > 4 else w for w in content]
            if stripped != content:
                s_un = _phrase_score(stripped, name)
            if s_un > s:
                s, val = s_un, 0
            if neg:
                val = 1 - val
            if s >= 0.5:
                consider({"column": _df_name(prep, name), "op": "==", "value": val}, s)
        elif info.kind == "categorical" and name in prep.raw:
            try:
                levels = list(pd.Series(prep.raw[name]).dropna().unique())[:50]
            except Exception:  # noqa: BLE001
                levels = []
            name_bonus = 0.05 if set(content) & ctoks else 0.0
            for lev in levels:
                ltoks = [t for t in words(str(lev))] or [str(lev).lower()]
                lt = [t for t in content if t not in ctoks] or content
                s = _phrase_score(lt, " ".join(ltoks))
                if fem and set(ltoks) & {"female", "f", "woman", "women"}:
                    s = max(s, 0.9)
                if mal and set(ltoks) & {"male", "m", "man", "men"}:
                    s = max(s, 0.9)
                s = min(1.0, s + name_bonus) if s > 0 else 0.0
                if s < 0.5:
                    continue
                onehot = f"{name}={lev}"
                if onehot in prep.df.columns and not neg:
                    f = {"column": onehot, "op": "==", "value": 1}
                elif onehot in prep.df.columns and neg:
                    f = {"column": onehot, "op": "==", "value": 0}
                else:
                    f = {"column": name, "op": "!=" if neg else "==", "value": lev, "raw": True}
                consider(f, s)
    return best[0], best[1], notes


def _age_column(prep: Prepared) -> Optional[str]:
    for info in prep.columns:
        if info.kind in ("continuous", "count") and set(_col_tokens(info.name)) & {"age", "aged", "years", "yrs"}:
            return info.name
    return None


# --------------------------------------------------------------------------- helpers
def _clean_phrase(p: Optional[str], trailing: bool = True) -> Optional[str]:
    if p is None:
        return None
    p = re.sub(r"\s+", " ", p.replace("|", " ")).strip(" ,;.:!?\"'")
    toks = p.split()
    while toks and toks[0] in LEAD_DROP:
        toks.pop(0)
    if trailing:
        while toks and toks[-1] in TRAIL_AUX:
            toks.pop()
    p = " ".join(toks).strip(" ,;.:")
    return p or None


def _cut(text: str, span: tuple[int, int], fill: str = " | ") -> str:
    return text[: span[0]] + fill + text[span[1]:]


def _normalize(text: str) -> str:
    s = text.strip().lower()
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", " - ").replace("−", "-")
    contractions = {"doesn't": "does not", "don't": "do not", "didn't": "did not", "isn't": "is not",
                    "aren't": "are not", "wasn't": "was not", "weren't": "were not", "won't": "will not",
                    "can't": "cannot", "can not": "cannot", "hasn't": "has not", "haven't": "have not",
                    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not"}
    for k, v in contractions.items():
        s = s.replace(k, v)
    s = re.sub(r"\b(" + "|".join(NUMBER_WORDS) + r")(?=\s+(?:percent|times|points?|percentage|standard|sds?\b|fold|units?|%))",
               lambda m: NUMBER_WORDS[m.group(1)], s)
    s = re.sub(r"\bhalf\s+(?:a|an|one)\s+(standard|sd)", r"0.5 \1", s)
    s = re.sub(r"\s+", " ", s)
    return s


# --------------------------------------------------------------------------- main
def parse_claim(text: str, prep: Prepared) -> Claim:
    raw = text if isinstance(text, str) else ("" if text is None else str(text))
    if not raw.strip():
        return Claim(raw="", testable=False, untestable_reason="no claim given",
                     notes=["No claim was given."])
    claim = Claim(raw=raw)
    try:
        _parse(claim, prep)
    except Exception as e:  # noqa: BLE001
        claim.notes.append(f"The claim could not be fully read ({type(e).__name__}: {e}).")
    try:
        claim.estimand = _estimand(claim, prep)
    except Exception as e:  # noqa: BLE001
        claim.notes.append(f"Could not build the estimand ({type(e).__name__}: {e}).")
        claim.estimand = Estimand(claim.treatment_col, claim.outcome_col)
    return claim


def _parse(claim: Claim, prep: Prepared) -> None:
    s = _normalize(claim.raw)
    work = s

    # hedges --------------------------------------------------------------------
    for h in HEDGES:
        if re.search(r"\b" + re.escape(h) + r"\b", s):
            claim.hedges.append(h)
    work = re.sub(r"\bon\s+average\b,?", " ", work)
    work = re.sub(r"\b(?:\w+\s+)?(?:suggests?|shows?|finds?|found|showed|indicates?|believes?)\s+that\b", " | ", work)

    # untestable ----------------------------------------------------------------
    reasons: list[str] = []
    m = MEDIATION_RE.search(work)
    if m:
        reasons.append(f"It is about the path the effect takes (\"{m.group(0).strip()}\"): a mediation claim. "
                       "The data can test the total effect, not how much goes through a particular path "
                       "without extra assumptions that cannot be checked.")
        work = _cut(work, m.span())
    cf = COUNTERFACTUAL_RE.search(s)
    if cf:
        reasons.append("It is worded as a counterfactual about specific units (what would have happened to them). "
                       "Data can only show average effects across groups, not what would have happened to "
                       "particular individuals.")
    m = COMPARISON_RE.search(work)
    if m:
        reasons.append(f"It compares against another treatment (\"{m.group(0).strip()}\"). The data hold one "
                       "treatment versus none, so the comparison with the other treatment cannot be tested.")
        work = _cut(work, m.span())
    if reasons:
        claim.testable = False
        claim.untestable_reason = " ".join(reasons)

    # "the more X, the more Y" --------------------------------------------------
    mm = MORE_MORE_RE.search(work)
    if mm:
        up = {"more", "higher", "greater", "better", "bigger", "larger"}
        same = (mm.group("a") in up) == (mm.group("b") in up)
        claim.direction = "increase" if same else "decrease"
        claim.shape = "linear"
        t_phrase = _clean_phrase(mm.group("t"))
        o_phrase = mm.group("o")
        claim.treatment_phrase = t_phrase
        work = work[: mm.start()] + " " + work[mm.end():]
        o_phrase, _ = _strip_population(claim, prep, o_phrase)
        claim.outcome_phrase = _clean_phrase(re.split(r"[,;|]", o_phrase)[0])
        _strip_population(claim, prep, work)
        _extract_size(claim, work)
        _match_columns(claim, prep)
        return

    # shape ----------------------------------------------------------------------
    for rx, shape in SHAPE_CLAUSE:
        m = rx.search(work)
        if m:
            if claim.shape is None:
                claim.shape = shape
            elif claim.shape != shape:
                claim.notes.append(f"More than one shape mentioned; using '{claim.shape}'.")
            work = _cut(work, m.span(), " | " if m.group(0).lstrip()[:1] in ",;" else " ")

    # population -----------------------------------------------------------------
    work, _ = _strip_population(claim, prep, work)

    # size -------------------------------------------------------------------------
    work = _extract_size(claim, work)

    # counterfactual-only parse ------------------------------------------------------
    if cf and _counterfactual_parse(claim, work):
        _match_columns(claim, prep)
        return

    # verb -------------------------------------------------------------------------
    hits = []
    m = NO_EFFECT_RE.search(work)
    if m:
        hits.append((m.start(), -1, m, "no_effect"))
    for rank, (rx, d) in enumerate(VERB_TABLE):
        m = rx.search(work)
        if m:
            hits.append((m.start(), rank, m, d))
    if not hits:
        claim.notes.append("No verb saying what the treatment does to the outcome was found "
                           "(e.g. 'increases', 'reduces', 'affects'). Columns were matched from the words used.")
        _fallback_mentions(claim, prep, work)
        return
    hits.sort(key=lambda h: (h[0], h[1]))
    _, _, vm, d = hits[0]
    pre, post = work[: vm.start()], work[vm.end():]
    if d == "no_effect":
        claim.direction = "no_effect"
        if claim.size is not None:
            claim.notes.append("A size was given together with 'no effect'; the size is ignored for the direction.")
    else:
        if d.startswith("x"):
            claim.direction = "increase"
            if claim.size is None:
                claim.size, claim.size_unit = float(d[1:]), "times"
        elif d == "half":
            claim.direction = "decrease"
            if claim.size is None:
                claim.size, claim.size_unit = 50.0, "percent"
                claim.notes.append("'Halves' is read as a 50% decrease.")
        else:
            claim.direction = d
        neg = NEG_BEFORE.search(pre)
        if neg:
            pre = pre[: neg.start()]
            if d == "nonzero":
                claim.direction = "no_effect"
            else:
                claim.negated = True
                word = {"increase": "an increase", "decrease": "a decrease"}.get(claim.direction, "an effect")
                claim.notes.append(f"The claim is negated: it says the effect is NOT {word}. It is supported "
                                   f"if the data rule out {word}, not if they show the opposite.")
        if d == "nonzero" and re.search(r"caus|lead|led|result|produc|bring", vm.group(0)):
            claim.notes.append(f"'{vm.group(0).strip()}' with no direction is read as 'changes' (any direction).")

    # passive: "earnings are raised by the program"
    passive = re.search(r"\b(?:is|are|was|were|be|been|gets?|got)\s+$", pre) and re.search(r"ed$", vm.group(0).strip())
    by = re.match(r"\s*by\s+([^,;.|]+)", post) if passive else None
    if by:
        claim.treatment_phrase = _clean_phrase(by.group(1))
        claim.outcome_phrase = _clean_phrase(re.sub(r"\b(?:is|are|was|were|be|been|gets?|got)\s*$", "", pre).split("|")[-1],
                                             trailing=False)
    else:
        subj = re.split(r"[|,;:]", pre)[-1]
        claim.treatment_phrase = _clean_phrase(subj)
        obj = re.split(r"[|,;:.]|\s(?:by|with|when|if|than|because|while|but|and|so)\s", " " + post + " ")[0]
        obj = re.sub(r"^\s*(?:on|in|of|for|at|to)\s+", " ", obj)
        claim.outcome_phrase = _clean_phrase(obj, trailing=False)
        if claim.outcome_phrase:
            claim.outcome_phrase = re.sub(r"\s+(?:higher|lower|more|less)$", "", claim.outcome_phrase) or None
    _match_columns(claim, prep)


def _strip_population(claim: Claim, prep: Prepared, work: str) -> tuple[str, bool]:
    found: list[tuple[str, Optional[dict], float, list]] = []   # phrase, filter, score, candidates
    # ages / numeric conditions
    for rx in (AGED_RE, AGE_RE):
        m = rx.search(work)
        if not m:
            continue
        # a bare comparator needs some population context
        txt = m.group(0)
        if rx is AGE_RE and not (re.search(r"\b(?:among|amongst|for|in|within|aged?|who|that)\b", txt)
                                 or re.search(POP_WORDS, txt) or m.group("yrs")):
            continue
        num = float(m.group("num"))
        if rx is AGED_RE:
            tail = m.group("tail")
            op = "<=" if re.search(r"under|younger|below|less", tail) else ">="
        else:
            op = CMP_OP.get(re.sub(r"\s+", " ", m.group("cmp")), ">")
            if m.group("plus"):
                op = ">="
        col = _age_column(prep)
        value: Any = int(num) if num.is_integer() else num
        phrase = _clean_phrase(re.sub(r"^\s*(?:among|amongst|for|in|within)\s+", "", txt), trailing=False)
        if col:
            f = {"column": _df_name(prep, col), "op": op, "value": value}
            found.append((phrase, f, 1.0 if "age" in _col_tokens(col) else 0.9, [(f["column"], 1.0)]))
        else:
            found.append((phrase, None, 0.0, []))
            claim.notes.append(f"The claim is about '{phrase}', but no age column was found to restrict the data.")
        work = _cut(work, m.span())
        break
    m = NUM_COND_RE.search(work)
    if m:
        ranked = _rank_columns(prep, m.group("col"))
        num = float(m.group("num"))
        value = int(num) if num.is_integer() else num
        phrase = _clean_phrase(m.group(0), trailing=False)
        if ranked and ranked[0][1] >= 0.5:
            f = {"column": _df_name(prep, ranked[0][0]), "op": CMP_OP.get(re.sub(r"\s+", " ", m.group("cmp")), ">"),
                 "value": value}
            found.append((phrase, f, ranked[0][1], [(_df_name(prep, c), sc) for c, sc in ranked[:3]]))
            work = _cut(work, m.span())
    # groups
    pos = 0
    while True:
        m = GROUP_RE.search(work, pos)
        if not m:
            break
        grp = m.group("grp").strip()
        f, sc, _n = _pop_from_group(prep, grp)
        strong = m.group("prep") in ("among", "amongst", "within")
        if f is not None and sc >= 0.5:
            found.append((_clean_phrase(grp, trailing=False), f, sc, [(f["column"], sc)]))
            work = _cut(work, m.span())
            pos = m.start() + 3
        elif strong:
            found.append((_clean_phrase(grp, trailing=False), None, 0.0, []))
            claim.notes.append(f"The claim is about '{grp}', but no column or category matches that group.")
            work = _cut(work, m.span())
            pos = m.start() + 3
        else:
            pos = m.end()
    if not found:
        return work, False
    best = max(found, key=lambda x: x[2])
    if claim.population_phrase is None or best[2] > claim.population_score:
        claim.population_phrase = ", ".join(p for p, *_ in found if p)
        claim.population_filter = best[1]
        claim.population_score = float(best[2])
        cands = []
        for _p, _f, _s, cl in found:
            cands.extend(cl)
        claim.candidates["population"] = sorted(cands, key=lambda kv: -kv[1])[:3]
    if sum(1 for x in found if x[1] is not None) > 1:
        claim.notes.append("More than one population condition was given; only the best-matched one "
                           f"({best[0]}) is applied as a filter.")
    return work, True


def _extract_size(claim: Claim, work: str) -> str:
    # "twice" / "2x" style multipliers without a number pattern hit
    m = re.search(r"\btwice\b|\bthree\s+times\b", work)
    if m and claim.size is None:
        claim.size, claim.size_unit = (2.0 if "twice" in m.group(0) else 3.0), "times"
        work = _cut(work, m.span(), " ")
    for m in SIZE_RE.finditer(work):
        num_s = m.group("num")
        unit = _unit_of(m.group("unit"))
        has_by = m.group(0).lstrip().startswith("by")
        bound = m.group("bound")
        after = m.group("after")
        try:
            num = float(num_s)
        except ValueError:
            continue
        if unit is None and not (has_by or bound or after):
            continue
        if unit is None and 1900 <= num <= 2100 and not has_by:
            continue
        if claim.size is not None:
            claim.notes.append(f"More than one size found; using the first ({claim.size:g}).")
            break
        claim.size = num
        claim.size_unit = unit or "absolute"
        if bound:
            b = re.sub(r"\s+", " ", bound.strip())
            claim.bound = BOUND_MAP.get(b, BOUND_MAP.get(b.rstrip("."), None))
        if after and claim.direction is None:
            w = after.strip()
            claim.direction = "increase" if w in ("higher", "more", "greater", "better", "larger") else "decrease"
        fill = " " + (after.strip() if after else "") + " "
        work = work[: m.start()] + fill + work[m.end():]
        if claim.size_unit == "absolute" and not m.group("unit"):
            claim.notes.append("The size has no unit; it is read as outcome units.")
        break
    if claim.bound is None:
        m = re.search(r"\bno\s+more\s+than\b", work)
        if m and claim.size is None:
            pass
    return work


def _counterfactual_parse(claim: Claim, work: str) -> bool:
    m = re.search(r"\b(?:if|had|without)\b[^,;|]*?(?=,|\bwould\b|\btheir\b|$)", work)
    if not m:
        return False
    cond = m.group(0)
    rest = work[: m.start()] + " " + work[m.end():]
    t = re.sub(r"\b(?:if|had|they|he|she|it|we|you|i|not|been|have|has|without|gotten|got|received|taken|"
               r"the|a|an|being)\b", " ", cond)
    claim.treatment_phrase = _clean_phrase(t)
    o = re.sub(r"\bwould\b.*$", "", rest)
    o = re.sub(r"\b(?:their|his|her|its|the)\b", " ", o)
    claim.outcome_phrase = _clean_phrase(o, trailing=False)
    comp = re.search(r"\b(higher|more|greater|better|lower|less|fewer|worse)\b", rest)
    if comp:
        up = comp.group(1) in ("higher", "more", "greater", "better")
        flipped = bool(re.search(r"\bnot\b|\bwithout\b", cond))
        claim.direction = "increase" if up != flipped else "decrease"
    return bool(claim.treatment_phrase or claim.outcome_phrase)


def _fallback_mentions(claim: Claim, prep: Prepared, work: str) -> None:
    """No verb: take the columns named in the text, first = treatment, second = outcome."""
    toks = _content_tokens(work)
    hits = []
    for info in prep.columns:
        if info.kind in ("constant", "text"):
            continue
        ctoks = _col_tokens(info.name)
        positions = [i for i, t in enumerate(toks) if any(_tok_sim(t, c) >= 0.9 for c in ctoks)]
        if positions:
            hits.append((min(positions), info.name))
    hits.sort()
    if hits:
        claim.treatment_phrase = hits[0][1]
    if len(hits) > 1:
        claim.outcome_phrase = hits[1][1]
    _match_columns(claim, prep)


def _match_columns(claim: Claim, prep: Prepared) -> None:
    tr = _rank_columns(prep, claim.treatment_phrase)
    orank = _rank_columns(prep, claim.outcome_phrase)
    claim.candidates["treatment"] = [(_df_name(prep, c), s) for c, s in tr[:3]]
    claim.candidates["outcome"] = [(_df_name(prep, c), s) for c, s in orank[:3]]
    claim.candidates.setdefault("population", [])
    best = (None, None, -1.0)
    tops_t = tr[:5] or [(None, 0.0)]
    tops_o = orank[:5] or [(None, 0.0)]
    for tc, ts in tops_t:
        for oc, os_ in tops_o:
            if tc is not None and tc == oc:
                continue
            if ts + os_ > best[2]:
                best = (tc, oc, ts + os_)
    tc, oc = best[0], best[1]
    ts = dict(tr).get(tc, 0.0) if tc else 0.0
    os_ = dict(orank).get(oc, 0.0) if oc else 0.0
    if tc and ts < 0.3:
        tc, ts = None, 0.0
    if oc and os_ < 0.3:
        oc, os_ = None, 0.0
    claim.treatment_col = _df_name(prep, tc) if tc else None
    claim.outcome_col = _df_name(prep, oc) if oc else None
    claim.treatment_score = float(ts)
    claim.outcome_score = float(os_)
    claim.needs_confirmation = ts < 0.8 or os_ < 0.8
    if claim.treatment_phrase and not tc:
        claim.notes.append(f"No column matches the treatment '{claim.treatment_phrase}'; please pick it.")
    if claim.outcome_phrase and not oc:
        claim.notes.append(f"No column matches the outcome '{claim.outcome_phrase}'; please pick it.")
    if not claim.treatment_phrase:
        claim.notes.append("Could not find the treatment in the claim.")
    if not claim.outcome_phrase:
        claim.notes.append("Could not find the outcome in the claim.")
    for role, col in (("treatment", tc), ("outcome", oc)):
        if col and _df_name(prep, col) == col and len(prep.encoding.get(col, [col])) > 1 \
                and any(not c.endswith("__missing") and c != col for c in prep.encoding.get(col, [])):
            claim.notes.append(f"The {role} '{col}' has several categories; it must be set as the {role} role "
                               "(coded as one column) before it can be analysed.")
    if claim.direction is None and (tc or oc):
        claim.notes.append("No direction found; the claim is read as 'has some effect'.")
        claim.direction = "nonzero"


def _estimand(claim: Claim, prep: Prepared) -> Estimand:
    t, o = claim.treatment_col, claim.outcome_col
    t_kind = None
    if t:
        info = _info(prep, t)
        if info is None:
            orig = next((k for k, v in prep.encoding.items() if t in v), None)
            info = _info(prep, orig) if orig else None
        t_kind = info.kind if info else None
    pop = claim.population_phrase or "everyone in the data"
    pop_txt = f", among {claim.population_phrase}" if claim.population_phrase else ""
    o_txt = o or "the outcome"
    t_txt = t or "the treatment"
    if t_kind in ("continuous", "count"):
        contrast = "dose_response"
        text = f"Average change in {o_txt} per one-unit increase in {t_txt}{pop_txt}"
    elif claim.size_unit in ("percent", "times"):
        contrast = "relative"
        if t in (None, "treated", "treatment"):
            text = (f"Average change in {o_txt} for treated versus untreated units, relative to the untreated "
                    f"mean{pop_txt}")
        else:
            text = (f"Average change in {o_txt} for units with {t_txt} versus without, relative to the mean "
                    f"without it{pop_txt}")
    else:
        contrast = "difference"
        who = "treated and untreated units" if t in (None, "treated", "treatment") else \
              f"units with and without {t_txt}"
        text = f"Average difference in {o_txt} between {who}{pop_txt}"
    return Estimand(treatment=t, outcome=o, contrast=contrast, target="ATE", population=pop, text=text)
