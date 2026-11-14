"""Reading list: methods references for the design, plus an optional online search.

This section never changes the grade (affects_grade=False, status None). The online search
sends only the words of the claim's treatment and outcome phrases to OpenAlex -- no data
values, no identity.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Optional

from ..schema import Claim, Design, Prepared, Section

KEY = "literature"
TITLE = "Reading list"
OPENALEX = "https://api.openalex.org/works"


def _ref(title, authors, year, venue="", doi=None, url=None, topic=""):
    return {"title": title, "authors": authors, "year": year, "venue": venue, "doi": doi,
            "url": url or (f"https://doi.org/{doi}" if doi else None), "topic": topic, "source": "curated"}


# DOIs are included only where we are confident of them; otherwise omitted.
REFS = {
    "experiment": [
        _ref("The econometrics of randomized experiments", "Athey, S. & Imbens, G. W.", 2017,
             "Handbook of Economic Field Experiments, vol. 1", doi="10.1016/bs.hefe.2016.10.003",
             topic="analysing randomized experiments"),
    ],
    "attrition": [
        _ref("What Works Clearinghouse Standards Handbook, Version 5.0", "What Works Clearinghouse (IES)", 2022,
             "U.S. Department of Education", url="https://ies.ed.gov/ncee/wwc/Handbooks",
             topic="attrition boundaries"),
        _ref("Training, wages, and sample selection: Estimating sharp bounds on treatment effects", "Lee, D. S.",
             2009, "Review of Economic Studies", doi="10.1111/j.1467-937X.2009.00536.x", topic="trimming bounds"),
        _ref("Nonparametric bounds on treatment effects", "Manski, C. F.", 1990,
             "American Economic Review (Papers & Proceedings)", topic="worst-case bounds"),
    ],
    "interference": [
        _ref("Toward causal inference with interference", "Hudgens, M. G. & Halloran, M. E.", 2008,
             "Journal of the American Statistical Association", doi="10.1198/016214508000000292",
             topic="interference / spillovers"),
        _ref("Optimal design of experiments in the presence of interference",
             "Baird, S., Bohren, J. A., McIntosh, C. & Özler, B.", 2018, "Review of Economics and Statistics",
             doi="10.1162/rest_a_00716", topic="randomized saturation designs"),
    ],
    "did": [
        _ref("Difference-in-differences with multiple time periods", "Callaway, B. & Sant'Anna, P. H. C.", 2021,
             "Journal of Econometrics", doi="10.1016/j.jeconom.2020.12.001", topic="staggered adoption"),
        _ref("Difference-in-differences with variation in treatment timing", "Goodman-Bacon, A.", 2021,
             "Journal of Econometrics", doi="10.1016/j.jeconom.2021.03.014", topic="two-way fixed effects"),
        _ref("What's trending in difference-in-differences? A synthesis of the recent econometrics literature",
             "Roth, J., Sant'Anna, P. H. C., Bilinski, A. & Poe, J.", 2023, "Journal of Econometrics",
             doi="10.1016/j.jeconom.2023.03.008", topic="overview"),
    ],
    "iv": [
        _ref("Identification of causal effects using instrumental variables", "Angrist, J. D., Imbens, G. W. & Rubin, D. B.",
             1996, "Journal of the American Statistical Association", doi="10.1080/01621459.1996.10476902",
             topic="local average treatment effect"),
        _ref("Weak instruments in instrumental variables regression: Theory and practice",
             "Andrews, I., Stock, J. H. & Sun, L.", 2019, "Annual Review of Economics",
             doi="10.1146/annurev-economics-080218-025643", topic="weak instruments"),
        _ref("Estimation of the parameters of a single equation in a complete system of stochastic equations",
             "Anderson, T. W. & Rubin, H.", 1949, "Annals of Mathematical Statistics",
             doi="10.1214/aoms/1177730090", topic="weak-instrument-robust inference"),
    ],
    "rd": [
        _ref("Robust nonparametric confidence intervals for regression-discontinuity designs",
             "Calonico, S., Cattaneo, M. D. & Titiunik, R.", 2014, "Econometrica", doi="10.3982/ECTA11757",
             topic="local polynomial estimation"),
        _ref("Manipulation of the running variable in the regression discontinuity design: A density test",
             "McCrary, J.", 2008, "Journal of Econometrics", doi="10.1016/j.jeconom.2007.05.005",
             topic="manipulation test"),
        _ref("Simple local polynomial density estimators", "Cattaneo, M. D., Jansson, M. & Ma, X.", 2020,
             "Journal of the American Statistical Association", topic="manipulation test"),
    ],
    "observational": [
        _ref("Double/debiased machine learning for treatment and structural parameters",
             "Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W. & Robins, J.", 2018,
             "The Econometrics Journal", doi="10.1111/ectj.12097", topic="adjustment with many covariates"),
        _ref("Making sense of sensitivity: Extending omitted variable bias", "Cinelli, C. & Hazlett, C.", 2020,
             "Journal of the Royal Statistical Society: Series B", doi="10.1111/rssb.12348",
             topic="sensitivity to hidden confounding"),
        _ref("Sensitivity analysis in observational research: Introducing the E-value", "VanderWeele, T. J. & Ding, P.",
             2017, "Annals of Internal Medicine", doi="10.7326/M16-2607", topic="E-value"),
    ],
    "discovery": [
        _ref("Causation, Prediction, and Search (2nd ed.)", "Spirtes, P., Glymour, C. & Scheines, R.", 2000,
             "MIT Press", topic="constraint-based causal discovery"),
        _ref("On the completeness of orientation rules for causal discovery in the presence of latent confounders "
             "and selection bias", "Zhang, J.", 2008, "Artificial Intelligence", topic="FCI algorithm"),
        _ref("A linear non-Gaussian acyclic model for causal discovery",
             "Shimizu, S., Hoyer, P. O., Hyvärinen, A. & Kerminen, A.", 2006, "Journal of Machine Learning Research",
             url="https://www.jmlr.org/papers/v7/shimizu06a.html", topic="LiNGAM"),
    ],
}


def _query_words(claim: Optional[Claim]) -> str:
    if claim is None:
        return ""
    parts = [claim.treatment_phrase or "", claim.outcome_phrase or ""]
    words = re.findall(r"[A-Za-z][A-Za-z\-]+", " ".join(parts))
    return " ".join(words[:12])


def _online(query: str) -> list[dict]:
    url = OPENALEX + "?" + urllib.parse.urlencode({"search": query, "per-page": 5})
    req = urllib.request.Request(url, headers={"User-Agent": "CausalityAI", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 (fixed https host)
        payload = json.loads(resp.read().decode("utf-8"))
    out = []
    for w in (payload.get("results") or [])[:5]:
        doi = w.get("doi")
        if isinstance(doi, str):
            doi = doi.replace("https://doi.org/", "")
        authors = [((a or {}).get("author") or {}).get("display_name") for a in (w.get("authorships") or [])]
        authors = [a for a in authors if a]
        out.append({"title": w.get("display_name") or w.get("title"),
                    "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
                    "year": w.get("publication_year"), "venue": "", "doi": doi,
                    "url": (f"https://doi.org/{doi}" if doi else w.get("id")), "topic": "found by search",
                    "source": "openalex"})
    return out


def run(prep: Prepared, design: Design, claim: Optional[Claim], online: bool = False) -> Section:
    try:
        return _run(prep, design, claim, online)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, status=None, affects_grade=False,
                       summary="This reading list does not affect the grade.",
                       error=f"The reading list could not be built ({type(e).__name__}).")


def _run(prep, design, claim, online) -> Section:
    kind = getattr(design, "kind", None) if design is not None else None
    groups: list[str] = []
    if claim is None and kind is None:
        groups = ["discovery"]
    elif kind in REFS:
        groups = [kind]
    else:
        groups = ["observational"]
    if kind == "experiment":
        groups.append("attrition")
    elif prep is not None and prep.roles.outcome and prep.roles.outcome in prep.df \
            and prep.df[prep.roles.outcome].isna().any():
        groups.append("attrition")
    r = prep.roles if prep is not None else None
    if r is not None and (r.cluster or r.saturation):
        groups.append("interference")
    if claim is None and "discovery" not in groups:
        groups.append("discovery")
    refs = [dict(x) for g in groups for x in REFS[g]]
    warnings: list[str] = []
    metrics: dict = {"online": bool(online), "query": None}
    if online:
        q = _query_words(claim)
        metrics["query"] = q or None
        if not q:
            warnings.append("Online search unavailable: no treatment or outcome words to search for.")
        else:
            try:
                found = _online(q)
                refs.extend(found)
                metrics["n_online"] = len(found)
            except Exception:  # noqa: BLE001
                warnings.append("Online search unavailable.")
    metrics["references"] = refs
    summary = ("Background reading on the methods behind these checks"
               + (" and papers found by an online search for the claim's topic (not vetted)" if metrics.get("n_online") else "")
               + ". This list is optional and does not affect the grade.")
    return Section(key=KEY, title=TITLE, status=None, summary=summary, metrics=metrics, warnings=warnings,
                   affects_grade=False, caps_verdict=False)
