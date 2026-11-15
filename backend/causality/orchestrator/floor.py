"""Floor rules: the only checks code makes on the model's verdict.

`apply(draft, reading, evidence) -> (verdict, floor)`. Pure function over plain dicts; never raises
and never mutates its inputs. The floor may keep a label or move it to Inconclusive; it never upgrades.

The three rules (see ORCHESTRATOR.md, part B):

1. assumed_is_not_tested    - every untestable assumption named by any evidence item is listed in
                              `assumed`; a sensitivity item read as "supports" becomes "not informative".
2. needs_a_real_interval    - a label other than Inconclusive needs an estimate with a finite 95%
                              interval of width > 0, for the same design, treatment and outcome as the reading
                              (and the same other roles, cutoff and covariates, so the interval belongs to the
                              specification the verdict reports).
3. no_more_than_the_interval - the label must not say more than the 95% interval shows.

A draft that cannot be read (not a dict, or a label outside contract.LABELS) is treated as
Inconclusive and recorded under the rule name "unreadable_draft".
"""
from __future__ import annotations

import copy
import math

from . import contract

RULE_ASSUMED = "assumed_is_not_tested"
RULE_INTERVAL = "needs_a_real_interval"
RULE_NO_MORE = "no_more_than_the_interval"
RULE_UNREADABLE = "unreadable_draft"

NO_EFFECT_BAND = 0.1  # No effect: 95% interval inside plus or minus this many outcome SDs.


# ---------------------------------------------------------------- small helpers

def _num(x):
    """A finite float, or None. Booleans are not numbers here."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _fmt(x) -> str:
    v = _num(x)
    if v is None:
        return repr(x)
    if v == 0:
        return "0"
    if abs(v) >= 1e5 or abs(v) < 1e-3:
        return f"{v:.3g}"
    return f"{v:.4g}"


def _copy(x):
    try:
        return copy.deepcopy(x)
    except Exception:
        try:
            return dict(x) if isinstance(x, dict) else x
        except Exception:
            return x


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _str_list(x) -> list:
    if isinstance(x, str):
        return [x] if x.strip() else []
    if isinstance(x, (list, tuple)):
        return [s for s in x if isinstance(s, str) and s.strip()]
    return []


def _items(evidence) -> list:
    if not isinstance(evidence, (list, tuple)):
        return []
    return [e for e in evidence if isinstance(e, dict)]


def _find(items: list, eid):
    if eid is None:
        return None
    for e in items:
        try:
            if e.get("id") == eid:
                return e
        except Exception:
            continue
    return None


def _name(x):
    return x.strip() if isinstance(x, str) and x.strip() else None


def _reading_cols(reading: dict):
    roles = _dict(reading.get("roles"))
    t = _name(reading.get("treatment")) or _name(roles.get("treatment"))
    o = _name(reading.get("outcome")) or _name(roles.get("outcome"))
    return _name(reading.get("design")), t, o


def _item_cols(item: dict):
    nums = _dict(item.get("numbers"))
    args = _dict(item.get("args"))
    nroles, aroles = _dict(nums.get("roles")), _dict(args.get("roles"))
    design = _name(nums.get("design")) or _name(args.get("design"))
    t = _name(nroles.get("treatment")) or _name(aroles.get("treatment"))
    o = _name(nroles.get("outcome")) or _name(aroles.get("outcome"))
    return design, t, o


_UNSET = object()


def _item_field(item: dict, key: str):
    """numbers[key] when numbers has it, else args[key], else _UNSET."""
    nums, args = _dict(item.get("numbers")), _dict(item.get("args"))
    if key in nums:
        return nums[key]
    if key in args:
        return args[key]
    return _UNSET


def _cov_set(x):
    """None (every non-role column), a frozenset of names, or _UNSET when unreadable."""
    if x is None:
        return None
    if isinstance(x, str):
        return frozenset([x]) if x.strip() else frozenset()
    if isinstance(x, (list, tuple)):
        return frozenset(s for s in x if isinstance(s, str))
    return _UNSET


def _all_other_columns(items: list, roles: dict):
    """What "every non-role column" stands for: the table's columns (from a table_facts item) minus
    the role columns. None when no table_facts item lists the columns."""
    for it in items or []:
        if not isinstance(it, dict) or it.get("tool") != "table_facts":
            continue
        cols = _dict(it.get("numbers")).get("columns")
        if isinstance(cols, list) and cols:
            names = {c.get("name") for c in cols if isinstance(c, dict) and isinstance(c.get("name"), str)}
            used = {v for k, v in roles.items() if k != "cutoff" and isinstance(v, str)}
            return frozenset(names - used)
    return None


def _spec_diffs(reading: dict, item: dict, items: list = None) -> list:
    """Rule 2's "same reading" beyond design, treatment and outcome: every other identifying role
    (instrument, running, cutoff, unit, time, cluster, strata) and the adjustment covariates."""
    diffs = []
    r_roles = _dict(reading.get("roles"))
    i_roles = _dict(_dict(item.get("numbers")).get("roles")) or _dict(_dict(item.get("args")).get("roles"))
    for k in contract.ROLE_KEYS:
        if k in ("treatment", "outcome"):
            continue
        rv, iv = _name(r_roles.get(k)), _name(i_roles.get(k))
        if rv != iv:
            diffs.append(f"{k} {iv!r} in the estimate vs {rv!r} in the reading")

    # cutoff: a cutoff the reading states must be the one the estimate used; a cutoff the estimate was
    # given must be stated in the reading. Left out on both sides = found from the data.
    r_cut = r_roles.get("cutoff")
    i_cut_given = i_roles.get("cutoff")
    i_cut_used = _dict(item.get("numbers")).get("cutoff_used")
    if r_cut is not None:
        rc = _num(r_cut)
        ic = _num(i_cut_given) if i_cut_given is not None else _num(i_cut_used)
        if rc is None or ic is None or not math.isclose(rc, ic, rel_tol=1e-9, abs_tol=1e-9):
            diffs.append(f"cutoff {_fmt(ic) if ic is not None else None!r} in the estimate vs {r_cut!r} in the reading")
    elif i_cut_given is not None:
        diffs.append(f"cutoff {i_cut_given!r} in the estimate vs None in the reading")

    r_cov = _cov_set(reading.get("covariates"))
    raw = _item_field(item, "covariates")
    i_cov = None if raw is _UNSET else _cov_set(raw)
    # "Every non-role column" on one side and a list of names on the other are the same adjustment
    # set when the list is exactly those columns. The table's column list comes from the table_facts
    # item; without one the two cannot be shown to be the same, so they count as different.
    same = r_cov == i_cov
    if not same and r_cov is not _UNSET and i_cov is not _UNSET and (r_cov is None) != (i_cov is None):
        every = _all_other_columns(items, {**i_roles, "treatment": _item_cols(item)[1], "outcome": _item_cols(item)[2]})
        named = i_cov if r_cov is None else r_cov
        same = every is not None and named == every
    if r_cov is _UNSET or i_cov is _UNSET or not same:
        def show(c, original):
            return "every non-role column" if c is None else repr(sorted(c)) if c is not _UNSET else repr(original)
        diffs.append(f"covariates {show(i_cov, raw)} in the estimate vs {show(r_cov, reading.get('covariates'))} "
                     "in the reading")
    return diffs


def _ci95(item: dict):
    ci = _dict(_dict(item.get("numbers")).get("ci"))
    raw = ci.get("95")
    if raw is None:
        raw = ci.get(95)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None, None, raw
    return _num(raw[0]), _num(raw[1]), raw


# ---------------------------------------------------------------- rule 1

def _rule_assumed(verdict: dict, items: list, violations: list) -> None:
    untestable = []
    for e in items:
        for s in _str_list(_dict(e.get("numbers")).get("untestable")):
            if s not in untestable:
                untestable.append(s)

    raw_assumed = verdict.get("assumed")
    assumed = _str_list(raw_assumed) if not isinstance(raw_assumed, list) else list(raw_assumed)
    present = {s for s in assumed if isinstance(s, str)}
    missing = [s for s in untestable if s not in present]
    if missing:
        assumed.extend(missing)
        violations.append({
            "rule": RULE_ASSUMED,
            "detail": (f"{len(missing)} assumption(s) that no data can test were not listed as assumed: "
                       + "; ".join(missing) + "."),
            "action": f"Added {len(missing)} item(s) to what the verdict assumes.",
        })
    if missing or not isinstance(raw_assumed, list):
        verdict["assumed"] = assumed

    judgments = verdict.get("judgments")
    if not isinstance(judgments, list):
        return
    new_judgments, changed_ids = [], []
    for j in judgments:
        if isinstance(j, dict) and j.get("reading") == "supports":
            item = _find(items, j.get("evidence_id"))
            if item is not None and item.get("tool") == "sensitivity":
                j = dict(j)
                j["reading"] = "not informative"
                changed_ids.append(str(j.get("evidence_id")))
        new_judgments.append(j)
    if changed_ids:
        verdict["judgments"] = new_judgments
        violations.append({
            "rule": RULE_ASSUMED,
            "detail": ("A sensitivity analysis says how strong an unmeasured cause would have to be to change "
                       "the result; it cannot show that no such cause exists. It was read as support in "
                       f"evidence {', '.join(changed_ids)}."),
            "action": f"Changed the reading of evidence {', '.join(changed_ids)} from \"supports\" to \"not informative\".",
        })


# ---------------------------------------------------------------- rule 2

def _rule_interval(verdict: dict, reading: dict, items: list):
    """Returns (problem_detail or None, estimate item or None)."""
    eid = verdict.get("estimate_evidence_id")
    if eid is None or (isinstance(eid, str) and not eid.strip()):
        return "The verdict does not point at an effect estimate (no estimate evidence id).", None
    item = _find(items, eid)
    if item is None:
        return f"The verdict points at evidence {eid!r}, which is not in the evidence list.", None
    if item.get("tool") != "estimate_effect":
        return (f"Evidence {eid!r} comes from {item.get('tool')!r}, not from an effect estimate."), None
    if item.get("ok") is not True:
        err = item.get("error")
        return (f"The estimate in evidence {eid!r} did not run"
                + (f": {err}" if isinstance(err, str) and err else ".")), None
    nums = _dict(item.get("numbers"))
    if nums.get("refused"):
        why = nums.get("refusal_reason")
        return (f"The estimator declined to give an estimate in evidence {eid!r}"
                + (f": {why}" if isinstance(why, str) and why else ".")), None
    lo, hi, raw = _ci95(item)
    if lo is None or hi is None:
        return (f"Evidence {eid!r} has no 95% interval with two finite bounds (it has {raw!r})."), None
    if not hi - lo > 0:
        return (f"The 95% interval in evidence {eid!r} runs from {_fmt(lo)} to {_fmt(hi)}, "
                "so it has no width; that is not a usable measure of uncertainty."), None

    r_design, r_t, r_o = _reading_cols(reading)
    i_design, i_t, i_o = _item_cols(item)
    diffs = []
    if r_design is None or r_design != i_design:
        diffs.append(f"design {i_design!r} in the estimate vs {r_design!r} in the reading")
    if r_t is None or r_t != i_t:
        diffs.append(f"treatment {i_t!r} in the estimate vs {r_t!r} in the reading")
    if r_o is None or r_o != i_o:
        diffs.append(f"outcome {i_o!r} in the estimate vs {r_o!r} in the reading")
    diffs += _spec_diffs(reading, item, items)
    if diffs:
        return (f"The estimate in evidence {eid!r} was not computed for the reading the verdict uses: "
                + "; ".join(diffs) + "."), None
    return None, item


# ---------------------------------------------------------------- rule 3

def _rule_no_more(label: str, direction, item: dict):
    """Returns problem_detail or None."""
    lo, hi, _ = _ci95(item)
    eid = item.get("id")
    span = f"The 95% interval in evidence {eid!r} runs from {_fmt(lo)} to {_fmt(hi)}"

    if label == contract.NO_EFFECT:
        sd = _num(_dict(item.get("numbers")).get("outcome_sd"))
        if sd is None or sd <= 0:
            return (f"{span}, but the estimate gives no usable outcome standard deviation, so there is no "
                    "way to say how small 'no effect' has to be.")
        band = NO_EFFECT_BAND * sd
        if lo >= -band and hi <= band:
            return None
        return (f"{span}. 'No effect' needs it inside plus or minus {_fmt(band)} "
                f"({NO_EFFECT_BAND} times the outcome standard deviation of {_fmt(sd)}).")

    if direction not in ("increase", "decrease", "none"):
        return (f"{span}, but the verdict does not say which direction the claim points "
                f"(claimed direction is {direction!r}), so '{label}' cannot be checked against it.")

    if label == contract.SUPPORTED:
        if direction == "increase":
            if lo > 0:
                return None
            return f"{span}. 'Supported' for a claimed increase needs the whole interval above zero."
        if direction == "decrease":
            if hi < 0:
                return None
            return f"{span}. 'Supported' for a claimed decrease needs the whole interval below zero."
        # claim of no effect: supported only if the interval is tight around zero
        sd = _num(_dict(item.get("numbers")).get("outcome_sd"))
        if sd is not None and sd > 0:
            band = NO_EFFECT_BAND * sd
            if lo >= -band and hi <= band:
                return None
            return (f"{span}. 'Supported' for a claim of no effect needs it inside plus or minus {_fmt(band)} "
                    f"({NO_EFFECT_BAND} times the outcome standard deviation of {_fmt(sd)}).")
        return (f"{span}, but the estimate gives no usable outcome standard deviation, so a claim of no "
                "effect cannot be checked.")

    if label == contract.NOT_SUPPORTED:
        if direction == "increase":
            if hi < 0:
                return None
            return f"{span}. 'Not supported' for a claimed increase needs the whole interval below zero."
        if direction == "decrease":
            if lo > 0:
                return None
            return f"{span}. 'Not supported' for a claimed decrease needs the whole interval above zero."
        if lo > 0 or hi < 0:
            return None
        return f"{span}. 'Not supported' for a claim of no effect needs the whole interval on one side of zero."
    return None


def _plain_sentence(rule: str, item) -> str:
    if rule == RULE_NO_MORE and isinstance(item, dict):
        lo, hi, _ = _ci95(item)
        return (f"The data cannot settle this claim: the 95% interval for the effect runs from {_fmt(lo)} "
                f"to {_fmt(hi)}, which is not enough to back the verdict the analysis first gave.")
    if rule == RULE_UNREADABLE:
        return "The data cannot settle this claim: the verdict could not be read, so no conclusion is drawn."
    return ("The data cannot settle this claim: there is no usable effect estimate with a real 95% interval "
            "for the treatment, outcome and design used, so no conclusion about the effect is drawn.")


# ---------------------------------------------------------------- public

def _downgrade(verdict: dict, rule: str, detail: str, old_label, item, violations: list) -> None:
    verdict["label"] = contract.INCONCLUSIVE
    verdict["sentence"] = _plain_sentence(rule, item)
    violations.append({"rule": rule, "detail": detail,
                       "action": f"Changed the label from {old_label!r} to 'Inconclusive' and replaced the sentence."})


def _apply(draft, reading, evidence):
    violations: list = []
    items = _items(evidence)
    reading = _dict(reading)

    if isinstance(draft, dict):
        verdict = _copy(draft)
        if not isinstance(verdict, dict):
            verdict = dict(draft)
    else:
        verdict = {}
    for k, v in contract.empty_verdict().items():
        verdict.setdefault(k, v)

    if not isinstance(draft, dict):
        verdict["label"] = contract.INCONCLUSIVE
        verdict["sentence"] = _plain_sentence(RULE_UNREADABLE, None)
        violations.append({"rule": RULE_UNREADABLE,
                           "detail": f"The draft verdict is not an object (got {type(draft).__name__}).",
                           "action": "Used an empty verdict labelled 'Inconclusive'."})

    _rule_assumed(verdict, items, violations)

    label = verdict.get("label")
    if label not in contract.LABELS:
        _downgrade(verdict, RULE_UNREADABLE,
                   f"The label {label!r} is not one of: " + ", ".join(contract.LABELS) + ".",
                   label, None, violations)
        label = contract.INCONCLUSIVE

    if label != contract.INCONCLUSIVE:
        problem, item = _rule_interval(verdict, reading, items)
        if problem:
            _downgrade(verdict, RULE_INTERVAL, problem, label, None, violations)
        else:
            problem = _rule_no_more(label, verdict.get("claimed_direction"), item)
            if problem:
                _downgrade(verdict, RULE_NO_MORE, problem, label, item, violations)

    return verdict, {"changed": bool(violations), "violations": violations}


def apply(draft, reading, evidence):
    """Apply the three floor rules. Returns (verdict, floor). Never raises, never mutates inputs."""
    try:
        return _apply(draft, reading, evidence)
    except Exception as exc:  # last resort: anything unexpected ends Inconclusive
        verdict = contract.empty_verdict()
        verdict["sentence"] = _plain_sentence(RULE_UNREADABLE, None)
        return verdict, {"changed": True, "violations": [{
            "rule": RULE_UNREADABLE,
            "detail": f"The floor rules could not read the verdict ({type(exc).__name__}).",
            "action": "Replaced the verdict with an empty one labelled 'Inconclusive'."}]}
