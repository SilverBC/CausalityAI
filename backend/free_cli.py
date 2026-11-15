"""Optional helpers for a model that is free to analyse the data itself (the inverted design).

The model does its own analysis. These commands are there when they save it time; none is required.
Each prints plain lines with numbers and no pass/fail labels.

  python free_cli.py facts        data.csv
  python free_cli.py layout       data.csv --treatment T --outcome Y
  python free_cli.py checks       data.csv --treatment T --outcome Y [--covariates a,b]
  python free_cli.py design-tests data.csv --design did|iv|rd|experiment|observational --treatment T --outcome Y
                                  [--unit U] [--time P] [--instrument Z] [--running X] [--cutoff C] [--covariates a,b]
  python free_cli.py estimate     data.csv   (same options as design-tests)
  python free_cli.py sensitivity  data.csv   (same options as design-tests)
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
ROLES = ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata")
TOOL = {"facts": "table_facts", "checks": "general_checks", "design-tests": "design_tests",
        "estimate": "estimate_effect", "sensitivity": "sensitivity"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["facts", "layout", "checks", "design-tests", "estimate", "sensitivity"])
    p.add_argument("data")
    p.add_argument("--design", default="observational", choices=["experiment", "did", "iv", "rd", "observational"])
    for r in ROLES:
        p.add_argument(f"--{r}")
    p.add_argument("--cutoff", type=float)
    p.add_argument("--covariates", help="comma-separated column names; default is every non-role column")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    try:
        from causality import data
        with open(a.data, "rb") as f:
            df = data.load_table(f.read(), "data.csv")
        cov = [c.strip() for c in a.covariates.split(",") if c.strip()] if a.covariates else None
        if a.command == "layout":
            from causality.orchestrator import layout
            lines, _ = layout.run(df, {"treatment": a.treatment, "outcome": a.outcome})
            print("\n".join(lines))
            return 0
        from causality.orchestrator import evidence
        roles = {k: getattr(a, k) for k in ROLES}
        roles["cutoff"] = a.cutoff
        if a.command == "facts":
            args = {}
        elif a.command == "checks":
            args = {"treatment": a.treatment, "outcome": a.outcome, "covariates": cov}
        else:
            args = {"design": a.design, "roles": roles, "covariates": cov}
        item = evidence.call(TOOL[a.command], args, df, "x")
        if not item.get("ok"):
            print("Could not run: " + str(item.get("error")))
            return 2
        print("\n".join(item.get("lines") or []))
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"Could not run: {type(e).__name__}: {e}. Nothing was concluded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
