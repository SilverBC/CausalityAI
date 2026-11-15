"""CausalityAI tools for a coding agent. You declare the design and the columns; the tool computes.

  python agent_cli.py card         data.csv
  python agent_cli.py checks       data.csv --treatment T --outcome Y [--covariates a,b,c]
  python agent_cli.py structure    data.csv --treatment T --outcome Y [--instruments z1,z2]
  python agent_cli.py design-tests data.csv --design did|iv|rd|experiment|observational --treatment T --outcome Y
                                   [--unit U] [--time P] [--instrument Z] [--running X] [--cutoff C] [--cluster G]
                                   [--covariates a,b,c]
  python agent_cli.py estimate     data.csv   (same options as design-tests)

--covariates limits adjustment to the columns you name (default: every column that is not a role).
Output is a few plain lines. Nothing here guesses a design or asks a question.
"""
from __future__ import annotations

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["card", "checks", "structure", "design-tests", "estimate"])
    p.add_argument("data")
    p.add_argument("--design", default="observational",
                   choices=["experiment", "did", "iv", "rd", "observational"])
    for r in ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata"):
        p.add_argument(f"--{r}")
    p.add_argument("--cutoff", type=float)
    p.add_argument("--covariates", help="comma-separated column names")
    p.add_argument("--instruments", help="comma-separated candidate instruments (structure only)")
    a = p.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    from causality import agent
    from causality.data import InputError
    try:
        df = agent.load(a.data)
        cov = [c.strip() for c in a.covariates.split(",") if c.strip()] if a.covariates else None
        if cov:
            missing = [c for c in cov if c not in df.columns]
            if missing:
                raise InputError(f"These covariates are not columns of the table: {', '.join(missing)}.")
        roles = {k: getattr(a, k) for k in agent.ROLE_KEYS if getattr(a, k)}
        roles["cutoff"] = a.cutoff
        if a.command == "card":
            print(agent.facts_card(df))
            return 0
        if not a.treatment or not a.outcome:
            raise InputError("Give --treatment and --outcome.")
        if a.command == "checks":
            lines = agent.general_checks(df, a.treatment, a.outcome, cov)
        elif a.command == "structure":
            inst = [c.strip() for c in (a.instruments or "").split(",") if c.strip()]
            lines = agent.structure_facts(df, a.treatment, a.outcome, inst, cov)
        elif a.command == "design-tests":
            lines = agent.design_tests(df, a.design, roles, cov)
        else:
            lines = agent.estimate_effect(df, a.design, roles, cov)
        print("\n".join(lines))
        return 0
    except InputError as e:
        print(f"Cannot run: {e}")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"Cannot run: something went wrong ({type(e).__name__}: {e}). Nothing was concluded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
