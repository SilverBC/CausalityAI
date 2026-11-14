"""Build app-ready CSVs from two classic real-world causal datasets.

Raw files (samples/real/raw/, kept in the repo so this script reproduces the CSVs):
  - NBER / Dehejia & Wahba (1999) LaLonde NSW subsample with 1974 earnings:
      nswre74_treated.txt, nswre74_control.txt, psid_controls.txt
      https://users.nber.org/~rdehejia/data/
  - Card & Krueger (1994) New Jersey / Pennsylvania fast-food survey:
      njmin.zip -> public.dat, codebook   https://davidcard.berkeley.edu/data_sets/njmin.zip

Outputs:
  lalonde_experimental.csv    NSW treated (185) vs NSW randomized controls (260)       benchmark ATE ≈ +$1,794
  lalonde_observational.csv   NSW treated (185) vs PSID non-experimental controls (2,490)  same truth, naive diff ≈ −$15,000
  card_krueger_minwage.csv    410 stores × 2 survey waves, long format                    benchmark DiD ≈ +2.76 FTE (SE 1.36)
"""
from pathlib import Path

import io
import zipfile

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent

COLS = ["treat", "age", "education", "black", "hispanic", "married", "no_degree", "earnings_1974", "earnings_1975", "earnings_1978"]


def lalonde():
    read = lambda f: pd.read_csv(RAW / f, sep=r"\s+", header=None, names=COLS)  # noqa: E731
    treated, control, psid = read("nswre74_treated.txt"), read("nswre74_control.txt"), read("psid_controls.txt")
    for d in (treated, control, psid):
        d["unemployed_1974"] = (d["earnings_1974"] == 0).astype(int)
        d["unemployed_1975"] = (d["earnings_1975"] == 0).astype(int)
        for c in ["treat", "age", "education", "black", "hispanic", "married", "no_degree"]:
            d[c] = d[c].astype(int)
    exp = pd.concat([treated, control], ignore_index=True)
    obs = pd.concat([treated, psid], ignore_index=True)
    exp.to_csv(OUT / "lalonde_experimental.csv", index=False)
    obs.to_csv(OUT / "lalonde_observational.csv", index=False)
    d_exp = exp.loc[exp.treat == 1, "earnings_1978"].mean() - exp.loc[exp.treat == 0, "earnings_1978"].mean()
    d_obs = obs.loc[obs.treat == 1, "earnings_1978"].mean() - obs.loc[obs.treat == 0, "earnings_1978"].mean()
    print(f"LaLonde experimental n={len(exp)}  diff in means = {d_exp:,.0f}")
    print(f"LaLonde observational n={len(obs)} diff in means = {d_obs:,.0f}")


def card_krueger():
    names = ["sheet", "chain", "co_owned", "state", "southj", "centralj", "northj", "pa1", "pa2", "shore",
             "ncalls", "empft", "emppt", "nmgrs", "wage_st", "inctime", "firstinc", "bonus", "pctaff", "meals",
             "open", "hrsopen", "psoda", "pfry", "pentree", "nregs", "nregs11",
             "type2", "status2", "date2", "ncalls2", "empft2", "emppt2", "nmgrs2", "wage_st2", "inctime2", "firstin2",
             "special2", "meals2", "open2r", "hrsopen2", "psoda2", "pfry2", "pentree2", "nregs2", "nregs112"]
    with zipfile.ZipFile(RAW / "njmin.zip") as z:
        raw = z.read(next(n for n in z.namelist() if n.lower().endswith("public.dat"))).decode()
    d = pd.read_csv(io.StringIO(raw), sep=r"\s+", header=None, names=names, na_values=".")
    chain = d["chain"].map({1: "Burger King", 2: "KFC", 3: "Roy Rogers", 4: "Wendy's"})
    base = pd.DataFrame({"store_id": d["sheet"], "chain": chain, "co_owned": d["co_owned"], "new_jersey": d["state"]})
    fte1 = d["empft"] + d["nmgrs"] + 0.5 * d["emppt"]
    fte2 = d["empft2"] + d["nmgrs2"] + 0.5 * d["emppt2"]
    w1 = base.assign(post=0, fte_employment=fte1, starting_wage=d["wage_st"])
    w2 = base.assign(post=1, fte_employment=fte2, starting_wage=d["wage_st2"])
    long = pd.concat([w1, w2], ignore_index=True).sort_values(["store_id", "post"])
    # treated store-period (NJ after the raise): the DiD estimator in backend/causality needs a treatment that
    # switches on over time, not just a state indicator
    long.insert(long.columns.get_loc("post") + 1, "nj_after_raise", long["new_jersey"] * long["post"])
    long.to_csv(OUT / "card_krueger_minwage.csv", index=False)
    m = long.groupby(["new_jersey", "post"])["fte_employment"].mean().unstack()
    did = (m.loc[1, 1] - m.loc[1, 0]) - (m.loc[0, 1] - m.loc[0, 0])
    print("Card-Krueger mean FTE:\n", m.round(2))
    print(f"Card-Krueger DiD of means = {did:.2f} (paper Table 3: 2.76)")


if __name__ == "__main__":
    lalonde()
    card_krueger()
