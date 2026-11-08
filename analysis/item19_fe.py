"""Item 19: Appendix H citation models plus research-topic fixed effects (top level-0 / level-1 OpenAlex concept).
Copy of fe_models.py so Appendix H stays reproducible; the first three specs must equal fe_model_results.csv.

Input: fe_model.parquet (one row per analyzed COVID-19 work; built with duckdb from covid_works_ow,
works_primary_location and works_authorships). Run: python fe_models.py <path to fe_model.parquet>
"""
import sys
import numpy as np
import pandas as pd
import pyfixest as pf

df = pd.read_parquet(sys.argv[1])
df["class_"] = pd.Categorical(df["class_"], categories=["0%", "1-50%", "51-99%", "100%"])
df["log_cites2"] = np.log1p(df["cites2"])
df["size"] = df["num_authors"].astype(str)   # team size as a fixed effect: one level per size 2..30

# Nested specifications: (1) size + month, (2) + country, (3) + venue = full two-way FE requested.
fes = {"size+month": "size + ym", "+country": "size + ym + country", "+venue (full)": "size + ym + country + venue",
       "+country+topic0": "size + ym + country + l0", "+country+topic1": "size + ym + country + l1",
       "+venue+topic1 (full)": "size + ym + country + venue + l1"}
rows = []
for model in ("PPML", "OLS log(1+c)"):
    for name, fe in fes.items():
        rhs = "C(class_, contr.treatment(base='0%'))"
        if model == "PPML":
            fit = pf.fepois(f"cites2 ~ {rhs} | {fe}", data=df, vcov={"CRV1": "venue"})
        else:
            fit = pf.feols(f"log_cites2 ~ {rhs} | {fe}", data=df, vcov={"CRV1": "venue"})
        t = fit.tidy()
        for term, r in t.iterrows():
            lvl = term.split("[T.")[-1].rstrip("]")
            rows.append(dict(model=model, spec=name, term=lvl, coef=r["Estimate"], se=r["Std. Error"],
                             lo=r["2.5%"], hi=r["97.5%"], n=fit._N))
out = pd.DataFrame(rows)
# exp(coef)-1 = % change in citations vs No-OS (exact for PPML, approximate for log-OLS)
out["pct"] = 100 * (np.exp(out.coef) - 1)
out["pct_lo"] = 100 * (np.exp(out.lo) - 1)
out["pct_hi"] = 100 * (np.exp(out.hi) - 1)
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
out.to_csv(sys.argv[1].replace(".parquet", "_results.csv"), index=False)
sys.exit(0)  # continuous frac_exp model not needed for item 19

# Continuous version on the full spec: share of outbreak scientists on the team.
for model, f in (("PPML", pf.fepois), ("OLS log(1+c)", pf.feols)):
    y = "cites2" if model == "PPML" else "log_cites2"
    fit = f(f"{y} ~ frac_exp | size + ym + country + venue", data=df, vcov={"CRV1": "venue"})
    r = fit.tidy().iloc[0]
    print(f"{model} frac_exp (0->1): coef {r['Estimate']:.3f} [{r['2.5%']:.3f}, {r['97.5%']:.3f}], "
          f"{100*(np.exp(r['Estimate'])-1):.1f}% , N={fit._N}")
