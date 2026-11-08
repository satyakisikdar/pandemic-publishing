"""Item 18 robustness (advisor): (1) full spec without venue='none'/country='unknown'; (2) log(team size) regressor."""
import sys, warnings
import numpy as np, pandas as pd, pyfixest as pf
warnings.filterwarnings("ignore")
df = pd.read_parquet(sys.argv[1])
df["class_"] = pd.Categorical(df["class_"], categories=["0%", "1-50%", "51-99%", "100%"])
df["log_cites2"] = np.log1p(df["cites2"]); df["size"] = df["num_authors"].astype(str); df["log_size"] = np.log(df["num_authors"])
rhs = "C(class_, contr.treatment(base='0%'))"
def show(tag, fit):
    t = fit.tidy()
    for term, r in t.iterrows():
        lvl = term.split("[T.")[-1].rstrip("]")
        print(f"{tag:42s} {lvl:9s} {100*(np.exp(r['Estimate'])-1):7.1f}% [{100*(np.exp(r['2.5%'])-1):6.1f}, {100*(np.exp(r['97.5%'])-1):6.1f}]  N={fit._N}")
sub = df[(df.venue != "none") & (df.country != "unknown")]
print(f"restricted sample: {len(sub):,} of {len(df):,}")
show("PPML full, known venue+country", pf.fepois(f"cites2 ~ {rhs} | size + ym + country + venue", data=sub, vcov={"CRV1": "venue"}))
show("OLS  full, known venue+country", pf.feols(f"log_cites2 ~ {rhs} | size + ym + country + venue", data=sub, vcov={"CRV1": "venue"}))
show("PPML +country, log(size) regressor", pf.fepois(f"cites2 ~ {rhs} + log_size | ym + country", data=df, vcov={"CRV1": "venue"}))
show("OLS  +country, log(size) regressor", pf.feols(f"log_cites2 ~ {rhs} + log_size | ym + country", data=df, vcov={"CRV1": "venue"}))
show("PPML full, log(size) regressor", pf.fepois(f"cites2 ~ {rhs} + log_size | ym + country + venue", data=df, vcov={"CRV1": "venue"}))
