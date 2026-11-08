# pandemic-publishing

Code for "Pandemic publishing fortified knowledge silos through expert dilution in COVID-19 science"
(Sikdar, Krishna, Venturini, Charpignon, Kumar, Rinaldi, Tudisco, Fortunato, Majumder). It extends our
earlier comment "What we should learn from pandemic publishing" (Nature Human Behaviour, 2024),
whose code is archived at https://doi.org/10.5281/zenodo.10368779. We classify the authors of COVID-19 papers published in 2020-2022 as outbreak
scientists, bellwethers (established researchers from other fields) or newcomers, and compare team
composition, venues, citations and collaboration across these groups. All data come from the May 2025
OpenAlex snapshot.

Data: https://doi.org/10.5281/zenodo.24047334 (Zenodo, CC BY 4.0). Code: MIT.

## Contents

- `notebook/first-principles.ipynb`: main analysis. Preprocessing from the snapshot, author and team
  classification, Figs 1-7 and the appendix tables. The last cells hold the queries added during revision.
  Outputs are stripped.
- `notebook/notebook_utils.py`: helper functions for the notebook.
- `analysis/`: revision analyses as duckdb SQL and Python scripts (see the table below).
- `figures/`: pgfplots sources for every figure in the paper, and the venue tables they read.
- `data/`: empty here. Unpack the Zenodo `data/` folder into it.

The `item*` prefixes in `analysis/` are the revision items that prompted each script.

## Setup

Python 3.10 or later with the packages in `requirements.txt`. We ran the notebook in JupyterLab, not as a
script (`read_parquet` in `notebook_utils.py` calls `display()`), and the regressions
with `uv run --with pyfixest --with pandas --with pyarrow python <script>`. The SQL scripts need the duckdb
CLI. The figures need pdflatex with pgfplots, pgfplotstable, booktabs and opencolor.

Environment variables:

```bash
export OPENALEX_SNAPSHOT=/path/to/openalex/may-2025/parquet   # works/, works_authorships/, works_concepts/, ...
export OPENALEX_CSV=/path/to/openalex/may-2025/csv            # concepts.csv.gz, topics.csv.gz, sources*.csv.gz
export OPENALEX_MAILTO=you@example.org                         # used in OpenAlex API calls
```

`OPENALEX_CSV` can point at `data/openalex_csv/` from the Zenodo deposit.

## Reproducing the results

Unpack the deposit's `data/` into `data/` and its `analysis/` files into `analysis/`, then run the scripts
from inside `analysis/`. Every script marked "deposit" below runs without the snapshot.

In the notebook, run the Setup section and the first code cell of Preprocessing (it defines `work_types`),
then skip the two sections marked "needs the snapshot" (Preprocessing and Citation counts) and the cells that
read `parq_path` in the Fig 4 section and the revision analyses. Those need the full snapshot and about 60 GB
of RAM. Everything else reads the deposited parquets; `covid_works_ow.parquet` already holds the author
classes and the citation counts.

Scripts marked "snapshot" read the full OpenAlex snapshot. OpenAlex does not keep old snapshots, so the May
2025 release may no longer be available, and a newer one gives slightly different numbers. Their outputs are
in the deposit, so later steps still run.

Two inputs cannot be rebuilt from the snapshot and are in the deposit. `data/covid_journal_sources.parquet`
holds each journal's two-year mean citedness from the OpenAlex API, retrieved in September 2026 (the snapshot
has no journal statistics, and the API reports current values only). Publication dates of 1 January are
OpenAlex placeholders that the notebook randomizes within the year with seed 0, so the deposited
`works_2015_2022.parquet` is the reference copy of the dates.

| Script | Input | What it does |
|---|---|---|
| `fig7_export.sql` | deposit | Co-authorship multigraph: `fig7_slots.csv`, `fig7_edges.csv` |
| `fig7_robust.py` | deposit | Fig 7 within team-size bands, paper-weighted and by year, with author-bootstrap CIs. Checks the published Fig 7. |
| `make_fig7_robust_fig.py` | deposit | Writes `figures/figS4-collab-robustness.tex` |
| `fe_models.py`, `fe_robust.py` | deposit | Citation regressions with team-size, month, country and venue fixed effects (Appendix H) |
| `make_strat_figs.py` | deposit | Writes `figures/figS2` and `figS3` |
| `variantA.sql`, `variantA_report.sql` | deposit | Outbreak scientists defined without epidemiology (Appendix I) |
| `item14_export.sql`, `item14_coverage.sql` | snapshot | Affiliation country and institution for every authorship |
| `item14_country_null.py` | deposit | Mixing nulls that keep each author's country or institution (Appendix J). Checks the main null and runs a shuffle test. |
| `item19_topics.sql` | snapshot | Topic of each paper from its level-0 and level-1 concepts |
| `item19_fe.py` | deposit | Regressions with topic fixed effects. Checks the Appendix H estimates. |
| `item20_top_venues.sql` | snapshot | COVID-19 papers in six general journals by year, and their page lengths |
| `item27_tw.sql` | snapshot | Author classes with a 10-year or full-history training window. Checks the current classes. |
| `item27_report.sql`, `item27_rows.sql` | deposit | Main results under each training window (Appendix I) |
| `item30_income.sql` | deposit | Papers, authorships and citations by World Bank income group |
| `retention_2023_24.sql` | snapshot | What the 2020-2022 authors published in 2023-2024 (Appendix K) |

## Expected numbers

The scripts check their own outputs. With the deposit you should get 453,367 COVID-19 works in 2020-2022
(119,534, 176,113 and 157,720 by year); 102,794 outbreak scientists, 744,440 bellwethers and 493,133
newcomers; 10,171,294 co-authorship edges, of which 7.46% join two outbreak scientists against 2.77%
expected; and a citation premium for majority-outbreak-scientist teams over teams without one of +93.6%
(size, month and country fixed effects) and +19.6% (adding venue).

## License

Code under MIT (`LICENSE`). Deposited data under CC BY 4.0 (`DATA_LICENSE.md`). OpenAlex data are CC0.
