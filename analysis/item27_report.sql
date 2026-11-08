-- Item 27: headline results under alternative training windows, via variantA_report.sql.
-- Run from analysis:  ~/.local/bin/duckdb < item27_report.sql > item27_report.log 2>&1
-- Variant column of item27_types.parquet: t_base (current), t_10 (2010-2019), t_full (<= 2019).
-- Work classes are recomputed from per-work counts of the new labels; the class denominator stays the
-- number of classified authors (unchanged by relabeling), as in the notebook and variantA.sql.
SET threads = 8;
SET VARIABLE DATA = '../data';
.print '############ VARIANT t_base'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, a.publication_year, ty.t_base t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wk AS SELECT w.work_id, w.publication_year y, w.num_authors, w.num_cites_2yrs c2, w.type wtype,
  coalesce(s.ne,0) ne_cls, w.num_experts + w.num_bellwethers + w.num_newcomers n_cls,
  coalesce(s.ne,0) ne, coalesce(s.nb,0) nb, coalesce(s.nn,0) nn
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne, count(*) FILTER (WHERE t='bellwether') nb,
             count(*) FILTER (WHERE t='newcomer') nn FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT *, CASE WHEN ne_cls=0 THEN '0%' WHEN 1.0*ne_cls/n_cls<=0.5 THEN '1-50%'
  WHEN 1.0*ne_cls/n_cls<=0.999 THEN '51-99%' ELSE '100%' END cls FROM wk;
.read variantA_report.sql
.print '############ VARIANT t_10'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, a.publication_year, ty.t_10 t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wk AS SELECT w.work_id, w.publication_year y, w.num_authors, w.num_cites_2yrs c2, w.type wtype,
  coalesce(s.ne,0) ne_cls, w.num_experts + w.num_bellwethers + w.num_newcomers n_cls,
  coalesce(s.ne,0) ne, coalesce(s.nb,0) nb, coalesce(s.nn,0) nn
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne, count(*) FILTER (WHERE t='bellwether') nb,
             count(*) FILTER (WHERE t='newcomer') nn FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT *, CASE WHEN ne_cls=0 THEN '0%' WHEN 1.0*ne_cls/n_cls<=0.5 THEN '1-50%'
  WHEN 1.0*ne_cls/n_cls<=0.999 THEN '51-99%' ELSE '100%' END cls FROM wk;
.read variantA_report.sql
.print '############ VARIANT t_full'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, a.publication_year, ty.t_full t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wk AS SELECT w.work_id, w.publication_year y, w.num_authors, w.num_cites_2yrs c2, w.type wtype,
  coalesce(s.ne,0) ne_cls, w.num_experts + w.num_bellwethers + w.num_newcomers n_cls,
  coalesce(s.ne,0) ne, coalesce(s.nb,0) nb, coalesce(s.nn,0) nn
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne, count(*) FILTER (WHERE t='bellwether') nb,
             count(*) FILTER (WHERE t='newcomer') nn FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT *, CASE WHEN ne_cls=0 THEN '0%' WHEN 1.0*ne_cls/n_cls<=0.5 THEN '1-50%'
  WHEN 1.0*ne_cls/n_cls<=0.999 THEN '51-99%' ELSE '100%' END cls FROM wk;
.read variantA_report.sql
