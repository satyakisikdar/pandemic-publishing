-- Item 17, Variant A: drop Epidemiology as an outbreak-science concept; demoted experts become bellwethers.
-- Run with V = 'base' (current labels) or 'A'. Expects DATA (data/ dir) and the analysis dir as cwd.
CREATE OR REPLACE TEMP TABLE demoted AS SELECT author_id FROM 'expert_tags.parquet' WHERE NOT (other OR cov);
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, a.publication_year,
  CASE WHEN getvariable('V')='A' AND a.author_type='expert' AND a.author_id IN (SELECT author_id FROM demoted) THEN 'bellwether' ELSE a.author_type END t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a;
CREATE OR REPLACE TEMP TABLE wk AS SELECT w.work_id, w.publication_year y, w.num_authors, w.num_cites_2yrs c2, w.type wtype,
  w.num_experts - CASE WHEN getvariable('V')='A' THEN coalesce(d.nd,0) ELSE 0 END ne_cls,
  w.num_experts + w.num_bellwethers + w.num_newcomers n_cls,   -- notebook's class denominator (classified authors), unchanged by reclassification
  coalesce(s.ne,0) ne, coalesce(s.nb,0) nb, coalesce(s.nn,0) nn
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne, count(*) FILTER (WHERE t='bellwether') nb, count(*) FILTER (WHERE t='newcomer') nn FROM auth GROUP BY 1) s USING(work_id)
  LEFT JOIN (SELECT work_id, count(*) nd FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') WHERE author_type='expert' AND author_id IN (SELECT author_id FROM demoted) GROUP BY 1) d USING(work_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT *, CASE WHEN ne_cls=0 THEN '0%' WHEN 1.0*ne_cls/n_cls<=0.5 THEN '1-50%' WHEN 1.0*ne_cls/n_cls<=0.999 THEN '51-99%' ELSE '100%' END cls FROM wk;
