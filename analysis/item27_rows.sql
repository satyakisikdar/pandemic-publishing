-- Item 27: the two Appendix I rows not covered by variantA_report.sql, for each training window:
-- (1) top-20 journals shared by No-OS and Only-OS teams (same rule as append_item17_cells.py);
-- (2) inputs for the FE regressions (fe_model_item27_<V>.parquet: class_ and frac_exp relabelled), run with fe_models.py.
-- Run from analysis:  ~/.local/bin/duckdb < item27_rows.sql > item27_rows.log 2>&1
SET threads = 8;
SET VARIABLE DATA = '../data';
.print '############ VARIANT t_base'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, ty.t_base t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT w.work_id, coalesce(s.ne,0) ne_cls,
  CASE WHEN coalesce(s.ne,0)=0 THEN '0%' WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.5 THEN '1-50%'
       WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.999 THEN '51-99%' ELSE '100%' END cls
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE j AS SELECT f.work_id, v.cls, s.source_name FROM 'fig4_reasons.parquet' f
  JOIN wc v USING (work_id) JOIN read_parquet(getvariable('DATA')||'/covid_journal_sources.parquet') s USING (source_id) WHERE f.has_if;
CREATE OR REPLACE TEMP TABLE t20 AS SELECT * FROM (SELECT *, rank() OVER (PARTITION BY cls ORDER BY n DESC) r
  FROM (SELECT cls, source_name, count(*) n FROM j GROUP BY ALL)) WHERE r <= 20;
SELECT 't_base' v, (SELECT count(*) FROM t20 a JOIN t20 b USING (source_name) WHERE a.cls='0%' AND b.cls='100%') shared_no_only;
COPY (SELECT m.* REPLACE (v.cls AS class_, 1.0*v.ne_cls/m.num_authors AS frac_exp)
      FROM 'fe_model.parquet' m JOIN wc v USING (work_id)) TO 'fe_model_item27_t_base.parquet';
.print '############ VARIANT t_10'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, ty.t_10 t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT w.work_id, coalesce(s.ne,0) ne_cls,
  CASE WHEN coalesce(s.ne,0)=0 THEN '0%' WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.5 THEN '1-50%'
       WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.999 THEN '51-99%' ELSE '100%' END cls
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE j AS SELECT f.work_id, v.cls, s.source_name FROM 'fig4_reasons.parquet' f
  JOIN wc v USING (work_id) JOIN read_parquet(getvariable('DATA')||'/covid_journal_sources.parquet') s USING (source_id) WHERE f.has_if;
CREATE OR REPLACE TEMP TABLE t20 AS SELECT * FROM (SELECT *, rank() OVER (PARTITION BY cls ORDER BY n DESC) r
  FROM (SELECT cls, source_name, count(*) n FROM j GROUP BY ALL)) WHERE r <= 20;
SELECT 't_10' v, (SELECT count(*) FROM t20 a JOIN t20 b USING (source_name) WHERE a.cls='0%' AND b.cls='100%') shared_no_only;
COPY (SELECT m.* REPLACE (v.cls AS class_, 1.0*v.ne_cls/m.num_authors AS frac_exp)
      FROM 'fe_model.parquet' m JOIN wc v USING (work_id)) TO 'fe_model_item27_t_10.parquet';
.print '############ VARIANT t_full'
CREATE OR REPLACE TEMP TABLE auth AS SELECT a.work_id, a.author_id, ty.t_full t
  FROM read_parquet(getvariable('DATA')||'/covid_works_authors_ow.parquet') a JOIN 'item27_types.parquet' ty USING (author_id);
CREATE OR REPLACE TEMP TABLE wc AS SELECT w.work_id, coalesce(s.ne,0) ne_cls,
  CASE WHEN coalesce(s.ne,0)=0 THEN '0%' WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.5 THEN '1-50%'
       WHEN 1.0*s.ne/(w.num_experts+w.num_bellwethers+w.num_newcomers)<=0.999 THEN '51-99%' ELSE '100%' END cls
  FROM read_parquet(getvariable('DATA')||'/covid_works_ow.parquet') w
  LEFT JOIN (SELECT work_id, count(*) FILTER (WHERE t='expert') ne FROM auth GROUP BY 1) s USING (work_id);
CREATE OR REPLACE TEMP TABLE j AS SELECT f.work_id, v.cls, s.source_name FROM 'fig4_reasons.parquet' f
  JOIN wc v USING (work_id) JOIN read_parquet(getvariable('DATA')||'/covid_journal_sources.parquet') s USING (source_id) WHERE f.has_if;
CREATE OR REPLACE TEMP TABLE t20 AS SELECT * FROM (SELECT *, rank() OVER (PARTITION BY cls ORDER BY n DESC) r
  FROM (SELECT cls, source_name, count(*) n FROM j GROUP BY ALL)) WHERE r <= 20;
SELECT 't_full' v, (SELECT count(*) FROM t20 a JOIN t20 b USING (source_name) WHERE a.cls='0%' AND b.cls='100%') shared_no_only;
COPY (SELECT m.* REPLACE (v.cls AS class_, 1.0*v.ne_cls/m.num_authors AS frac_exp)
      FROM 'fe_model.parquet' m JOIN wc v USING (work_id)) TO 'fe_model_item27_t_full.parquet';
