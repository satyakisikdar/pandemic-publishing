-- Item 19: does the outbreak-science gradient hold within research topics?
-- Run from analysis:  ~/.local/bin/duckdb < item19_topics.sql > item19_topics.log 2>&1
-- Topic = highest-scoring level-0 OpenAlex concept of the work (all OW works have one); finer FE = highest-scoring
-- level-1 concept. Domains group the level-0 concepts. Writes fe_model_item19.parquet (fe_model + l0, l1, tdomain).
.timer on
.print '== start'
SELECT now();
SET threads = 8; SET memory_limit = '40GB';
CREATE TEMP TABLE c AS
SELECT work_id, level, concept_name, score
FROM read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_concepts/*.parquet')
WHERE publication_year BETWEEN 2020 AND 2022 AND level IN (0, 1) AND work_id IN (SELECT work_id FROM 'fe_model.parquet');
CREATE TEMP TABLE lab AS
SELECT work_id,
  arg_max(concept_name, score) FILTER (WHERE level = 0) l0, max(score) FILTER (WHERE level = 0) s0,
  arg_max(concept_name, score) FILTER (WHERE level = 1) l1,
  count(*) FILTER (WHERE level = 0 AND score > 0.30) n_l0_gt030,
  bool_or(level = 0 AND score > 0.30 AND concept_name IN ('Psychology','Business','Political science','Geography','Economics',
          'Sociology','Philosophy','Art','History')) has_social
FROM c GROUP BY 1;
CREATE TEMP TABLE m AS
SELECT f.*, coalesce(l.l0, 'none') l0, coalesce(l.l1, 'none') l1, l.s0, l.n_l0_gt030, l.has_social,
  CASE WHEN l.l0 = 'Medicine' THEN 'Medicine' WHEN l.l0 = 'Biology' THEN 'Biology'
       WHEN l.l0 IN ('Psychology','Business','Political science','Geography','Economics','Sociology','Philosophy','Art','History')
         THEN 'Social sciences and humanities'
       ELSE 'Physical and computational sciences' END tdomain
FROM 'fe_model.parquet' f LEFT JOIN lab l USING (work_id);

.print '== CHECK: every work labeled (expect 453367, 0 none)'
SELECT count(*), count(*) FILTER (WHERE l0 = 'none') none_l0, count(*) FILTER (WHERE l1 = 'none') none_l1,
       round(100.0 * avg((s0 > 0.30)::int), 1) pct_s0_gt030 FROM m;
.print '== multi-label: share with a second level-0 concept > 0.30; Medicine-labeled with a social-science level-0 > 0.30'
SELECT round(100.0 * avg((n_l0_gt030 >= 2)::int), 1) pct_multi,
       round(100.0 * avg(has_social::int) FILTER (WHERE tdomain = 'Medicine'), 1) pct_med_with_social FROM m;
.print '== domain x class: papers; class within domain (row %)'
SELECT tdomain, count(*) n, round(100.0 * count(*) / sum(count(*)) over (), 1) pct_all,
  round(100.0 * avg((class_ = '0%')::int), 1) no_os, round(100.0 * avg((class_ = '1-50%')::int), 1) minority,
  round(100.0 * avg((class_ = '51-99%')::int), 1) majority, round(100.0 * avg((class_ = '100%')::int), 1) only_os
FROM m GROUP BY 1 ORDER BY n DESC;
.print '== domain within class (column %): where No-OS vs Majority-OS papers sit'
SELECT class_, tdomain, round(100.0 * n / sum(n) over (partition by class_), 1) pct
FROM (SELECT class_, tdomain, count(*) n FROM m GROUP BY ALL) ORDER BY 1, 3 DESC;
.print '== within-domain gradients: median journal citedness (has_if) and 2020 citations, by class'
SELECT tdomain, class_, count(*) n,
  round(median(impact_factor) FILTER (WHERE has_if), 2) med_cited, count(*) FILTER (WHERE has_if) n_if,
  round(avg(cites2) FILTER (WHERE y = 2020), 1) mean_c2020, median(cites2) FILTER (WHERE y = 2020) med_c2020,
  count(*) FILTER (WHERE y = 2020) n2020
FROM m GROUP BY ALL ORDER BY 1, 2;
.print '== Psychology alone'
SELECT class_, count(*) n, round(median(impact_factor) FILTER (WHERE has_if), 2) med_cited,
  round(avg(cites2) FILTER (WHERE y = 2020), 1) mean_c2020, median(cites2) FILTER (WHERE y = 2020) med_c2020
FROM m WHERE l0 = 'Psychology' GROUP BY 1 ORDER BY 1;
COPY m TO 'fe_model_item19.parquet' (FORMAT parquet);
.print '== done'
SELECT now();
