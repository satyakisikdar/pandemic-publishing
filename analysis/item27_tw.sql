-- Item 27: sensitivity of the author classes to the 5-year training window (TW).
-- Run from analysis, in the background, logging:
--   ~/.local/bin/duckdb < item27_tw.sql > item27_tw.log 2>&1
-- Stage 1 rebuilds the current 2015-2019 classes from the raw May-2025 snapshot and must reproduce
-- covid_works_authors_ow exactly (102,794 / 744,440 / 493,133); stage 2 scans <= 2014 for the same OW authors.
-- Rules (notebook cells 8-9, 27, 42-52): works = 8 types, 1-30 authors, not retracted, not paratext;
-- outbreak-science work = one of the four outbreak concepts with score > 0.30; COVID-route experts use
-- 2015-2019 only (COVID concepts on older works are noise). Writes item27_types.parquet.
.timer on
.print '== start'
SELECT now();
SET memory_limit = '60GB'; SET threads = 12;
SET VARIABLE P = getenv('OPENALEX_SNAPSHOT');

CREATE TEMP TABLE ow AS
SELECT DISTINCT author_id, author_type FROM '../data/covid_works_authors_ow.parquet';
.print '== OW authors by type (expect 102794 / 744440 / 493133)'
SELECT author_type, count(*) FROM ow GROUP BY 1 ORDER BY 1;

-- eligible works <= 2019, general preprocessing rules
CREATE TEMP TABLE w AS
SELECT work_id, publication_year y FROM read_parquet(getvariable('P') || '/works/*.parquet')
WHERE publication_year <= 2019 AND num_authors BETWEEN 1 AND 30 AND NOT is_retracted AND NOT is_paratext
  AND type IN ('article', 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter',
               'preprint', 'review', 'conference-paper');
.print '== eligible works <= 2019'
SELECT count(*), min(y), max(y) FROM w;

-- every (OW author, eligible work <= 2019) pair
CREATE TEMP TABLE aw AS
SELECT DISTINCT a.author_id, a.work_id, w.y
FROM read_parquet(getvariable('P') || '/works_authorships/*.parquet') a
JOIN w USING (work_id) SEMI JOIN ow USING (author_id)
WHERE a.publication_year <= 2019;
.print '== OW-author x work pairs <= 2019'
SELECT count(*), count(DISTINCT author_id) FROM aw;

CREATE TEMP TABLE osw AS
SELECT DISTINCT work_id FROM read_parquet(getvariable('P') || '/works_concepts/*.parquet')
WHERE publication_year <= 2019 AND score > 0.30 AND concept_id IN (107130276, 89623803, 116675565, 524204448);
.print '== outbreak-science works <= 2019 (any eligibility)'
SELECT count(*) FROM osw;

-- COVID-route TW authors, exactly as the notebook: covid_works.parquet works before 2020
CREATE TEMP TABLE cov_tw AS
SELECT DISTINCT a.author_id FROM '../data/works_authors_2015_2022.parquet' a
JOIN '../data/covid_works.parquet' c USING (work_id) WHERE c.publication_year < 2020;

CREATE TEMP TABLE per AS
SELECT o.author_id, o.author_type,
       count(*) FILTER (WHERE aw.y BETWEEN 2015 AND 2019) any_15_19,
       count(*) FILTER (WHERE aw.y BETWEEN 2010 AND 2014) any_10_14,
       count(*) FILTER (WHERE aw.y <= 2014) any_le14,
       count(*) FILTER (WHERE aw.y BETWEEN 2015 AND 2019 AND osw.work_id IS NOT NULL) os_15_19,
       count(*) FILTER (WHERE aw.y BETWEEN 2010 AND 2014 AND osw.work_id IS NOT NULL) os_10_14,
       count(*) FILTER (WHERE aw.y <= 2014 AND osw.work_id IS NOT NULL) os_le14,
       max(aw.y) FILTER (WHERE aw.y <= 2014 AND osw.work_id IS NOT NULL) last_os_le14,
       bool_or(c.author_id IS NOT NULL) cov
FROM ow o LEFT JOIN aw USING (author_id) LEFT JOIN osw USING (work_id) LEFT JOIN cov_tw c ON c.author_id = o.author_id
GROUP BY 1, 2;

CREATE TEMP TABLE types AS
SELECT *,
  CASE WHEN os_15_19 > 0 OR cov THEN 'expert' WHEN any_15_19 > 0 THEN 'bellwether' ELSE 'newcomer' END t_rebuilt,
  CASE WHEN os_15_19 + os_10_14 > 0 OR cov THEN 'expert' WHEN any_15_19 + any_10_14 > 0 THEN 'bellwether' ELSE 'newcomer' END t_10,
  CASE WHEN os_15_19 + os_le14 > 0 OR cov THEN 'expert' WHEN any_15_19 + any_le14 > 0 THEN 'bellwether' ELSE 'newcomer' END t_full
FROM per;

.print '== CHECK: rebuilt 2015-2019 classes vs covid_works_authors_ow (all off-diagonal counts must be 0)'
SELECT author_type, t_rebuilt, count(*) FROM types GROUP BY ALL ORDER BY 1, 2;
.print '== transitions, 10-year window (2010-2019)'
SELECT author_type, t_10, count(*) FROM types GROUP BY ALL ORDER BY 1, 2;
.print '== transitions, full history (<= 2019)'
SELECT author_type, t_full, count(*) FROM types GROUP BY ALL ORDER BY 1, 2;
.print '== promoted to expert (full history): year of last pre-2015 outbreak-science paper'
SELECT CASE WHEN last_os_le14 >= 2010 THEN cast(last_os_le14 AS varchar) WHEN last_os_le14 >= 2000 THEN '2000-2009'
            ELSE 'before 2000' END last_os, count(*)
FROM types WHERE author_type <> 'expert' AND t_full = 'expert' GROUP BY 1 ORDER BY 1;

COPY (SELECT author_id, author_type t_base, t_rebuilt, t_10, t_full, last_os_le14 FROM types)
  TO 'item27_types.parquet' (FORMAT parquet);
.print '== done'
SELECT now();
