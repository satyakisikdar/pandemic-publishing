-- Item 14 step 1: affiliation coverage of the OW authorship slots (May-2025 snapshot).
-- Run from analysis:  ~/.local/bin/duckdb < item14_coverage.sql
SET memory_limit = '40GB'; SET threads = 8;
CREATE TEMP TABLE slots AS
SELECT DISTINCT work_id, author_id, author_type
FROM read_parquet('../data/covid_works_authors_ow.parquet');
CREATE TEMP TABLE aff AS
SELECT a.work_id, a.author_id,
       count(DISTINCT a.country_code) AS n_country,
       count(DISTINCT a.institution_id) AS n_inst
FROM read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_authorships/*.parquet') a
SEMI JOIN slots s ON s.work_id = a.work_id AND s.author_id = a.author_id
WHERE a.publication_year BETWEEN 2020 AND 2022 AND a.assigned_institution  -- direct affiliations only; unassigned rows are parent institutions in the lineage
GROUP BY ALL;
SELECT s.author_type, count(*) AS slots,
       round(100.0 * count(*) FILTER (WHERE n_country >= 1) / count(*), 1) AS pct_country,
       round(100.0 * count(*) FILTER (WHERE n_country > 1) / count(*), 1) AS pct_multi_country,
       round(100.0 * count(*) FILTER (WHERE n_inst >= 1) / count(*), 1) AS pct_inst,
       round(100.0 * count(*) FILTER (WHERE n_inst > 1) / count(*), 1) AS pct_multi_inst
FROM slots s LEFT JOIN aff f USING (work_id, author_id)
GROUP BY ROLLUP (s.author_type) ORDER BY 1;
-- works whose every slot has a country
SELECT count(*) AS works, count(*) FILTER (WHERE all_cov) AS works_all_slots_with_country,
       round(100.0 * count(*) FILTER (WHERE all_cov) / count(*), 1) AS pct
FROM (SELECT s.work_id, bool_and(coalesce(n_country, 0) >= 1) AS all_cov
      FROM slots s LEFT JOIN aff f USING (work_id, author_id) GROUP BY 1);
