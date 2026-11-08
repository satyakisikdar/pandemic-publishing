-- Item 26 / Discussion: did the 2020-2022 COVID-19 authors keep publishing on COVID-19 in 2023-2024?
-- Same COVID-19 slice rules as the OW (notebook cells 8-9, 27, 42-52): COVID core concept score > 0.30,
-- 8 work types, 2-30 authors, DOI present, not retracted, not paratext. May-2025 snapshot.
-- Run from analysis:  ~/.local/bin/duckdb < retention_2023_24.sql
SET memory_limit = '40GB';
SET threads = 8;

CREATE TEMP TABLE covid_w AS
SELECT DISTINCT w.work_id, w.publication_year AS yr
FROM read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works/*.parquet') w
JOIN read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_concepts/*.parquet') c
  ON c.work_id = w.work_id
WHERE w.publication_year BETWEEN 2020 AND 2024
  AND c.publication_year BETWEEN 2020 AND 2024
  AND c.concept_id IN (3008058167, 3006700255, 3007834351) AND c.score > 0.30
  AND w.type IN ('article', 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter',
                 'preprint', 'review', 'conference-paper')
  AND w.num_authors BETWEEN 2 AND 30 AND w.doi IS NOT NULL
  AND NOT w.is_retracted AND NOT w.is_paratext;

-- context: COVID-19 works per year under these rules (2020-2022 should be close to the OW's 453,367 total)
SELECT yr, count(*) AS covid_works FROM covid_w GROUP BY yr ORDER BY yr;

CREATE TEMP TABLE late_authors AS
SELECT DISTINCT a.author_id, w.yr
FROM read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_authorships/*.parquet') a
JOIN covid_w w ON w.work_id = a.work_id
WHERE w.yr IN (2023, 2024) AND a.publication_year IN (2023, 2024);

-- any 2023-2024 work at all, under the general preprocessing rules (8 types, 1-30 authors, not retracted/paratext; no DOI rule)
CREATE TEMP TABLE any_late AS
SELECT DISTINCT a.author_id
FROM read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_authorships/*.parquet') a
JOIN read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works/*.parquet') w
  ON w.work_id = a.work_id
WHERE a.publication_year IN (2023, 2024) AND w.publication_year IN (2023, 2024)
  AND w.type IN ('article', 'journal-article', 'proceedings-article', 'posted-content', 'book-chapter',
                 'preprint', 'review', 'conference-paper')
  AND w.num_authors BETWEEN 1 AND 30 AND NOT w.is_retracted AND NOT w.is_paratext;

CREATE TEMP TABLE ow AS
SELECT DISTINCT author_id, author_type
FROM read_parquet('../data/covid_works_authors_ow.parquet');

-- share of each OW author type with >= 1 COVID-19 work in 2023, in 2024, in either year, and with any work at all in 2023-2024
COPY (
SELECT o.author_type,
       count(*) AS ow_authors,
       count(*) FILTER (WHERE l23.author_id IS NOT NULL) AS in_2023,
       count(*) FILTER (WHERE l24.author_id IS NOT NULL) AS in_2024,
       count(*) FILTER (WHERE l23.author_id IS NOT NULL OR l24.author_id IS NOT NULL) AS in_2023_24,
       round(100.0 * in_2023 / ow_authors, 1) AS pct_2023,
       round(100.0 * in_2024 / ow_authors, 1) AS pct_2024,
       round(100.0 * in_2023_24 / ow_authors, 1) AS pct_2023_24,
       count(*) FILTER (WHERE x.author_id IS NOT NULL) AS any_work_2023_24,
       round(100.0 * any_work_2023_24 / ow_authors, 1) AS pct_any_work_2023_24,
       round(100.0 - pct_any_work_2023_24, 1) AS pct_no_work_2023_24
FROM ow o
LEFT JOIN any_late x USING (author_id)
LEFT JOIN (SELECT author_id FROM late_authors WHERE yr = 2023) l23 USING (author_id)
LEFT JOIN (SELECT author_id FROM late_authors WHERE yr = 2024) l24 USING (author_id)
GROUP BY ALL ORDER BY o.author_type
) TO 'retention_2023_24.csv' (HEADER);

SELECT * FROM read_csv('retention_2023_24.csv');
