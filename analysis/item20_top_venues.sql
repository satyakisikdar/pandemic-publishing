-- Item 20: why placement in top venues concentrated in 2020; tests her commentary hypothesis.
-- Run from analysis:  ~/.local/bin/duckdb < item20_top_venues.sql
-- Named journals: Lancet, NEJM, JAMA, BMJ (both OpenAlex ids), Nature, Science. Page length from works_biblio
-- (numeric first/last page only); <= 2 pages is a proxy for correspondence/commentary typed as 'article'.
SET threads = 8;
create temp table named as select * from (values (49861241,'Lancet'),(62468778,'NEJM'),(172573765,'JAMA'),
  (192814187,'BMJ'),(4393917726,'BMJ'),(137773608,'Nature'),(3880285,'Science')) t(source_id, j);
create temp table p as select f.*, n.j from 'fe_model.parquet' f join named n on cast(n.source_id as varchar) = f.venue;
create temp table b as select work_id, try_cast(first_page as int) fp, try_cast(last_page as int) lp
  from read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_biblio/*.parquet')
  where work_id in (select work_id from p);
create temp table q as select p.*, case when b.fp is not null and b.lp >= b.fp then b.lp - b.fp + 1 end pages
  from p left join b using (work_id);
select y, count(*) named_papers, round(100.0 * count(*) / any_value(tot), 2) pct_of_all,
  count(pages) with_pages, round(100.0 * avg((pages <= 2)::int) filter (where pages is not null), 1) pct_le2pages,
  round(median(pages), 1) med_pages, round(100.0 * avg((wtype = 'review')::int), 1) pct_review
from q join (select y, count(*) tot from 'fe_model.parquet' group by 1) t using (y) group by 1 order by 1;
select class_, y, count(*) n, round(100.0 * avg((pages <= 2)::int) filter (where pages is not null), 1) pct_le2pages
from q group by all order by 1, 2;
select j, y, count(*) n from q group by all order by 1, 2;
-- share of each composition's output in the six journals, and page-number coverage by journal
select class_, y, count(*) filter (where n.j is not null) n6, count(*) n_all,
  round(100.0 * n6 / n_all, 2) pct_in_6
from 'fe_model.parquet' f left join named n on cast(n.source_id as varchar) = f.venue group by all order by 1, 2;
select j, count(*) n, count(pages) with_pages from q group by 1 order by 1;
