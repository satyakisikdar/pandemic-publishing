-- Item 14 (null model preserving country / institution). Run from analysis:
--   ~/.local/bin/duckdb < item14_export.sql
-- Adds a country (and institution) to every OW authorship slot and to both ends of every co-author edge.
-- Author index i and type t match fig7_export.sql (dense index ordered by author_id).
-- Country/institution come from the slot's direct affiliations (assigned_institution) in the May-2025
-- snapshot; with several, the alphabetically first country / smallest institution id is kept (2.9% of slots
-- have more than one country). Slots with no country get code 'XX' and form their own stratum.
SET memory_limit = '40GB'; SET threads = 8;
create temp table a as
  select work_id, author_id, publication_year yr,
         case author_type when 'expert' then 0 when 'bellwether' then 1 else 2 end t
  from '../data/covid_works_authors_ow.parquet';
create temp table ids as select author_id, (row_number() over (order by author_id)) - 1 i from a group by 1;
create temp table wid as select work_id, (row_number() over (order by work_id)) - 1 w from a group by 1;
create temp table aff as
  select work_id, author_id, min(country_code) cc, min(institution_id) inst,
         max(country_code) cc_max, max(institution_id) inst_max  -- sensitivity to the tie rule
  from read_parquet(getenv('OPENALEX_SNAPSHOT') || '/works_authorships/*.parquet') x
  semi join a using (work_id, author_id)
  where x.publication_year between 2020 and 2022 and x.assigned_institution
  group by all;
create temp table s as
  select a.work_id, a.author_id, ids.i, a.t, wid.w, coalesce(aff.cc, 'XX') cc, aff.inst, coalesce(aff.cc_max, 'XX') cc_max, aff.inst_max
  from a join ids using (author_id) join wid using (work_id) left join aff using (work_id, author_id);
create temp table ccode as select cc, (row_number() over (order by cc)) - 1 c from (select distinct cc from s union select distinct cc_max from s);
copy (select s.i, s.t, s.w, ccode.c, s.inst, cm.c c_max, s.inst_max from s join ccode using (cc)
      join ccode cm on cm.cc = s.cc_max) to 'item14_slots.csv' (header);
copy (select ccode.c, ccode.cc from ccode order by c) to 'item14_countries.csv' (header);
copy (select x.i u, y.i v, x.w, cx.c cu, cy.c cv
      from s x join s y on x.work_id = y.work_id and x.author_id < y.author_id
      join ccode cx on cx.cc = x.cc join ccode cy on cy.cc = y.cc)
  to 'item14_edges.csv' (header);
select count(*) slots, count(inst) slots_with_inst, count(DISTINCT inst) institutions from s;
