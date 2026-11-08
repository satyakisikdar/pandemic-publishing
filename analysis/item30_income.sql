-- Item 30: contribution and visibility of authors from low- and middle-income countries.
-- Run from analysis:  ~/.local/bin/duckdb < item30_income.sql
-- Income groups: World Bank FY21 (wb_income_fy21.csv). Not listed by the World Bank, set to high income:
-- Taiwan (TW) and the French overseas departments (GP, GF, MQ, RE, as France). Other unlisted codes -> 'unclassified'.
-- Paper country = modal country of the paper's direct affiliations (fe_model.parquet, built in append_rev_cells.py;
-- ties alphabetical; 'unknown' when no author has an affiliation country).
-- Slot country = item14_slots.csv (alphabetically first direct-affiliation country; XX = none).
SET threads = 8;
create temp table wb as
  select iso2, fy21 from read_csv('wb_income_fy21.csv', skip = 2, header = true) where fy21 is not null
  union all select * from (values ('TW','H'),('GP','H'),('GF','H'),('MQ','H'),('RE','H')) t(iso2, fy21);
create temp macro grp(code) as case code when 'H' then 'High' when 'UM' then 'Upper-middle'
  when 'LM' then 'Lower-middle' when 'L' then 'Low' else code end;

create temp table p as
  select f.*, case when f.country = 'unknown' then 'Unknown' else coalesce(grp(wb.fy21), 'Unclassified') end inc
  from 'fe_model.parquet' f left join wb on wb.iso2 = f.country;

-- 1. papers by income group of the team's modal country
select inc, count(*) papers, round(100.0 * count(*) / sum(count(*)) over (), 1) pct_papers,
       round(100.0 * avg((class_ = '0%')::int), 1) pct_no_os,
       round(avg(cites2), 2) mean_cites2, median(cites2) med_cites2,
       round(median(impact_factor) filter (where has_if), 2) med_if
from p group by 1 order by papers desc;

-- 2. authorship slots by income group of the slot's country, with the outbreak-scientist share
create temp table s as
  select sl.t, case when c.cc = 'XX' then 'Unknown' else coalesce(grp(wb.fy21), 'Unclassified') end inc
  from read_csv('item14_slots.csv') sl join read_csv('item14_countries.csv') c using (c)
  left join wb on wb.iso2 = c.cc;
select inc, count(*) slots, round(100.0 * count(*) / sum(count(*)) over (), 1) pct_slots,
       round(100.0 * avg((t = 0)::int), 1) pct_os_slots, round(100.0 * avg((t = 2)::int), 1) pct_newcomer_slots
from s group by 1 order by slots desc;

-- 3. visibility: mean two-year citations of high-income vs low/middle-income teams, same composition and year
copy (
select class_, y,
       count(*) filter (where inc = 'High') n_hic,
       count(*) filter (where inc in ('Upper-middle', 'Lower-middle', 'Low')) n_lmic,
       avg(cites2) filter (where inc = 'High') mean_hic,
       avg(cites2) filter (where inc in ('Upper-middle', 'Lower-middle', 'Low')) mean_lmic,
       mean_lmic / mean_hic ratio,
       median(cites2) filter (where inc = 'High') med_hic,
       median(cites2) filter (where inc in ('Upper-middle', 'Lower-middle', 'Low')) med_lmic,
       median(impact_factor) filter (where has_if and inc = 'High') if_hic,
       median(impact_factor) filter (where has_if and inc in ('Upper-middle', 'Lower-middle', 'Low')) if_lmic
from p group by all order by class_, y
) to 'item30_cites_by_income.csv' (header);
select class_, y, n_hic, n_lmic, round(mean_hic, 1) mean_hic, round(mean_lmic, 1) mean_lmic, round(ratio, 2) ratio,
       med_hic, med_lmic, round(if_hic, 2) if_hic, round(if_lmic, 2) if_lmic
from read_csv('item30_cites_by_income.csv');
-- pooled: LMIC citations relative to what the same papers would get at the HIC mean of their class x year
select round(sum(n_lmic * mean_lmic) / sum(n_lmic * mean_hic), 3) pooled_ratio from read_csv('item30_cites_by_income.csv');
