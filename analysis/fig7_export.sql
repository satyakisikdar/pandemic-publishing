-- Fig 7 vertex-bootstrap inputs (items 12, 13, 15). Run from analysis:
--   ~/.local/bin/duckdb < fig7_export.sql
create temp table a as
  select work_id, author_id, publication_year yr,
         case author_type when 'expert' then 0 when 'bellwether' then 1 else 2 end t
  from '../data/covid_works_authors_ow.parquet';
create temp table w as select work_id, count(*) k from a group by 1;
create temp table ids as select author_id, (row_number() over (order by author_id)) - 1 i, any_value(t) t from a group by 1;
-- one row per authorship slot
copy (select ids.i, ids.t, a.yr, w.k from a join ids using (author_id) join w using (work_id))
  to 'fig7_slots.csv' (header);
-- one row per co-author pair per shared work (multi-edges), with the work's year and classified team size
copy (select x.i u, y.i v, ax.yr, w.k
      from a ax join a ay on ax.work_id = ay.work_id and ax.author_id < ay.author_id
      join ids x on x.author_id = ax.author_id join ids y on y.author_id = ay.author_id
      join w on w.work_id = ax.work_id)
  to 'fig7_edges.csv' (header);
