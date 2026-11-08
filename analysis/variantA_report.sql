-- Fig 2A: author-type shares per year
SELECT 'F2A' f, y, max(n) FILTER (WHERE t='expert') experts, round(100.0*max(n) FILTER (WHERE t='expert')/sum(n),1) pct_exp,
       round(100.0*max(n) FILTER (WHERE t='bellwether')/sum(n),1) pct_bell, round(100.0*max(n) FILTER (WHERE t='newcomer')/sum(n),1) pct_new
FROM (SELECT publication_year y, t, count(DISTINCT author_id) n FROM auth GROUP BY ALL) GROUP BY 1,2 ORDER BY 2;
SELECT 'experts_total' f, count(DISTINCT author_id) FROM auth WHERE t='expert';
-- Fig 2B: work-class shares
SELECT 'F2B' f, cls, count(*) n, round(100.0*count(*)/(SELECT count(*) FROM wc),1) pct,
  round(100.0*count(*) FILTER (WHERE y=2020)/(SELECT count(*) FROM wc WHERE y=2020),1) p2020,
  round(100.0*count(*) FILTER (WHERE y=2022)/(SELECT count(*) FROM wc WHERE y=2022),1) p2022 FROM wc GROUP BY 2 ORDER BY 2;
-- Fig 4: journal mean citedness
SELECT 'F4' f, cls, count(*) n, round(median(m.impact_factor),2) med_mc, round(quantile_cont(m.impact_factor,0.25),2) q1, round(quantile_cont(m.impact_factor,0.75),2) q3
FROM wc JOIN 'fe_model.parquet' m USING(work_id) WHERE m.has_if GROUP BY 2 ORDER BY 2;
-- Fig 5: citations
SELECT 'F5' f, y, cls, count(*) n, round(avg(c2),1) mean, median(c2) med, round(100*avg((c2=0)::int),1) uncited FROM wc GROUP BY 2,3 ORDER BY 2,3;
-- Fig 7: mixing (multi-edge pairs from per-work slot counts, null = slot shares)
WITH e AS (SELECT sum(ne*(ne-1)/2) ee, sum(nb*(nb-1)/2) bb, sum(nn*(nn-1)/2) nn_, sum(ne*nb) eb, sum(ne*nn) en, sum(nb*nn) bn, sum(ne) se, sum(nb) sb, sum(nn) sn FROM wc),
     t AS (SELECT *, ee+bb+nn_+eb+en+bn tot, se+sb+sn slots FROM e)
SELECT 'F7' f, tot edges, round(100.0*se/slots,1) q_exp,
  round(100.0*ee/tot,2) obs_EE, round(100.0*pow(se/slots,2),2) null_EE, round(100.0*(ee/tot - pow(se/slots,2)),2) d_EE,
  round(100.0*(bb/tot - pow(sb/slots,2)),2) d_BB, round(100.0*(nn_/tot - pow(sn/slots,2)),2) d_NN,
  round(100.0*(eb/tot - 2*se*sb/slots/slots),2) d_EB, round(100.0*(en/tot - 2*se*sn/slots/slots),2) d_EN, round(100.0*(bn/tot - 2*sb*sn/slots/slots),2) d_BN FROM t;
