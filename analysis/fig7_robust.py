"""Fig 7 robustness (revision items 12, 13, 15).

Inputs (from fig7_export.sql / duckdb on covid_works_authors_ow.parquet):
  fig7_edges.csv  u, v, yr, k: one row per co-author pair per shared work (10,171,294 multi-edges),
                  with the work's year and classified team size k
  fig7_slots.csv  i, t, yr, k: one row per authorship slot (author index, type 0/1/2 = OS/bellw./newc.)
  fig7_work_counts.csv: per-work type counts, used only for the number of works per panel

Mixing follows the main figure: multi-edge shares; null = label permutation across authorship slots,
expected share q_i^2 / 2 q_i q_j with q = slot shares of the population analyzed.

CIs: vertex bootstrap (Snijders & Borgatti 1999). Authors are resampled with replacement and each
edge is weighted by the product of its endpoints' draw counts; slots by their author's count.
Papers are not independent (authors recur, OS average 4.14 papers), so resampling papers would
understate the spread by about half.

Writes fig7_robust.csv (long format: view, subset, cell, obs, null, diff, lo/hi CIs).
"""
import numpy as np
import pandas as pd

CELLS = ['EE', 'BB', 'NN', 'EB', 'EN', 'BN']
B = 1000
rng = np.random.default_rng(0)

sl = pd.read_csv('fig7_slots.csv')
e = pd.read_csv('fig7_edges.csv')
wk = pd.read_csv('fig7_work_counts.csv')
wk['k'] = wk[['ne', 'nb', 'nn']].sum(1)

N = sl.i.max() + 1
t = np.zeros(N, int)
t[sl.i.to_numpy()] = sl.t.to_numpy()
s_i, s_t, s_yr, s_k = (sl[c].to_numpy() for c in ('i', 't', 'yr', 'k'))
u, v, e_yr, e_k = (e[c].to_numpy() for c in ('u', 'v', 'yr', 'k'))
lo_t, hi_t = np.minimum(t[u], t[v]), np.maximum(t[u], t[v])
code = np.select([(lo_t == 0) & (hi_t == 0), (lo_t == 1) & (hi_t == 1), (lo_t == 2) & (hi_t == 2),
                  (lo_t == 0) & (hi_t == 1), (lo_t == 0) & (hi_t == 2)], [0, 1, 2, 3, 4], 5)
assert len(e) == 10_171_294 and len(sl) == 2_558_150


def stats(m, em, sm, ew):
    """m: per-author draw counts; em/sm: edge/slot masks; ew: edge weights. -> obs, null, diff (%)."""
    q = np.bincount(s_t[sm], weights=m[s_i[sm]], minlength=3)
    q = q / q.sum()
    E = np.bincount(code[em], weights=ew[em] * m[u[em]] * m[v[em]], minlength=6)
    E = E / E.sum()
    null = np.array([q[0]**2, q[1]**2, q[2]**2, 2*q[0]*q[1], 2*q[0]*q[2], 2*q[1]*q[2]])
    return 100*E, 100*null, 100*(E - null)


def run(view, subset, em, sm, wm, ew=None):
    ew = np.ones(len(e)) if ew is None else ew
    obs, null, diff = stats(np.ones(N), em, sm, ew)
    boots = np.array([np.concatenate(stats(np.bincount(rng.integers(0, N, N), minlength=N).astype(float),
                                           em, sm, ew)) for _ in range(B)])
    pct = lambda j: np.percentile(boots[:, j], [2.5, 97.5])
    rows = []
    for i, c in enumerate(CELLS):
        (ol, oh), (nl, nh), (dl, dh) = pct(i), pct(6 + i), pct(12 + i)
        rows.append(dict(view=view, subset=subset, cell=c, works=int(wm.sum()),
                         edges_unweighted=int(em.sum()), obs=obs[i], null=null[i], diff=diff[i],
                         obs_lo=ol, obs_hi=oh, null_lo=nl, null_hi=nh, diff_lo=dl, diff_hi=dh))
    return rows


ALL_E, ALL_S = np.ones(len(e), bool), np.ones(len(sl), bool)
rows = run('main', 'all', ALL_E, ALL_S, np.ones(len(wk), bool))
# item 13a: each paper counts once: edges weighted 1/C(k,2); slots unweighted (main null)
rows += run('paper_weighted', 'all', ALL_E, ALL_S, wk.k.to_numpy() >= 2, 2 / (e_k * (e_k - 1)))
# item 13b: team-size bins on classified authors, null permuted within bin
for lab, lo, hi in [('2', 2, 2), ('3-5', 3, 5), ('6-10', 6, 10), ('11-30', 11, 30)]:
    rows += run('size_bin', lab, (e_k >= lo) & (e_k <= hi), (s_k >= lo) & (s_k <= hi),
                (wk.k >= lo).to_numpy() & (wk.k <= hi).to_numpy())
# item 15: per OW year, null permuted within year
for y in (2020, 2021, 2022):
    rows += run('year', str(y), e_yr == y, s_yr == y, wk.yr.to_numpy() == y)

out = pd.DataFrame(rows)
out.to_csv('fig7_robust.csv', index=False)

# self-checks against the published Fig 7 (CLAUDE.md step 6)
m = out[out.view == 'main'].set_index('cell')
assert abs(m.loc['EE', 'obs'] - 7.46) < 0.01 and abs(m.loc['EE', 'diff'] - 4.70) < 0.01
assert abs(m.loc['BN', 'diff'] + 7.39) < 0.01
assert out[out.view == 'size_bin'].groupby('subset').works.first().sum() == (wk.k >= 2).sum()
assert (np.sign(out['diff_lo']) == np.sign(out['diff_hi'])).all(), 'a CI crosses zero'
pd.set_option('display.width', 200)
print(out[['view', 'subset', 'cell', 'works', 'obs', 'diff', 'diff_lo', 'diff_hi']].round(2).to_string(index=False))
hw = lambda x, c: np.maximum(x[c] - x[c + '_lo'], x[c + '_hi'] - x[c]).max()
print('main max half-width: obs %.3f diff %.3f; all panels diff %.3f'
      % (hw(m, 'obs'), hw(m, 'diff'), hw(out, 'diff')))
