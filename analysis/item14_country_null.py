"""Item 14: mixing-matrix null that preserves each author's country (or institution).

Inputs from item14_export.sql: item14_slots.csv (i, t, w, c, inst), item14_edges.csv (u, v, w, cu, cv),
item14_countries.csv (c, cc; 'XX' = no country recorded).

Null: author types are permuted among authorship slots *within each country*. Writing q^(c) for the type
shares of country c's slots, an edge whose ends sit in countries a and b has expected type pair
q^(a) (x) q^(b), so the expected share of cell (i, j) is the edge-weighted mean of q_i^(a) q_j^(b)
(+ q_j^(a) q_i^(b) off the diagonal). With one country this is the main null (q_i^2, 2 q_i q_j).
CIs: the vertex bootstrap of fig7_robust.py (authors resampled; edges weighted m[u] m[v], slots m[i]),
reported as bias-corrected estimates (diff - bootstrap bias) with basic bootstrap intervals, which are
centered on them (see the comment in the loop).

Views: 'country_all' keeps every slot, with no-country slots as their own stratum (main view);
'country_complete' keeps only works whose every slot has a country (sensitivity);
'institution_all' uses institutions as strata: no-institution slots form one stratum and institutions with
fewer than 10 slots are pooled into another (3% of slots), since a tiny stratum fixes its own labels and
would make the null equal the observed by construction.
Strata pairs are aggregated sparsely (unique (stratum of u, stratum of v) pairs), so 35k institutions are cheap.
Writes item14_country_null.csv; prints self-checks and descriptives.
"""
import numpy as np
import pandas as pd

CELLS = ['EE', 'BB', 'NN', 'EB', 'EN', 'BN']
B = 1000
rng = np.random.default_rng(0)

sl = pd.read_csv('item14_slots.csv')
e = pd.read_csv('item14_edges.csv')
cc = pd.read_csv('item14_countries.csv', keep_default_na=False)
assert len(sl) == 2_558_150 and len(e) == 10_171_294
C = len(cc)
XX = int(cc.loc[cc.cc == 'XX', 'c'].iloc[0])
N = sl.i.max() + 1
t = np.zeros(N, int)
t[sl.i.to_numpy()] = sl.t.to_numpy()
s_i, s_t, s_w, s_c = (sl[k].to_numpy() for k in ('i', 't', 'w', 'c'))
u, v, e_w, cu, cv = (e[k].to_numpy() for k in ('u', 'v', 'w', 'cu', 'cv'))


def cell_code(tu, tv):
    lo, hi = np.minimum(tu, tv), np.maximum(tu, tv)
    return np.select([(lo == 0) & (hi == 0), (lo == 1) & (hi == 1), (lo == 2) & (hi == 2),
                      (lo == 0) & (hi == 1), (lo == 0) & (hi == 2)], [0, 1, 2, 3, 4], 5)


code = cell_code(t[u], t[v])

# slot row of each edge end, so any slot-level stratum can be carried to edges
key_s = s_i.astype(np.int64) * (s_w.max() + 1) + s_w
order = np.argsort(key_s)
pos = order[np.searchsorted(key_s[order], u.astype(np.int64) * (s_w.max() + 1) + e_w)]
pos_v = order[np.searchsorted(key_s[order], v.astype(np.int64) * (s_w.max() + 1) + e_w)]
assert (s_i[pos] == u).all() and (s_i[pos_v] == v).all() and (s_c[pos] == cu).all() and (s_c[pos_v] == cv).all()


def strata(g_slot):
    """Dense stratum code per slot -> (slot codes, G, unique edge-pair index, pair -> (a, b))."""
    g = pd.factorize(g_slot)[0]
    G = g.max() + 1
    pr = g[pos].astype(np.int64) * G + g[pos_v]
    uniq, pidx = np.unique(pr, return_inverse=True)
    return g, G, pidx, uniq // G, uniq % G


def stats(m, em, sm, S):
    g, G, pidx, pa, pb = S
    Q = np.bincount(g[sm] * 3 + s_t[sm], weights=m[s_i[sm]], minlength=G * 3).reshape(G, 3)
    Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-300)
    ew = m[u[em]] * m[v[em]]
    M = np.bincount(pidx[em], weights=ew, minlength=len(pa))
    M = M / M.sum()
    A, Bq = Q[pa], Q[pb]
    f = lambda i, j: (M * A[:, i] * Bq[:, j]).sum()
    null = np.array([f(0, 0), f(1, 1), f(2, 2), f(0, 1) + f(1, 0), f(0, 2) + f(2, 0), f(1, 2) + f(2, 1)])
    E = np.bincount(code[em], weights=ew, minlength=6)
    E = E / E.sum()
    return 100 * E, 100 * null, 100 * (E - null)


S_one = strata(np.zeros(len(sl), int))
S_cty = strata(s_c)
inst = sl.inst.to_numpy()
isize = pd.Series(inst).map(pd.Series(inst).value_counts()).to_numpy()
g_inst = np.where(np.isnan(inst), -1, np.where(isize < 10, -2, inst))
S_inst = strata(g_inst)
print('strata: countries %d, institution strata %d, institution edge pairs %d' % (S_cty[1], S_inst[1], len(S_inst[3])))

ALL_E, ALL_S, ONES = np.ones(len(e), bool), np.ones(len(sl), bool), np.ones(N)

# self-check 1: one stratum reproduces the main Fig 7 null
_, n0, d0 = stats(ONES, ALL_E, ALL_S, S_one)
assert abs(n0[0] - 2.77) < 0.01 and abs(d0[0] - 4.70) < 0.01 and abs(d0[5] + 7.39) < 0.01, (n0, d0)

# self-check 2: three real within-country shuffles of slot labels agree with the closed form
_, n_cf, _ = stats(ONES, ALL_E, ALL_S, S_cty)
gaps = []
for r in range(3):
    ts = s_t.copy()
    for grp in pd.Series(np.arange(len(sl))).groupby(s_c).indices.values():
        ts[grp] = ts[rng.permutation(grp)]
    sh = np.bincount(cell_code(ts[pos], ts[pos_v]), minlength=6) / len(e) * 100
    gaps.append(np.abs(sh - n_cf).max())
print('shuffle vs closed form, max gap (pp):', np.round(gaps, 3))
assert max(gaps) < 0.1

# tie-rule sensitivity (point estimates): last instead of first country / institution for multi-affiliation slots
cmax, imax = sl.c_max.to_numpy(), sl.inst_max.to_numpy()
isz2 = pd.Series(imax).map(pd.Series(imax).value_counts()).to_numpy()
for lab, S1, S2 in [('country', S_cty, strata(cmax)),
                    ('institution', S_inst, strata(np.where(np.isnan(imax), -1, np.where(isz2 < 10, -2, imax))))]:
    d1, d2 = stats(ONES, ALL_E, ALL_S, S1)[2], stats(ONES, ALL_E, ALL_S, S2)[2]
    print('tie rule (%s), first:' % lab, np.round(d1, 3), 'last:', np.round(d2, 3),
          'max |diff| = %.3f pp, same signs: %s' % (np.abs(d1 - d2).max(), (np.sign(d1) == np.sign(d2)).all()))

complete_w = ~np.isin(np.arange(s_w.max() + 1), np.unique(s_w[s_c == XX]))
views = {'main': (ALL_E, ALL_S, S_one), 'country_all': (ALL_E, ALL_S, S_cty), 'country_complete': (complete_w[e_w], complete_w[s_w], S_cty),
         'institution_all': (ALL_E, ALL_S, S_inst)}
rows = []
for view, (em, sm, S) in views.items():
    obs, null, diff = stats(ONES, em, sm, S)
    boots = np.array([np.concatenate(stats(np.bincount(rng.integers(0, N, N), minlength=N).astype(float), em, sm, S))
                      for _ in range(B)])
    point = np.concatenate([obs, null, diff])
    # basic (reverse-percentile) bootstrap: 2*point - quantiles. Resampled authors drawn twice double their
    # slots in small institution strata, which shifts the bootstrap distribution by ~0.1-0.2 pp; the basic
    # interval removes that shift. For the country views it is within 0.01 pp of the percentile interval.
    pct = lambda j: tuple(2 * point[j] - np.percentile(boots[:, j], [97.5, 2.5]))
    bias = boots.mean(0)[12:] - diff
    print(view, 'bootstrap bias in diff (pp):', np.round(bias, 3))
    for k, cell in enumerate(CELLS):
        (ol, oh), (nl, nh), (dl, dh) = pct(k), pct(6 + k), pct(12 + k)
        rows.append(dict(view=view, cell=cell, works=int(np.unique(e_w[em]).size), edges=int(em.sum()),
                         obs=obs[k], null=null[k], diff=diff[k], ratio=obs[k] / null[k],
                         diff_bc=diff[k] - bias[k], ratio_bc=obs[k] / (null[k] + bias[k]),
                         obs_lo=ol, obs_hi=oh, null_lo=nl, null_hi=nh, diff_lo=dl, diff_hi=dh))
out = pd.DataFrame(rows)
out.to_csv('item14_country_null.csv', index=False)
pd.set_option('display.width', 200)
print(out[['view', 'cell', 'works', 'obs', 'null', 'diff', 'diff_bc', 'diff_lo', 'diff_hi', 'ratio', 'ratio_bc']].round(3).to_string(index=False))

# descriptives for the text
known = (cu != XX) & (cv != XX)
print('edges with both countries known: %.1f%%; of those cross-country: %.1f%%'
      % (100 * known.mean(), 100 * (cu[known] != cv[known]).mean()))
d = pd.DataFrame({'c': s_c, 't': s_t})
d = d[d.c != XX]
g = d.groupby('c').t.agg(slots='size', os_share=lambda x: (x == 0).mean())
g = g.join(cc.set_index('c')).sort_values('slots', ascending=False)
print('OS slot share overall (known countries): %.1f%%' % (100 * (d.t == 0).mean()))
print(g.head(10).assign(os_share=lambda x: (100 * x.os_share).round(1)).to_string())
inst = sl.inst.dropna().value_counts()
print('institutions: %d; median slots %.0f; slots in institutions with >=100 slots: %.1f%%; with <10: %.1f%%'
      % (len(inst), inst.median(), 100 * inst[inst >= 100].sum() / inst.sum(), 100 * inst[inst < 10].sum() / inst.sum()))
