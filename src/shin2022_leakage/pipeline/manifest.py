"""R2 Hypothesis A: build the frozen, paired fold / training-multiset manifest BEFORE any model is fitted.

For each participant, scheme (WS-5F path-keyed MOABB-like; LORO path-keyed), repeat/draw and fold it fixes:
  scored  : the unique, quality-eligible test trials (canonical copy, protocol Q, rule W1), each scored ONCE
  LEAKED  : training multiset of the MOABB-like path-keyed split of the 110-path release (protocol N)
  CLEAN   : the same fold's training trials with every alias copy removed (content-hash grouped; protocol D)
  PLACEBO : CLEAN + the LEAKED arm's non-crossing alias copies + |crossing| within-training clones, class matched
            and run matched (WS-5F: BW30-run trials already in training; LORO: one seeded other run), so the training
            budget, class counts and duplicate count equal LEAKED while no test trial has a twin in training.
  SHAM_S02: S02 only (no alias exists): CLEAN + k-bar clones of its own BW30-run training trials (descriptive).
Training multisets are lists of feature-cache row indices (a row repeated = a duplicated item), stored sorted.
The manifest itself is trial-level and stays LOCAL; only its SHA256 and aggregate counts are exported.
Labels are used only for stratification and class matching; no feature is read.
"""
import argparse, json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from . import dataset as E
from . import core as C

AFFECTED = ['S01'] + [f'S{i:02d}' for i in range(3, 11)]
SHAM = 'S02'
REPEATS = range(5)
SEED_BASE = 20261027


def subject_items(cache, table, s):
    m = np.flatnonzero(table.subject == s)
    rows, paths, alias = table.rows[m], table.path[m], table.alias_copy[m]
    y = E.labels(cache, rows)
    q = np.array([cache.path_info[p]['protocols']['Q'] and not a for p, a in zip(paths, alias)])
    return rows, paths, alias, y, q


def draw_clones(rng, need_labels, pools):
    """Class-matched sampling without replacement through an ordered list of candidate pools.
    need_labels: labels of the crossing items to match. pools: list of (rows, labels). Returns (rows, n_fallback)."""
    out, fallback = [], 0
    for lab in (1, 2):
        k = int(np.sum(need_labels == lab))
        got = []
        for level, (prow, plab) in enumerate(pools):
            if len(got) >= k:
                break
            cand = np.setdiff1d(prow[plab == lab], np.asarray(got, int))
            cand = rng.permutation(cand)
            take = cand[:k - len(got)].tolist()
            fallback += len(take) if level >= 2 else 0
            got += take
        if len(got) < k:
            raise RuntimeError('placebo pool exhausted')
        out += got
    return np.asarray(out, int), fallback


def ws5f_folds(cache, table, s, r, kbar=None):
    rows, paths, alias, y, q = subject_items(cache, table, s)
    canon_fold = {}
    folds = np.empty(len(rows), int)
    for f, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(rows, y)):
        folds[te] = f
    for i in np.flatnonzero(~alias):
        canon_fold[int(rows[i])] = int(folds[i])
    has_alias = {int(rows[i]) for i in np.flatnonzero(alias)}
    alias_fold = {int(rows[i]): int(folds[i]) for i in np.flatnonzero(alias)}
    rng = np.random.default_rng(SEED_BASE + 100 * r + 7 * E.SUBJECTS.index(s))
    out = []
    for f in range(5):
        canon_tr = np.array([int(rows[i]) for i in np.flatnonzero(~alias) if folds[i] != f], int)
        scored = np.array([int(rows[i]) for i in np.flatnonzero(~alias & q) if folds[i] == f], int)
        al_tr = [int(rows[i]) for i in np.flatnonzero(alias) if folds[i] != f]
        crossing = np.array([a for a in al_tr if canon_fold[a] == f], int)
        extras = np.array([a for a in al_tr if canon_fold[a] != f], int)
        n_items_test = np.array([int(rows[i]) for i in range(len(rows)) if folds[i] == f], int)
        leaked = np.sort(np.r_[canon_tr, crossing, extras])
        clean = np.sort(canon_tr)
        lab = cache.a['label']
        bw_tr = np.array([c for c in canon_tr if c in has_alias], int)               # BW30-run trials in training
        grp3 = np.array([c for c in bw_tr if alias_fold[c] == f], int)              # alias copy sits in this test fold
        grp2 = np.array([c for c in bw_tr if alias_fold[c] != f], int)              # alias copy already in training
        if len(crossing):
            clones, fb = draw_clones(rng, lab[crossing], [(grp3, lab[grp3]), (grp2, lab[grp2]), (canon_tr, lab[canon_tr])])
        else:
            clones, fb = np.array([], int), 0
        placebo = np.sort(np.r_[canon_tr, extras, clones])
        rec = {'fold': f, 'scored': scored.tolist(), 'test_all_N': n_items_test.tolist(),
               'test_all_D': np.array([int(rows[i]) for i in np.flatnonzero(~alias) if folds[i] == f], int).tolist(),
               'exposed': sorted(set(crossing.tolist()) & set(scored.tolist())),
               'train': {'LEAKED': leaked.tolist(), 'CLEAN': clean.tolist(), 'PLACEBO': placebo.tolist()},
               'n_crossing': int(len(crossing)), 'n_extras': int(len(extras)), 'n_clone_fallback_other_run': int(fb)}
        if kbar is not None:  # S02 sham: k-bar class-balanced clones of its BW30-run training trials
            bw = np.array([c for c in canon_tr if cache.path_info_by_row[c].endswith('/BW/BW30.dat')], int)
            need = np.array([1, 2] * (kbar // 2) + ([int(rng.integers(1, 3))] if kbar % 2 else []), int)
            sh, _ = draw_clones(rng, need, [(bw, lab[bw]), (bw, lab[bw]), (canon_tr, lab[canon_tr])])
            rec['train']['SHAM_PLACEBO'] = np.sort(np.r_[canon_tr, sh]).tolist()
        out.append(rec)
    return out


def loro_folds(cache, table, s, d, kbar=None):
    rows, paths, alias, y, q = subject_items(cache, table, s)
    lab = cache.a['label']
    canon_paths = sorted({p for p, a in zip(paths, alias) if not a})
    q_paths = sorted({p for p, a, qq in zip(paths, alias, q) if qq})
    alias_rows = {int(r) for r, a in zip(rows, alias) if a}
    rng = np.random.default_rng(SEED_BASE + 1000 + 100 * d + 7 * E.SUBJECTS.index(s))
    out = []
    for P in sorted(set(paths)):
        in_p = paths == P
        is_alias_path = bool(alias[in_p][0])
        canon_tr = np.array([int(r) for r, p, a in zip(rows, paths, alias) if p != P and not a], int)
        al_tr = np.array([int(r) for r, p, a in zip(rows, paths, alias) if p != P and a], int)
        p_rows = set(int(r) for r in rows[in_p])
        crossing = np.array([a for a in al_tr if a in p_rows], int) if not is_alias_path else np.array([], int)
        extras = np.array([a for a in al_tr if a not in p_rows], int)
        scored = np.array([int(r) for r, p, a, qq in zip(rows, paths, alias, q) if p == P and qq], int)
        leaked = np.sort(np.r_[canon_tr, al_tr]) if not is_alias_path else \
            np.sort(np.array([int(r) for r, p in zip(rows, paths) if p != P], int))
        clean = np.sort(canon_tr) if not is_alias_path else np.array([], int)
        fb = 0
        src = []
        if len(crossing):
            # seeded random other quality-valid run of the same participant; second run tops up a short class
            order = [p for p in rng.permutation(np.array(q_paths, dtype=object)) if p != P]
            run_rows = [np.array([int(r) for r, p, a in zip(rows, paths, alias) if p == op and not a], int) for op in order[:2]]
            clones, fb = draw_clones(rng, lab[crossing], [(rr, lab[rr]) for rr in run_rows] + [(canon_tr, lab[canon_tr])])
            src = [str(order[0])]
        else:
            clones = np.array([], int)
        placebo = np.sort(np.r_[canon_tr, extras, clones]) if not is_alias_path else np.array([], int)
        rec = {'fold': P, 'is_alias_path': is_alias_path, 'scored': scored.tolist(),
               'test_all_N': [int(r) for r in rows[in_p]], 'test_all_D': [] if is_alias_path else [int(r) for r in rows[in_p]],
               'exposed': sorted(set(crossing.tolist()) & set(scored.tolist())),
               'train': {'LEAKED': leaked.tolist(), 'CLEAN': clean.tolist(), 'PLACEBO': placebo.tolist()},
               'n_crossing': int(len(crossing)), 'n_extras': int(len(extras)), 'clone_source_run': src,
               'n_clone_fallback_beyond_second_run': int(fb)}
        if kbar is not None and P.endswith('/BW/BW30.dat'):
            order = [p for p in rng.permutation(np.array(q_paths, dtype=object)) if p != P]
            pr = np.array([int(r) for r, p, a in zip(rows, paths, alias) if p == order[0] and not a], int)
            pr2 = np.array([int(r) for r, p, a in zip(rows, paths, alias) if p == order[min(1, len(order) - 1)] and not a], int)
            need = np.array([1, 2] * (kbar // 2) + ([int(rng.integers(1, 3))] if kbar % 2 else []), int)
            sh, _ = draw_clones(rng, need, [(pr, lab[pr]), (pr2, lab[pr2]), (canon_tr, lab[canon_tr])])
            rec['train']['SHAM_PLACEBO'] = np.sort(np.r_[canon_tr, sh]).tolist()
        elif kbar is not None and not is_alias_path:
            rec['train']['SHAM_PLACEBO'] = clean.tolist()
        out.append(rec)
    return out


def audit(cache, rec):
    """Byte-identity proof: scored test windows whose W1 sample hash occurs in each arm's training multiset."""
    h = cache.a['w1_hash']
    out = {}
    for arm, tr in rec['train'].items():
        trh = set(h[np.asarray(tr, int)]) if len(tr) else set()
        out[arm] = int(sum(1 for t in h[np.asarray(rec['scored'], int)] if t in trh)) if len(rec['scored']) else 0
    return out


def build(cache):
    table = E.build_table(cache, 'N', 'W1')
    cache.path_info_by_row = {}
    for r, p, a in zip(table.rows, table.path, table.alias_copy):
        if not a:
            cache.path_info_by_row[int(r)] = p
    man = {'WS5F': {}, 'LORO': {}}
    for s in AFFECTED:
        man['WS5F'][s] = {r: ws5f_folds(cache, table, s, r) for r in REPEATS}
        man['LORO'][s] = {d: loro_folds(cache, table, s, d) for d in REPEATS}
    kbar_ws = int(round(np.mean([fo['n_crossing'] for s in AFFECTED for r in REPEATS for fo in man['WS5F'][s][r]])))
    kbar_lo = int(round(np.mean([fo['n_crossing'] for s in AFFECTED for d in REPEATS for fo in man['LORO'][s][d]
                                 if fo['n_crossing']])))
    man['WS5F'][SHAM] = {r: ws5f_folds(cache, table, SHAM, r, kbar=kbar_ws) for r in REPEATS}
    man['LORO'][SHAM] = {d: loro_folds(cache, table, SHAM, d, kbar=kbar_lo) for d in REPEATS}
    return man, {'kbar_WS5F_per_fold': kbar_ws, 'kbar_LORO_BW30_fold': kbar_lo}


def check_invariants(cache, man):
    """Pre-fit validity checks of the placebo construction; raises on any violation. Returns aggregate counts."""
    lab = cache.a['label']
    agg = {}
    for sch, bys in man.items():
        a = {'folds': 0, 'scored_items': 0, 'crossing_items': 0, 'extras_items': 0, 'exposed_scored_items': 0,
             'audit_scored_with_twin_in_train': {'LEAKED': 0, 'CLEAN': 0, 'PLACEBO': 0, 'SHAM_PLACEBO': 0},
             'clone_fallbacks': 0, 'placebo_equals_leaked_folds': 0}
        for s, byr in bys.items():
            seen_scored = {}
            for r, folds in byr.items():
                sc_all = []
                for fo in folds:
                    tr = fo['train']
                    a['folds'] += 1
                    sc_all += fo['scored']
                    if not fo.get('is_alias_path', False):
                        L, Cl, P = (np.asarray(tr[k], int) for k in ('LEAKED', 'CLEAN', 'PLACEBO'))
                        if len(L) != len(P):
                            raise AssertionError(f'{sch} {s} {r} {fo["fold"]}: budget differs')
                        for c in (1, 2):
                            if np.sum(lab[L] == c) != np.sum(lab[P] == c):
                                raise AssertionError('class counts differ')
                        if len(L) - len(Cl) != fo['n_crossing'] + fo['n_extras']:
                            raise AssertionError('leaked != clean + alias copies')
                        if np.array_equal(L, P):
                            a['placebo_equals_leaked_folds'] += 1
                        # duplicate count: same number of repeated rows
                        dupL = len(L) - len(np.unique(L))
                        dupP = len(P) - len(np.unique(P))
                        if fo['n_crossing'] and dupP < dupL:
                            raise AssertionError('placebo has fewer within-training duplicates than leaked')
                    au = audit(cache, fo)
                    for k, v in au.items():
                        a['audit_scored_with_twin_in_train'][k] += v
                    if au.get('PLACEBO', 0) or au.get('CLEAN', 0) or au.get('SHAM_PLACEBO', 0):
                        raise AssertionError('placebo/clean arm contains a twin of a scored test item')
                    if au.get('LEAKED', 0) != len(fo['exposed']):
                        raise AssertionError('exposed count mismatch')
                    a['scored_items'] += len(fo['scored']); a['crossing_items'] += fo['n_crossing']
                    a['extras_items'] += fo['n_extras']; a['exposed_scored_items'] += len(fo['exposed'])
                    a['clone_fallbacks'] += fo.get('n_clone_fallback_other_run', 0) + fo.get('n_clone_fallback_beyond_second_run', 0)
                if len(sc_all) != len(set(sc_all)):
                    raise AssertionError('a scored trial appears twice within one repeat')
                seen_scored[r] = sorted(sc_all)
            vals = list(seen_scored.values())
            if any(v != vals[0] for v in vals):
                raise AssertionError('scored population differs between repeats')
        agg[sch] = a
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--registry', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', required=True)
    a = ap.parse_args()
    cache = E.Cache(a.cache, a.registry)
    man, kbar = build(cache)
    agg = check_invariants(cache, man)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, sort_keys=True))
    n_scored = {s: len({i for fo in man['LORO'][s][0] for i in fo['scored']}) for s in man['LORO']}
    summ = {'schema': 'shin2022_r2_hypothesis_A_manifest_summary_v1', 'scope': 'label-only fold construction; no model fitted',
            'feature_cache_sha256': C.sha256_file(a.cache), 'registry_sha256': C.sha256_file(a.registry),
            'manifest_sha256': C.sha256_file(out), 'affected_participants': AFFECTED, 'sham_participant': SHAM,
            'repeats': list(REPEATS), **kbar, 'per_scheme': agg,
            'scored_population_total': int(sum(n_scored.values())),
            'scored_population_range_per_participant': [int(min(n_scored.values())), int(max(n_scored.values()))]}
    Path(a.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(a.summary).write_text(json.dumps(summ, indent=1, sort_keys=True))
    print(json.dumps(summ, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
