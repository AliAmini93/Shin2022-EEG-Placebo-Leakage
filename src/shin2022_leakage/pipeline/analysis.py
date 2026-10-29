"""R2 frozen analysis: local per-participant results -> aggregate-only JSON and the pre-registered gate labels.

The gate logic is a literal transcription of R2_FROZEN_ANALYSIS_PLAN.json. Only aggregates (means, intervals, p, sign
counts) leave this script; per-participant values never do.
Usage: r2_analyze.py A --m3 ... --m6 ... --out ...
       r2_analyze.py B --dataset shin|stieger --res ... --out ... [--cache-summary ...]
"""
import argparse, json
from pathlib import Path

import numpy as np

from . import core as C

AFFECTED = ['S01'] + [f'S{i:02d}' for i in range(3, 11)]
ALL10 = [f'S{i:02d}' for i in range(1, 11)]


def subj_mean(res, scheme, s, key):
    vals = [v[key] for k, v in res.items() if k.startswith(f'{scheme}|{s}|') and key in v]
    return float(np.mean(vals)) if vals else None


def vec(res, scheme, subjects, a, b=None):
    out = []
    for s in subjects:
        x = subj_mean(res, scheme, s, a)
        y = subj_mean(res, scheme, s, b) if b else 0.0
        out.append(x - y)
    return np.array(out)


def analyze_a(m3, m6):
    R = {'M3': m3, 'M6': m6}
    prim, out = {}, {'primary': {}, 'secondary': {}, 'sham_S02': {}, 'levels': {}}
    for m, res in R.items():
        for sch in ('WS5F', 'LORO'):
            c = f'{m}_{sch}'
            d = vec(res, sch, AFFECTED, 'BA_LEAKED', 'BA_PLACEBO')
            s = C.paired_summary(d)
            s['magnitude'] = C.classify_magnitude(s)
            out['primary'][c] = s
            prim[c] = s['p_signflip_two_sided']
            sec = {}
            for name, (a, b, subs) in {'LEAKED_minus_CLEAN_affected9': ('BA_LEAKED', 'BA_CLEAN', AFFECTED),
                                       'PLACEBO_minus_CLEAN_affected9': ('BA_PLACEBO', 'BA_CLEAN', AFFECTED),
                                       'LEAKED_minus_CLEAN_all10': ('BA_LEAKED', 'BA_CLEAN', ALL10),
                                       'EXPOSED_LEAKED_minus_PLACEBO_affected9': ('BA_EXPOSED_LEAKED', 'BA_EXPOSED_PLACEBO', AFFECTED),
                                       'EXPOSED_PLACEBO_minus_CLEAN_affected9': ('BA_EXPOSED_PLACEBO', 'BA_EXPOSED_CLEAN', AFFECTED),
                                       'ASMOABB_LEAKED_asreported_minus_CLEAN_asreported_all10':
                                           ('BA_ASREPORTED_LEAKED', 'BA_ASREPORTED_CLEAN', ALL10)}.items():
                ss = C.paired_summary(vec(res, sch, subs, a, b)); ss['magnitude'] = C.classify_magnitude(ss)
                sec[name] = ss
            out['secondary'][c] = sec
            out['levels'][c] = {arm: C.level_summary(vec(res, sch, ALL10, f'BA_{arm}'))
                                for arm in ('LEAKED', 'CLEAN', 'PLACEBO')}
            sh = subj_mean(res, sch, 'S02', 'BA_SHAM_PLACEBO')
            cl = subj_mean(res, sch, 'S02', 'BA_CLEAN')
            reps = [v['BA_SHAM_PLACEBO'] - v['BA_CLEAN'] for k, v in res.items() if k.startswith(f'{sch}|S02|')]
            # n = 1 by construction: export booleans only (no participant-level number leaves this script)
            out['sham_S02'][c] = {'n_participants': 1, 'material_ge_2pp': bool(abs(sh - cl) >= 0.02),
                                  'all_repeats_within_2pp': bool(max(abs(x) for x in reps) < 0.02),
                                  'leaked_equals_clean_S02': subj_mean(res, sch, 'S02', 'BA_LEAKED') == cl}
    adj = C.holm(prim)
    survive = 0
    relevant = 0
    for c, s in out['primary'].items():
        s['p_holm4'] = adj[c]
        s['survives'] = bool(adj[c] < 0.05 and s['mean_diff'] > 0)
        survive += s['survives']; relevant += s['magnitude'] == 'RELEVANT_POSITIVE'
    null = sum(s['magnitude'] == 'PRACTICALLY_NULL' for s in out['primary'].values())
    material_sham = any(v['material_ge_2pp'] for v in out['sham_S02'].values())
    if survive >= 3 and relevant >= 2 and not material_sham:
        verdict = 'A_CONFIRMED_RELEVANT'
    elif survive >= 3 and not material_sham:
        verdict = 'A_CONFIRMED_SUB_SESOI'
    elif survive == 0 and null >= 2:
        verdict = 'A_REFUTED'
    else:
        verdict = 'A_INCONCLUSIVE'
    out['gate'] = {'n_survive_holm': survive, 'n_relevant_positive': relevant, 'n_practically_null': null,
                   'material_sham_S02': material_sham, 'verdict': verdict}
    out['determinism_audit_M6'] = m6.get('determinism_audit')
    out['fits'] = {'M3': m3.get('_meta', {}).get('n_fits'), 'M6': m6.get('_meta', {}).get('n_fits')}
    return out



def main():
    ap = argparse.ArgumentParser(description="R2 frozen Hypothesis A aggregate-only analysis")
    ap.add_argument("--m3", required=True, help="Local TS+LR outcomes JSON; never commit")
    ap.add_argument("--m6", required=True, help="Local EEGNet outcomes JSON; never commit")
    ap.add_argument("--plan", required=True, help="Original frozen plan JSON")
    ap.add_argument("--out", required=True, help="Write aggregate-only JSON outside this public repo")
    args = ap.parse_args()
    result = analyze_a(json.loads(Path(args.m3).read_text()), json.loads(Path(args.m6).read_text()))
    result["schema"] = "shin2022_r2_A_aggregates_v1"
    result["plan_sha256"] = C.sha256_file(args.plan)
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result["gate"], indent=2))

if __name__ == "__main__":
    main()
