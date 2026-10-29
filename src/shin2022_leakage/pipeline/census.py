"""Score-free census of the Shin2022 LIVE recordings (no model, no label-feature statistic).

Per distinct recording: trial segmentation from BCI2000 states, closed/unclosed feedback blocks, class
counts, label-agnostic feedback-duration distribution, flat-channel intervals in the first second of
feedback, amplifier sample-counter continuity (Counter1) and link-quality state, non-finite samples.
Trial-level tables are written ONLY to the local derived directory (outside Git); the exported JSON holds
per-recording aggregate counts and pooled, label-agnostic quantiles.
"""
import argparse, json
from pathlib import Path
import numpy as np

from . import reader as R


def segments(mask):
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def census_one(path):
    h = R.read_header(path)
    fs = h.sampling_rate
    sig, st = R.read_data(path, h)
    n = sig.shape[0]
    tc, fb, rc = st['TargetCode'], st['Feedback'], st['ResultCode']
    t_on, t_off = segments(tc > 0)
    f_on, f_off = segments(fb == 1)
    trials = []
    for a, b in zip(f_on, f_off):
        closed = b < n  # zero-padded edge: a block still open at EOF ends at n
        lab = int(tc[a])
        # target onset preceding this feedback onset
        prev = t_on[t_on <= a]
        cue = int(prev[-1]) if len(prev) else -1
        post_end = b
        while post_end < n and tc[post_end] > 0:
            post_end += 1
        rc_after = int(rc[min(b, n - 1)]) if closed else -1
        w1 = sig[a:min(a + int(fs), n)]
        flat_any = bool(np.any(np.ptp(w1, axis=0) == 0)) if len(w1) else True
        flat_all = bool(np.all(np.ptp(w1, axis=0) == 0)) if len(w1) else True
        trials.append(dict(fb_on=int(a), fb_off=int(b), closed=bool(closed), label=lab, cue_on=cue,
                           pre_fb_s=(a - cue) / fs if cue >= 0 else None, fb_s=(b - a) / fs,
                           tc_end=int(post_end), result_code=rc_after,
                           flat_any_1s=flat_any, flat_all_1s=flat_all,
                           nonfinite_1s=bool(not np.all(np.isfinite(w1)))))
    cnt = st.get('Counter1')
    counter = {}
    if cnt is not None:
        d = np.diff(cnt.astype(np.int64)) % (1 << 24)
        bad = np.flatnonzero(d != 1)
        counter = {'counter_steps_not_one': int(len(bad)),
                   'counter_missing_samples_total': int(np.sum(d[bad] - 1)) if len(bad) else 0}
    lq = st.get('LinkQuality1')
    flat_runs = 0
    allflat = np.all(np.diff(sig, axis=0) == 0, axis=1)
    if allflat.any():
        s0, s1 = segments(allflat)
        flat_runs = int(np.sum((s1 - s0) >= int(0.1 * fs)))
    return h, sig, st, trials, counter, (None if lq is None else np.unique(lq).tolist()), flat_runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--register', required=True, type=Path)
    p.add_argument('--store', required=True, type=Path)
    p.add_argument('--derived', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    reg = json.loads(a.register.read_text())['per_path']
    seen, rows, all_fb = {}, [], []
    for r in reg:
        sha = r['annex_sha256']
        row = {'path': r['path'].split('sourcedata/', 1)[1], 'subject': r['subject'], 'label': r['label'],
               'annex_sha256': sha, 'register_status': r['status'], 'register_flags': r['flags']}
        if sha in seen:
            row['alias_of'] = seen[sha]
            rows.append(row)
            continue
        seen[sha] = row['path']
        h, sig, st, trials, counter, lq, flat_runs = census_one(a.store / f'{sha}.dat')
        fbd = np.array([t['fb_s'] for t in trials if t['closed']])
        all_fb.extend(fbd.tolist())
        labs = [t['label'] for t in trials if t['closed']]
        row.update({
            'n_samples': int(sig.shape[0]), 'duration_s': sig.shape[0] / h.sampling_rate,
            'sampling_rate': h.sampling_rate, 'source_ch': h.source_ch,
            'n_target_segments': int(len(segments(st['TargetCode'] > 0)[0])),
            'n_feedback_blocks': len(trials), 'n_closed_feedback_blocks': int(len(fbd)),
            'unclosed_feedback_at_eof': int(sum(not t['closed'] for t in trials)),
            'closed_by_target_code': {str(k): int(labs.count(k)) for k in sorted(set(labs))},
            'closed_fb_shorter_than_1s': int(np.sum(fbd < 1.0)), 'closed_fb_shorter_than_3s': int(np.sum(fbd < 3.0)),
            'closed_fb_at_cap_6s': int(np.sum(fbd >= 5.99)),
            'pre_feedback_s_unique': sorted({round(t['pre_fb_s'], 3) for t in trials if t['pre_fb_s'] is not None}),
            'windows_1s_flat_any': int(sum(t['flat_any_1s'] and t['closed'] and t['fb_s'] >= 1.0 for t in trials)),
            'windows_1s_flat_all': int(sum(t['flat_all_1s'] and t['closed'] and t['fb_s'] >= 1.0 for t in trials)),
            'windows_1s_nonfinite': int(sum(t['nonfinite_1s'] and t['closed'] and t['fb_s'] >= 1.0 for t in trials)),
            'all_channel_flat_runs_ge_100ms': flat_runs,
            'link_quality_values': lq, **counter,
            'nonfinite_samples': int(np.sum(~np.isfinite(sig))),
        })
        rows.append(row)
        np.savez(a.derived / f'{sha}_trials.npz', **{k: np.array([t[k] if t[k] is not None else np.nan for t in trials])
                                                   for k in trials[0]})
        print(row['path'], row['n_closed_feedback_blocks'], row['closed_fb_shorter_than_1s'],
              row['windows_1s_flat_any'], row.get('counter_steps_not_one'), flush=True)
    fb = np.array(all_fb)
    distinct = [r for r in rows if 'alias_of' not in r]
    out = {'schema': 'shin2022_secondary_score_free_census_v1',
           'scope': 'label-agnostic structure only; no model fitted, no label-feature statistic computed',
           'distinct_recordings': len(distinct), 'paths': len(rows),
           'closed_feedback_blocks_distinct': int(sum(r['n_closed_feedback_blocks'] for r in distinct)),
           'unclosed_feedback_blocks_distinct': int(sum(r['unclosed_feedback_at_eof'] for r in distinct)),
           'closed_shorter_than_1s': int(np.sum(fb < 1)), 'closed_shorter_than_3s': int(np.sum(fb < 3)),
           'closed_at_6s_cap': int(np.sum(fb >= 5.99)),
           'pooled_feedback_duration_quantiles_s': {q: float(np.quantile(fb, q / 100)) for q in (5, 25, 50, 75, 95)},
           'windows_1s_flat_any_among_ge1s': int(sum(r['windows_1s_flat_any'] for r in distinct)),
           'windows_1s_flat_all_among_ge1s': int(sum(r['windows_1s_flat_all'] for r in distinct)),
           'recordings_with_counter_gaps': int(sum((r.get('counter_steps_not_one') or 0) > 0 for r in distinct)),
           'rows': rows}
    a.output.write_text(json.dumps(out, indent=1, sort_keys=True))
    print({k: v for k, v in out.items() if k != 'rows'})


if __name__ == '__main__':
    main()
