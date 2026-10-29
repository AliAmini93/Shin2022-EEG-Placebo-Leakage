"""Build the per-trial feature cache for the frozen Shin2022 protocol (local only, outside Git).

One row per feedback block of each DISTINCT recording (content SHA256). Features per pre-registered window:
  W1    [t0, t0+1 s)                       (prospective; defined when the block is closed and lasts >= 1 s)
  W3    [t0, t0+3 s)                       (MOABB naive; defined when 3 s of samples exist after t0)
  W3c   [t0, min(t0+3 s, t_end))           (prospective, variable length; defined with W1)
  Wpost [t_end, t_end+1 s)                 (retrospective outcome-period control; defined with W1)
  Wpre  [t_cue-1 s, t_cue)                 (pre-cue negative control; defined with W1)
Signals are filtered CAUSALLY per recording (sosfilt with steady-state initial conditions).
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from . import reader as R

FS = 250.0
FB_BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32)]
WINDOWS = ('W1', 'W3', 'W3c', 'Wpost', 'Wpre')
EPS_POW = 1e-20


def causal_bandpass(x, lo, hi, fs=FS, order=4):
    sos = butter(order, [lo, hi], btype='band', fs=fs, output='sos')
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    y, _ = sosfilt(sos, x, axis=0, zi=zi)
    return y


def oas_cov(seg):
    """OAS-shrunk covariance of a (samples x ch) segment, mean removed, with a tiny numerical floor."""
    x = seg - seg.mean(0, keepdims=True)
    n, p = x.shape
    s = x.T @ x / n
    mu = np.trace(s) / p
    alpha = np.mean(s ** 2)
    num = alpha + mu ** 2
    den = (n + 1.0) * (alpha - mu ** 2 / p)
    shrink = 1.0 if den == 0 else min(num / den, 1.0)
    c = (1.0 - shrink) * s + shrink * mu * np.eye(p)
    floor = max(1e-10 * mu, 1e-12)
    return c + floor * np.eye(p)


def spectral32(seg):
    """BCI-Report-style relative log band powers: alpha 8-12, beta 13-30 over 1-40 Hz (inclusive bins)."""
    x = seg - seg.mean(0, keepdims=True)
    L = x.shape[0]
    w = np.hanning(L)[:, None]
    P = np.abs(np.fft.rfft(x * w, axis=0)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / FS)
    ref = P[(f >= 1) & (f <= 40)].sum(0) + EPS_POW
    a = P[(f >= 8) & (f <= 12)].sum(0) + EPS_POW
    b = P[(f >= 13) & (f <= 30)].sum(0) + EPS_POW
    return np.r_[np.log(a / ref), np.log(b / ref)]


def laplacian_matrix(h):
    """Sparse BCI2000 SpatialFilter (rows: input, output, weight) -> dense (ch x outputs) for outputs used."""
    m = h.matrix('SpatialFilter')
    outs = sorted({int(float(r[1])) for r in m})
    W = np.zeros((h.source_ch, len(outs)))
    for r in m:
        i, o, wt = int(float(r[0])) - 1, outs.index(int(float(r[1]))), float(r[2])
        W[i, o] += wt
    return W, outs


def m0_feature(seg_raw, W, outs, h):
    """log alpha power of the classifier's two Laplacian outputs, weighted by the header Classifier signs."""
    lap = (seg_raw - seg_raw.mean(0, keepdims=True)) @ W
    L = lap.shape[0]
    P = np.abs(np.fft.rfft(lap * np.hanning(L)[:, None], axis=0)) ** 2
    f = np.fft.rfftfreq(L, 1.0 / FS)
    la = np.log(P[(f >= 8) & (f <= 12)].sum(0) + EPS_POW)
    cl = h.matrix('Classifier')  # rows: input channel, bin, output, weight
    val = 0.0
    for r in cl:
        val += float(r[3]) * la[outs.index(int(float(r[0])))]
    return val  # = log P(C4lap) - log P(C3lap) for the Shin2022 classifier


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--registry', required=True, type=Path)
    p.add_argument('--store', required=True, type=Path)
    p.add_argument('--derived', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    reg = json.loads(a.registry.read_text())['rows']
    done = set()
    cols = {k: [] for k in ('sha', 'block', 'subject', 'label', 'fb_s', 'closed', 'w1_base', 'w1_ok', 'w3_ok', 'wpre_ok',
                            'flat_any_1s', 'w1_hash', 'w3_hash', 'm0_W1', 'm0_W3')}
    covs = {w: [] for w in WINDOWS}
    specs = {w: [] for w in WINDOWS}
    fbcov = []
    e440_w1, e440_w3 = [], []
    for r in reg:
        sha = r['annex_sha256']
        if sha in done:
            continue
        done.add(sha)
        path = a.store / f'{sha}.dat'
        h = R.read_header(path)
        sig, _ = R.read_data(path, h)
        n = sig.shape[0]
        t = np.load(a.derived / f'{sha}_trials.npz')
        bp830 = causal_bandpass(sig, 8, 30)
        bp440 = causal_bandpass(sig, 4, 40)
        fbs = [causal_bandpass(sig, lo, hi) for lo, hi in FB_BANDS]
        Wlap, outs = laplacian_matrix(h)
        raw32 = np.fromfile(path, dtype=np.dtype([('sig', '<f4', (h.source_ch,)), ('st', 'u1', (h.statevector_len,))]),
                            count=n, offset=h.header_len)['sig']
        for k in range(len(t['fb_on'])):
            t0, te = int(t['fb_on'][k]), int(t['fb_off'][k])
            closed = bool(t['closed'][k])
            fb_s = float(t['fb_s'][k])
            cue = int(t['cue_on'][k]) if np.isfinite(t['cue_on'][k]) else -1
            seg1 = sig[t0:t0 + 250]
            finite = bool(np.all(np.isfinite(seg1))) and len(seg1) == 250
            flat = bool(t['flat_any_1s'][k])
            w1_base = closed and fb_s >= 1.0 and finite  # flat rule NOT applied (ablation A8 only)
            w1_ok = w1_base and not flat
            w3_ok = t0 + 750 <= n
            wpre_ok = w1_ok and cue - 250 >= 0 and te + 250 <= n
            cols['sha'].append(sha); cols['block'].append(k); cols['subject'].append(r['subject'])
            cols['label'].append(int(t['label'][k])); cols['fb_s'].append(fb_s); cols['closed'].append(closed)
            cols['w1_base'].append(w1_base); cols['w1_ok'].append(w1_ok); cols['w3_ok'].append(w3_ok); cols['wpre_ok'].append(wpre_ok)
            cols['flat_any_1s'].append(flat)
            cols['w1_hash'].append(hashlib.sha256(raw32[t0:t0 + 250].tobytes()).hexdigest())
            cols['w3_hash'].append(hashlib.sha256(raw32[t0:t0 + 750].tobytes()).hexdigest() if w3_ok else '')
            segs = {'W1': (t0, t0 + 250) if w1_base else None,
                    'W3': (t0, t0 + 750) if w3_ok else None,
                    'W3c': (t0, min(t0 + 750, te)) if w1_ok else None,
                    'Wpost': (te, te + 250) if wpre_ok else None,
                    'Wpre': (cue - 250, cue) if wpre_ok else None}
            for w, s in segs.items():
                if s is None:
                    covs[w].append(np.full((16, 16), np.nan)); specs[w].append(np.full(32, np.nan))
                else:
                    covs[w].append(oas_cov(bp830[s[0]:s[1]]))
                    specs[w].append(spectral32(sig[s[0]:s[1]]))
            cols['m0_W1'].append(m0_feature(sig[t0:t0 + 250], Wlap, outs, h) if w1_base else np.nan)
            cols['m0_W3'].append(m0_feature(sig[t0:t0 + 750], Wlap, outs, h) if w3_ok else np.nan)
            fbcov.append(np.stack([oas_cov(b[t0:t0 + 250]) for b in fbs]) if w1_base else np.full((7, 16, 16), np.nan))
            e440_w1.append(bp440[t0:t0 + 250].T.astype(np.float32) if w1_base else np.zeros((16, 250), np.float32))
            e440_w3.append(bp440[t0:t0 + 750].T.astype(np.float32) if w3_ok else np.zeros((16, 750), np.float32))
        print(r['path'], len(t['fb_on']), flush=True)
    out = {k: np.asarray(v) for k, v in cols.items()}
    for w in WINDOWS:
        out[f'cov830_{w}'] = np.asarray(covs[w])
        out[f'spec32_{w}'] = np.asarray(specs[w])
    out['covFB_W1'] = np.asarray(fbcov)
    out['eeg440_W1'] = np.asarray(e440_w1)
    out['eeg440_W3'] = np.asarray(e440_w3)
    np.savez(a.output, **out)
    print('rows', len(out['sha']), 'w1_ok', int(out['w1_ok'].sum()), 'w3_ok', int(out['w3_ok'].sum()),
          'wpre_ok', int(out['wpre_ok'].sum()))


if __name__ == '__main__':
    main()
