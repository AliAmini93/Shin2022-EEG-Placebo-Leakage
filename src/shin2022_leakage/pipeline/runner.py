"""R2 Hypothesis A runner: LEAKED / CLEAN / PLACEBO (and S02 SHAM) arms of the frozen manifest, frozen models only.

Models: M3 TS+LR (R1 exp_core M3, OAS covariance 8-30 Hz of W1) and M6 EEGNet-8,2 (R1 schedule, causal 4-40 Hz W1).
Seeds: WS-5F repeat r uses split seed r, placebo draw r and EEGNet seed r in every arm; LORO draw d uses placebo
draw d and EEGNet seed d in every arm (TS+LR is seed-free). A training multiset is the sorted list of cache rows, so
identical multisets with the same seed are fitted ONCE (memoised); a determinism audit re-fits pre-specified folds.
Per-participant metrics are written ONLY to a local JSON outside Git.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
import argparse, hashlib, json, time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from . import core as C

ARMS = ('LEAKED', 'CLEAN', 'PLACEBO', 'SHAM_PLACEBO')


def key_of(model, seed, train, test):
    h = hashlib.sha1(np.asarray(train, np.int64).tobytes() + b'|' + np.asarray(test, np.int64).tobytes()).hexdigest()
    return f'{model}|{seed}|{h}'


class Fitter:
    def __init__(self, cache_npz, model):
        self.model = model
        c = np.load(cache_npz, allow_pickle=False)
        self.y = c['label'].astype(int)
        self.hash = c['w1_hash']
        if model == 'M3':
            self.X = c['cov830_W1']
        else:
            self.Xnp = C.zscore_trials(c['eeg440_W1'])
            self._to_gpu()
        self.memo = {}
        self.n_fits = 0

    def _to_gpu(self):
        import torch
        C.gpu_wait()
        self.Xg = torch.from_numpy(self.Xnp).unsqueeze(1).cuda()
        self.yg = torch.from_numpy(self.y - 1).cuda()

    def _release(self):
        import torch
        del self.Xg, self.yg
        torch.cuda.empty_cache()

    def predict(self, train, test, seed):
        k = key_of(self.model, seed if self.model == 'M6' else None, train, test)
        if k in self.memo:
            return self.memo[k]
        self.n_fits += 1
        train = np.asarray(train, int); test = np.asarray(test, int)
        if self.model == 'M3':
            p = C.tslr_fit_predict(self.X[train], self.y[train], self.X[test])
        else:
            if C.owner_gpu_busy():
                self._release()
                print('owner GPU job detected; memory released; waiting', flush=True)
                self._to_gpu()
            p = C.eegnet_fit_predict(self.Xg, self.yg, train, test, seed)
        self.memo[k] = p
        return p

    def refit_uncached(self, train, test, seed):
        train = np.asarray(train, int); test = np.asarray(test, int)
        return C.eegnet_fit_predict(self.Xg, self.yg, train, test, seed)


def run_unit(fit, folds, seed):
    """One participant x scheme x repeat. Returns per-arm BA on the fixed scored population and secondary scores."""
    y = fit.y
    pred = {a: {} for a in ARMS}
    asrep = {'LEAKED': ([], []), 'CLEAN': ([], [])}
    exposed = []
    for fo in folds:
        test = np.unique(np.asarray(fo['test_all_N'], int))
        if not len(test):
            continue
        for arm, tr in fo['train'].items():
            if not len(tr):
                continue
            p = dict(zip(test.tolist(), fit.predict(tr, test, seed).tolist()))
            for r in fo['scored']:
                pred[arm][r] = p[r]
            if arm == 'LEAKED':
                for r in fo['test_all_N']:
                    asrep['LEAKED'][0].append(y[r]); asrep['LEAKED'][1].append(p[r])
            if arm == 'CLEAN':
                for r in fo['test_all_D']:
                    asrep['CLEAN'][0].append(y[r]); asrep['CLEAN'][1].append(p[r])
        exposed += fo['exposed']
    out = {}
    scored = sorted(pred['LEAKED'])
    for arm in ARMS:
        if pred[arm]:
            if sorted(pred[arm]) != scored:
                raise AssertionError('scored population differs between arms')
            out[f'BA_{arm}'] = C.balanced_accuracy(y[scored], [pred[arm][r] for r in scored])
            if exposed:
                out[f'BA_EXPOSED_{arm}'] = C.balanced_accuracy(y[exposed], [pred[arm][r] for r in exposed])
    for arm, (yt, yp) in asrep.items():
        if yt:
            out[f'BA_ASREPORTED_{arm}'] = C.balanced_accuracy(np.array(yt), np.array(yp))
    out['n_scored'] = len(scored); out['n_exposed'] = len(exposed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--plan', required=True, help='Read-only frozen R2 analysis plan JSON')
    ap.add_argument('--model', required=True, choices=('M3', 'M6'))
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if C.sha256_file(a.cache) != '66f9344fbc0a054253bcbbd719098bff7b4807a6d9de339724c9e1e7565a6075':
        raise SystemExit('feature cache hash mismatch')
    frozen = json.loads(Path(a.plan).read_text())
    if C.sha256_file(a.manifest) != frozen['hypothesis_A']['manifest_sha256']:
        raise SystemExit('manifest hash differs from the frozen plan')
    if a.model == 'M6':
        C.setup_torch_deterministic()
    man = json.loads(Path(a.manifest).read_text())
    fit = Fitter(a.cache, a.model)
    out = Path(a.out)
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for scheme in ('WS5F', 'LORO'):
        for s, byr in sorted(man[scheme].items()):
            for r, folds in sorted(byr.items()):
                k = f'{scheme}|{s}|{r}'
                if k in res:
                    continue
                res[k] = run_unit(fit, folds, int(r))
                out.write_text(json.dumps(res))
                print(f'{k} done fits={fit.n_fits} {time.time() - t0:.0f}s', flush=True)
    if a.model == 'M6' and 'determinism_audit' not in res:
        # pre-specified: refit the LEAKED arm of fold 0 of repeat 0 (WS-5F) for S01, S05 and S09 without the memo
        agree = []
        for s in ('S01', 'S05', 'S09'):
            fo = man['WS5F'][s]['0'][0]
            test = np.unique(np.asarray(fo['test_all_N'], int))
            p1 = fit.predict(fo['train']['LEAKED'], test, 0)
            p2 = fit.refit_uncached(fo['train']['LEAKED'], test, 0)
            agree.append(float(np.mean(p1 == p2)))
        res['determinism_audit'] = {'refits': 3, 'prediction_agreement': agree}
        out.write_text(json.dumps(res))
    res['_meta'] = {'model': a.model, 'n_fits': fit.n_fits, 'seconds': round(time.time() - t0), 'env': C.env_summary()}
    out.write_text(json.dumps(res))
    print('DONE', a.model, fit.n_fits, 'fits')


if __name__ == '__main__':
    main()
