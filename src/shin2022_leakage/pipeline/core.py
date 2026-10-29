"""Shared core of the Shin2022 R2 confirmatory round (frozen with the R2 protocol).

Statistics at the participant level (exact or seeded Monte-Carlo sign-flip, participant bootstrap, t interval, Holm),
the two frozen R1 models (M3 TS+LR, M6 EEGNet-8,2) re-exposed with an explicit training multiset, and the R1
GPU-politeness helpers. Nothing here prints or stores per-participant values; callers keep those local.
"""
import itertools
import os
from pathlib import Path

import numpy as np


BOOT_SEED = 20261027
N_BOOT = 10000
MC_FLIPS = 200000  # Monte-Carlo sign flips when 2**n is too large to enumerate (n > 20)
SESOI = 0.02


# ----------------------------------------------------------------------------------------------- statistics
def signflip_p(d, seed=BOOT_SEED):
    """Two-sided paired sign-flip p for the mean of d (exact for n <= 20, else seeded Monte Carlo incl. observed)."""
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.mean())
    if n <= 20:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        null = np.abs((signs * d[None, :]).mean(1))
        return float(np.mean(null >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    hits = 0
    for start in range(0, MC_FLIPS, 20000):
        k = min(20000, MC_FLIPS - start)
        signs = rng.choice([1.0, -1.0], size=(k, n))
        hits += int(np.sum(np.abs((signs * d[None, :]).mean(1)) >= obs - 1e-12))
    return float((hits + 1) / (MC_FLIPS + 1))


def boot_ci(v, level=0.95, seed=BOOT_SEED):
    """Percentile participant-bootstrap CI of the mean (10,000 resamples; index matrix depends only on n and seed)."""
    v = np.asarray(v, float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(N_BOOT, len(v)))
    m = v[idx].mean(1)
    a = 50 * (1 - level)
    lo, hi = np.percentile(m, [a, 100 - a])
    return [float(lo), float(hi)]


def t_ci(v, level=0.95):
    from scipy import stats
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return [float(v.mean() - q * se), float(v.mean() + q * se)]


def holm(pvals):
    keys = list(pvals)
    order = sorted(keys, key=lambda k: pvals[k])
    m, adj, run = len(keys), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        adj[k] = run
    return adj


def paired_summary(d):
    """Aggregate-only summary of per-participant paired differences d (no per-participant value is returned)."""
    d = np.asarray(d, float)
    return {'n': int(len(d)), 'mean_diff': float(d.mean()), 'ci95_boot': boot_ci(d), 'ci90_boot': boot_ci(d, 0.90),
            'ci95_t': t_ci(d), 'ci90_t': t_ci(d, 0.90), 'p_signflip_two_sided': signflip_p(d),
            'n_positive': int((d > 0).sum()), 'n_negative': int((d < 0).sum()), 'n_zero': int((d == 0).sum())}


def level_summary(v, chance=0.5):
    v = np.asarray(v, float)
    return {'n': int(len(v)), 'mean': float(v.mean()), 'ci95_boot': boot_ci(v), 'ci95_t': t_ci(v),
            'n_above_chance': int((v > chance).sum()), 'p_signflip_vs_chance_two_sided': signflip_p(v - chance)}


def classify_magnitude(s, sesoi=SESOI):
    """Pre-registered magnitude label from the t-based intervals (R1 reviewer: percentile bootstrap optimistic at small n)."""
    lo, hi = s['ci95_t']
    lo90, hi90 = s['ci90_t']
    if lo90 > -sesoi and hi90 < sesoi:
        return 'PRACTICALLY_NULL'
    if s['mean_diff'] >= sesoi and lo > 0:
        return 'RELEVANT_POSITIVE'
    if s['mean_diff'] <= -sesoi and hi < 0:
        return 'RELEVANT_NEGATIVE'
    return 'INCONCLUSIVE'


def balanced_accuracy(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, p)) if len(y) else float('nan')


# --------------------------------------------------------------------------------------------- frozen models
def tslr_fit_predict(Ctr, ytr, Cte):
    """M3 exactly as in R1 exp_core.make_model('M3')."""
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    m = make_pipeline(TangentSpace(metric='riemann'), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(Ctr, ytr)
    return m.predict(Cte)


def zscore_trials(a):
    """R1 EEGNet input normalisation: per trial and channel over time."""
    a = a.astype(np.float32)
    return (a - a.mean(2, keepdims=True)) / (a.std(2, keepdims=True) + 1e-6)


def eegnet_fit_predict(X, y, tr_rows, te_rows, seed, epochs=100, bs=64):
    """M6 training loop identical to R1 run_eegnet.fit_predict, on an explicit (possibly repeated) training multiset.

    X: torch tensor (n,1,C,T) on the GPU; y: torch int64 tensor (n,) of 0/1 labels on the GPU.
    tr_rows may contain repeated indices (duplicated training items); it is sorted by the caller.
    Returns predicted labels in {1, 2} for te_rows.
    """
    import torch
    import torch.nn as nn
    from .models import EEGNet
    dev = X.device
    torch.manual_seed(seed); np.random.seed(seed)
    model = EEGNet(C=X.shape[2], T=X.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    tr = torch.as_tensor(np.asarray(tr_rows), device=dev)
    gen = torch.Generator(device='cpu').manual_seed(seed)
    model.train()
    for _ in range(epochs):
        perm = tr[torch.randperm(len(tr), generator=gen).to(dev)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(X[b]), y[b])
            loss.backward()
            opt.step()
            model.max_norm()
    model.eval()
    with torch.no_grad():
        te = torch.as_tensor(np.asarray(te_rows), device=dev)
        out = torch.cat([model(X[te[i:i + 512]]).argmax(1) for i in range(0, len(te), 512)]) if len(te) else \
            torch.zeros(0, dtype=torch.long, device=dev)
    return out.cpu().numpy() + 1


def setup_torch_deterministic():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gpu_wait(log=print):
    """Never preempt owner jobs: block (holding no GPU memory) until 5 consecutive free polls (R1 helper)."""
    from .models import wait_for_free_gpu
    return wait_for_free_gpu(log)


def owner_gpu_busy():
    from .models import foreign_gpu_processes
    return bool(foreign_gpu_processes())


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def env_summary():
    import platform
    out = {'python': platform.python_version()}
    for mod in ('numpy', 'scipy', 'sklearn', 'pyriemann', 'torch'):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            out[mod] = None
    out['omp_threads'] = os.environ.get('OMP_NUM_THREADS')
    return out
