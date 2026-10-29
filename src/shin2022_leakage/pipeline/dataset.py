"""Data-agnostic port of the frozen R1 cache/table construction used by R2 Hypothesis A.

Loads *caller-supplied* data locally. All cache rows and any trial/participant
identifiers remain outside this repository; do not commit the inputs.
"""
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
SUBJECTS = [f'S{i:02d}' for i in range(1, 11)]

class Cache:
    def __init__(self, cache_path, registry_path):
        c = np.load(cache_path)
        self.a = {k: c[k] for k in c.files}
        self.reg = json.loads(Path(registry_path).read_text())['rows']
        self.by_sha = {}
        for i, s in enumerate(self.a['sha']):
            self.by_sha.setdefault(s, []).append(i)
        self.path_info = {r['path']: r for r in self.reg}

    def rows_for(self, sha, rule, protocol_is_q):
        idx = np.asarray(self.by_sha[sha])
        a = self.a
        if rule == 'W1':
            m = a['w1_ok'][idx]
        elif rule == 'W1noflat':  # ablation A8: flat-channel rule disabled
            m = a['w1_base'][idx]
        elif rule == 'S2':  # W1 trials with room for Wpre, Wpost and W3 (deviation D1: excludes final trials)
            m = a['w1_ok'][idx] & a['wpre_ok'][idx] & a['w3_ok'][idx]
        elif rule == 'S2long':  # POST HOC: W1 trials whose feedback lasts >= 3 s (W3 then has no post-outcome samples)
            m = a['w1_ok'][idx] & a['w3_ok'][idx] & (a['fb_s'][idx] >= 3.0)
        elif rule == 'S2short':  # POST HOC: W1 trials with 1 <= feedback < 3 s (W3 contains post-outcome samples)
            m = a['w1_ok'][idx] & a['w3_ok'][idx] & (a['fb_s'][idx] < 3.0)
        elif rule == 'W3':
            m = a['w3_ok'][idx] & (a['closed'][idx] if protocol_is_q else True)
        else:
            raise ValueError(rule)
        return idx[m]


@dataclass
class Table:
    rows: np.ndarray      # cache row index per item
    subject: np.ndarray   # participant group
    run: np.ndarray       # run key used for run-wise grouping (path for N, sha otherwise)
    path: np.ndarray      # nominal path of the entry
    alias_copy: np.ndarray  # True if the entry is a non-canonical alias path


def entries_for(cache, protocol):
    """Return list of (path, subject, run_key, alias_copy) for a named protocol or an explicit spec dict."""
    if isinstance(protocol, dict):
        return protocol['entries']
    out = []
    for r in cache.reg:
        is_alias = r['cue_task_record_mask'] == 'ALIAS'
        if protocol == 'N':
            out.append((r['path'], r['subject'], r['path'], is_alias))
        elif r['protocols'][protocol]:
            out.append((r['path'], r['subject'], r['annex_sha256'], False))
    return out


def build_table(cache, protocol, rule):
    is_q = (protocol not in ('N', 'D', 'BR')) if isinstance(protocol, str) else protocol.get('is_q', True)
    R, S, U, P, A = [], [], [], [], []
    for path, subj, run, alias in entries_for(cache, protocol):
        sha = cache.path_info[path]['annex_sha256']
        rows = cache.rows_for(sha, rule, is_q)
        R.extend(rows); S.extend([subj] * len(rows)); U.extend([run] * len(rows))
        P.extend([path] * len(rows)); A.extend([alias] * len(rows))
    return Table(np.asarray(R, int), np.asarray(S), np.asarray(U), np.asarray(P), np.asarray(A, bool))



def labels(cache, rows):
    return cache.a["label"][rows]
