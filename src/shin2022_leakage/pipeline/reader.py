"""Minimal, dependency-light BCI2000 .dat reader (header, parameters, states, signal).

Format (BCI2000 file format spec, v1.1): an ASCII header whose first line declares HeaderLen, SourceCh,
StatevectorLen and DataFormat, followed by a [ State Vector Definition ] section and a
[ Parameter Definition ] section; then binary records, one per sample: SourceCh values of DataFormat,
followed by StatevectorLen bytes of packed state bits (little-endian bit order).

Privacy: parameters listed in PRIVATE_PARAMETERS (and any string-typed parameter routed through
`identity_tokens`) are held only as keyed HMAC digests inside this process. Raw values are never
returned by the public API of this module.
"""
import hashlib
import hmac
import os
import re
from dataclasses import dataclass, field

import numpy as np

PRIVATE_PARAMETERS = ('SubjectName', 'DataDirectory', 'DataFile', 'SubjectSession', 'StorageTime_Raw')
_FMT = {'int16': '<i2', 'int32': '<i4', 'float32': '<f4'}
_SESSION_KEY = os.urandom(32)  # per-process key: digests are comparable within one run only
_PRIVATE_STORE = {}  # id(Header) -> {param: raw value}; module-private, never returned or serialized
HEADER_LIMIT = 8 * 1024 * 1024


def unpct(tok):
    if tok == '%':
        return ''
    return re.sub(r'%([0-9A-Fa-f]{2})', lambda m: chr(int(m.group(1), 16)), tok)


def _digest(value):
    return hmac.new(_SESSION_KEY, value.encode('utf-8', 'surrogateescape'), hashlib.sha256).hexdigest()


@dataclass
class StateDef:
    name: str
    length: int
    value: int
    byte: int
    bit: int


@dataclass
class Param:
    section: str
    dtype: str
    name: str
    value: object  # str | list[str] | list[list[str]]
    shape: tuple = ()


@dataclass
class Header:
    path: str
    header_len: int
    source_ch: int
    statevector_len: int
    data_format: str
    version: str
    states: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    param_counts: dict = field(default_factory=dict)
    header_sha256: str = ''
    _private_digests: dict = field(default_factory=dict, repr=False)

    # ---- public helpers -------------------------------------------------------------------------
    def scalar(self, name, default=None):
        p = self.params.get(name)
        if p is None:
            return default
        v = p.value
        if isinstance(v, list):
            return v[0] if v and not isinstance(v[0], list) else default
        return v

    def number(self, name, default=None):
        """Numeric value with BCI2000 unit suffix stripped; returns (value, unit)."""
        tok = self.scalar(name)
        if tok is None:
            return default, None
        m = re.match(r'^(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([A-Za-z%]*)$', str(tok))
        if not m:
            return default, None
        return float(m.group(1)), (m.group(2) or '')

    def seconds(self, name, block_seconds=None):
        """Duration parameter in seconds (bare numbers are sample blocks in BCI2000)."""
        v, unit = self.number(name)
        if v is None:
            return None
        if unit == 's':
            return v
        if unit == 'ms':
            return v / 1000.0
        if unit in ('', None) and block_seconds is not None:
            return v * block_seconds
        return None

    def numeric_list(self, name):
        p = self.params.get(name)
        if p is None or not isinstance(p.value, list):
            return None
        out = []
        for t in p.value:
            try:
                out.append(float(re.sub(r'[A-Za-z]+$', '', t)))
            except (TypeError, ValueError):
                return None
        return out

    def matrix(self, name):
        p = self.params.get(name)
        if p is None or p.dtype != 'matrix':
            return None
        return p.value

    def private_digest(self, name):
        """Keyed digest of a private parameter (comparable within this process; never exported raw)."""
        return self._private_digests.get(name)

    @property
    def sampling_rate(self):
        v, unit = self.number('SamplingRate')
        if v is None:
            return None
        return v * 1000.0 if unit == 'kHz' else v

    @property
    def block_size(self):
        v, _ = self.number('SampleBlockSize')
        return int(v) if v is not None else None

    @property
    def block_seconds(self):
        fs, bs = self.sampling_rate, self.block_size
        return bs / fs if fs and bs else None


def _read_raw_header(fh):
    first = fh.read(1024)
    line = first.split(b'\n', 1)[0]
    if line.startswith(b'BCI2000V='):
        m = re.search(rb'HeaderLen\s*=\s*(\d+)', line)
    else:  # pre-1.1 files start with "HeaderLen="
        m = re.search(rb'^HeaderLen\s*=\s*(\d+)', line)
    if not m:
        raise ValueError('not a BCI2000 file: HeaderLen missing in first line')
    n = int(m.group(1))
    if not 32 <= n <= HEADER_LIMIT:
        raise ValueError('implausible HeaderLen')
    fh.seek(0)
    raw = fh.read(n)
    if len(raw) != n:
        raise ValueError('truncated BCI2000 header')
    return raw


def _take_dim(tokens, i):
    """A list/matrix dimension is either an integer or a { label ... } list."""
    if i >= len(tokens):
        return 0, i
    if tokens[i] == '{':
        j = tokens.index('}', i)
        return j - i - 1, j + 1
    try:
        return int(tokens[i]), i + 1
    except ValueError:
        return 0, i + 1


def _parse_param_tokens(dtype, vals):
    if dtype.endswith('list'):
        n, i = _take_dim(vals, 0)
        return vals[i:i + n], (n,)
    if dtype == 'matrix':
        nr, i = _take_dim(vals, 0)
        nc, i = _take_dim(vals, i)
        cells, j = [], i
        for _ in range(nr * nc):
            if j < len(vals) and vals[j] == '{':  # nested sub-matrix cell: keep as opaque marker
                depth, k = 0, j
                while k < len(vals):
                    depth += vals[k] == '{'
                    depth -= vals[k] == '}'
                    k += 1
                    if depth == 0:
                        break
                cells.append('{...}')
                j = k
            else:
                cells.append(vals[j] if j < len(vals) else '')
                j += 1
        rows = [[unpct(c) for c in cells[r * nc:(r + 1) * nc]] for r in range(nr)]
        return rows, (nr, nc)
    return (vals[0] if vals else ''), ()


def parse_header_bytes(raw, path='<bytes>'):
    text = raw.decode('latin1').replace('\r', '')
    lines = text.split('\n')
    first = lines[0]
    kv = {k.casefold(): v for k, v in re.findall(r'(\w+)\s*=\s*(\S+)', first)}
    sv = kv.get('statevectorlen') or kv.get('statevectorlength')
    h = Header(path=path, header_len=int(kv['headerlen']), source_ch=int(kv.get('sourcech', 0)),
               statevector_len=int(sv) if sv else 0, data_format=kv.get('dataformat', 'int16'),
               version=kv.get('bci2000v', 'pre-1.1'))
    h.header_sha256 = hashlib.sha256(raw).hexdigest()
    _PRIVATE_STORE[id(h)] = {}
    section = None
    for line in lines[1:]:
        s = line.strip().strip('\x00')
        g = re.fullmatch(r'\[\s*([^\]]+?)\s*\]', s)
        if g:
            section = g.group(1).casefold().replace(' ', '')
            continue
        if not s or s.startswith('//'):
            continue
        if section == 'statevectordefinition':
            t = s.split()
            if len(t) >= 5:
                try:
                    h.states[t[0]] = StateDef(t[0], int(t[1]), int(t[2]), int(t[3]), int(t[4]))
                except ValueError:
                    pass
        elif section == 'parameterdefinition':
            body = s.split('//', 1)[0]
            t = body.split()
            if len(t) < 3 or not t[2].endswith('='):
                continue
            sec, dtype, name = unpct(t[0]), t[1], t[2][:-1]
            h.param_counts[name] = h.param_counts.get(name, 0) + 1
            if h.param_counts[name] > 1:
                continue  # first definition wins; duplicates are reported via param_counts
            value, shape = _parse_param_tokens(dtype, t[3:])
            if name in PRIVATE_PARAMETERS:
                raw_v = unpct(value) if isinstance(value, str) else repr(value)
                h._private_digests[name] = _digest(raw_v)
                _PRIVATE_STORE.setdefault(id(h), {})[name] = raw_v
                value = '<redacted>'
            elif isinstance(value, str):
                value = unpct(value)
            elif dtype.endswith('list'):
                value = [unpct(x) for x in value]
            h.params[name] = Param(sec, dtype, name, value, shape)
    return h


def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def private_similarity(h1, h2, name='SubjectName'):
    """1 - normalized edit distance between two private values (only this number leaves the module)."""
    a = _PRIVATE_STORE.get(id(h1), {}).get(name)
    b = _PRIVATE_STORE.get(id(h2), {}).get(name)
    if a is None or b is None:
        return None
    if not a and not b:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


def read_header(path):
    with open(path, 'rb') as fh:
        raw = _read_raw_header(fh)
    return parse_header_bytes(raw, path=str(path))


def record_geometry(h, file_size):
    rec = h.source_ch * np.dtype(_FMT[h.data_format]).itemsize + h.statevector_len
    payload = file_size - h.header_len
    return rec, payload // rec if rec else 0, (payload % rec if rec else payload)


def read_data(path, h=None, calibrate=True):
    """Return (signal [n_samples x n_ch] float64 in physical units, states dict name->int64 array)."""
    h = h or read_header(path)
    size = os.path.getsize(path)
    rec, n, rem = record_geometry(h, size)
    dt = np.dtype([('sig', _FMT[h.data_format], (h.source_ch,)), ('st', 'u1', (h.statevector_len,))])
    arr = np.fromfile(path, dtype=dt, count=n, offset=h.header_len)
    sig = arr['sig'].astype(np.float64)
    if calibrate:
        off = h.numeric_list('SourceChOffset')
        gain = h.numeric_list('SourceChGain')
        if off and len(off) == h.source_ch:
            sig = sig - np.asarray(off)
        if gain and len(gain) == h.source_ch:
            sig = sig * np.asarray(gain)
    st = arr['st']
    states = {}
    for name, d in h.states.items():
        nbytes = (d.bit + d.length + 7) // 8
        acc = np.zeros(n, dtype=np.uint64)
        for k in range(nbytes):
            if d.byte + k < h.statevector_len:
                acc |= st[:, d.byte + k].astype(np.uint64) << np.uint64(8 * k)
        states[name] = ((acc >> np.uint64(d.bit)) & np.uint64((1 << d.length) - 1)).astype(np.int64)
    return sig, states


def write_bci2000(path, signal, states, params, data_format='float32'):
    """Writer for deterministic tests: signal [n x ch], states {name: (length, array)}, params lines."""
    n, ch = signal.shape
    defs, bitpos = [], 0
    for name, (length, _) in states.items():
        defs.append((name, length, bitpos // 8, bitpos % 8))
        bitpos += length
    svlen = (bitpos + 7) // 8
    body = '[ State Vector Definition ]\n' + ''.join(f'{nm} {ln} 0 {b} {bt}\n' for nm, ln, b, bt in defs)
    body += '[ Parameter Definition ]\n' + ''.join(l.rstrip('\n') + '\n' for l in params) + '\n'
    for _ in range(3):
        first = f'BCI2000V= 1.1 HeaderLen= {0:8d} SourceCh= {ch} StatevectorLen= {svlen} DataFormat= {data_format}\n'
        total = len(first) + len(body)
        first = f'BCI2000V= 1.1 HeaderLen= {total:8d} SourceCh= {ch} StatevectorLen= {svlen} DataFormat= {data_format}\n'
    hdr = (first + body).encode('latin1')
    packed = np.zeros((n, svlen), dtype=np.uint8)
    for (name, length, b, bt), (_, (_, arr)) in zip(defs, states.items()):
        vals = np.asarray(arr, dtype=np.uint64)
        bit0 = b * 8 + bt
        for k in range(length):
            bitv = ((vals >> np.uint64(k)) & np.uint64(1)).astype(np.uint8)
            pos = bit0 + k
            packed[:, pos // 8] |= (bitv << (pos % 8)).astype(np.uint8)
    dt = np.dtype([('sig', _FMT[data_format], (ch,)), ('st', 'u1', (svlen,))])
    rec = np.zeros(n, dtype=dt)
    rec['sig'] = signal
    rec['st'] = packed
    with open(path, 'wb') as fh:
        fh.write(hdr)
        fh.write(rec.tobytes())
