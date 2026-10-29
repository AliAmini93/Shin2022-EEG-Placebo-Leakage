"""Synthetic end-to-end cache/registry -> placebo manifest smoke test.

Does not use actual participant data or original recordings. Distinct recordings are
invented, and the alias shares an artificial recording-hash key by construction.
"""
from pathlib import Path
import json
import tempfile
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shin2022_leakage.pipeline.dataset import Cache
from shin2022_leakage.pipeline import manifest


class ManifestIntegrationTests(unittest.TestCase):
    def test_synthetic_full_ten_subject_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            cache_sha, labels, elig, hashes = [], [], [], []
            for sid in range(1, 11):
                s = f"S{sid:02d}"
                for run in range(5):
                    sha = f"SYNTHETIC_{s}_{run:02d}"
                    path = f"{s}/BW/BW30.dat" if run == 0 else f"{s}/CV/CV{run:02d}.dat"
                    rows.append({"path": path, "subject": s, "annex_sha256": sha,
                                 "cue_task_record_mask": "STRICT",
                                 "protocols": {"N": True, "Q": True, "D": True}})
                    for trial in range(24):
                        cache_sha.append(sha)
                        labels.append(trial % 2 + 1)
                        elig.append(True)
                        hashes.append(f"UNIQUE_{s}_{run}_{trial}")
                if sid != 2:
                    rows.append({"path": f"{s}/NT/NT0.dat", "subject": s,
                                 "annex_sha256": f"SYNTHETIC_{s}_00",
                                 "cue_task_record_mask": "ALIAS",
                                 "protocols": {"N": True, "Q": False, "D": False}})
            np.savez(root / "cache.npz",
                     sha=np.asarray(cache_sha),
                     label=np.asarray(labels, dtype=np.int64),
                     w1_ok=np.asarray(elig),
                     w1_hash=np.asarray(hashes))
            (root / "registry.json").write_text(json.dumps({"rows": rows}))
            cache = Cache(root / "cache.npz", root / "registry.json")
            plan, kbar = manifest.build(cache)
            report = manifest.check_invariants(cache, plan)
            for scheme in ("WS5F", "LORO"):
                self.assertEqual(report[scheme]["audit_scored_with_twin_in_train"]["PLACEBO"], 0)
                self.assertEqual(report[scheme]["audit_scored_with_twin_in_train"]["CLEAN"], 0)
                self.assertGreater(report[scheme]["exposed_scored_items"], 0)
                self.assertEqual(report[scheme]["clone_fallbacks"], 0)
                self.assertEqual(report[scheme]["scored_items"], 10 * 5 * 24 * 5)
            self.assertGreaterEqual(kbar["kbar_WS5F_per_fold"], 1)


if __name__ == "__main__":
    unittest.main()
