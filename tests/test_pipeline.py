"""Pipeline-level synthetic tests; no data downloads, no identifiable results."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shin2022_leakage.pipeline import core, manifest

class PipelineTests(unittest.TestCase):
    def test_class_matched_clones(self):
        rows = np.arange(12)
        labels = np.array([1]*6 + [2]*6)
        want = np.array([1,1,2,2,2])
        got, fb = manifest.draw_clones(np.random.default_rng(4), want, [(rows, labels)])
        self.assertEqual(fb, 0)
        self.assertEqual(np.bincount(labels[got])[1:].tolist(), [2, 3])
        self.assertEqual(len(set(got)), len(got))

    def test_sample_hash_audit(self):
        class C: pass
        cache = C()
        cache.a = {"w1_hash":np.array(["x","y","x","z","u"])}
        fo = {"scored":[0], "train":{"LEAKED":[2,3], "CLEAN":[1,3], "PLACEBO":[1,1,3]}}
        self.assertEqual(manifest.audit(cache, fo), {"LEAKED":1,"CLEAN":0,"PLACEBO":0})

    def test_paired_signflip_source_equivalence(self):
        dif = np.array([0.03] * 9)
        a = core.paired_summary(dif)
        self.assertEqual(a["n"], 9)
        self.assertEqual(a["n_positive"], 9)
        self.assertEqual(a["p_signflip_two_sided"], 2/512)

if __name__ == "__main__":
    unittest.main()
