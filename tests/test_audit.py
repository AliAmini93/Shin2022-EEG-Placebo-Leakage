"""Small deterministic tests. No EEG, participant data, GPU or downloads."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shin2022_leakage.audit import load_results, validate_results, describe
from shin2022_leakage.cli import main
from shin2022_leakage.stats import signflip_p, holm, paired_summary, summarize_contrasts


class AuditTests(unittest.TestCase):
    def test_frozen_aggregate_receipt(self):
        self.assertEqual(validate_results(load_results()), [])

    def test_corrupted_primary_mean_is_rejected(self):
        data = copy.deepcopy(load_results())
        data["primary_leaked_minus_placebo"]["EEGNet_WS5F"]["mean_pp"] += 1.0
        self.assertIn("EEGNet_WS5F: reference mean mismatch", validate_results(data))

    def test_frozen_scope_exposure_not_independent_people(self):
        r = load_results()["scope"]
        self.assertEqual(r["unique_scored_trials"] * r["evaluation_repeats"],
                         r["scored_instances_per_scheme"])
        self.assertLess(r["ws5f_scored_instances_exposed"], r["scored_instances_per_scheme"])

    def test_exact_signflip_and_holm(self):
        self.assertEqual(signflip_p([0.02] * 9), 2 / 512)
        self.assertEqual(holm({"a": 2 / 512, "b": 2 / 512, "c": 2 / 512, "d": 2 / 512}),
                         dict.fromkeys(("a", "b", "c", "d"), 8 / 512))

    def test_paired_summary_is_aggregate(self):
        r = paired_summary([0.01] * 9)
        self.assertEqual(r["positive"], 9)
        self.assertAlmostEqual(r["mean_diff"], 0.01)
        self.assertEqual(r["p_exact"], 2 / 512)
        self.assertEqual(r["ci95_t"], [0.01, 0.01])
        self.assertFalse(any(isinstance(v, list) and len(v) == 9 for v in r.values()))

    def test_recompute_input_validation(self):
        with self.assertRaises(ValueError):
            summarize_contrasts({"bad": [float("nan"), 0.01]})

    def test_cli_validate(self):
        self.assertEqual(main(["validate"]), 0)
        self.assertIn("EEGNet", describe(load_results()))


if __name__ == "__main__":
    unittest.main()
