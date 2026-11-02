"""Standalone CPU policy checks; run directly without GPU pytest conftest."""

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


def load_policy(enabled):
    path = (
        Path(__file__).resolve().parents[1]
        / "src/flag_gems/runtime/backend/_metax/ops/_mm_tuning_policy.py"
    )
    with patch.dict(
        os.environ, {"FLAGGEMS_METAX_MM_SMALL_BUCKET_M": str(int(enabled))}
    ):
        spec = importlib.util.spec_from_file_location("metax_policy_cpu", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.mm_nt_m_tuning_key


class PolicyTest(unittest.TestCase):
    def test_default_matches_previous_policy(self):
        key = load_policy(False)
        for m in range(1, 4097):
            self.assertEqual(key(m), m if m < 1024 else -((m + 31) // 32) * 32)

    def test_preserve_decode_and_large_keys(self):
        old, new = load_policy(False), load_policy(True)
        for m in list(range(1, 65)) + list(range(1024, 4097)):
            self.assertEqual(old(m), new(m))

    def test_boundaries_and_namespace(self):
        key = load_policy(True)
        self.assertEqual(key(64), 64)
        self.assertEqual(key(65), key(96))
        self.assertNotEqual(key(96), key(97))
        self.assertEqual(key(541), key(544))
        self.assertNotEqual(key(541), key(545))
        self.assertEqual(key(993), key(1023))
        self.assertNotEqual(key(1023), key(1024))
        small = {key(m) for m in range(65, 1024)}
        other = {key(m) for m in list(range(1, 65)) + list(range(1024, 8193))}
        self.assertFalse(small & other)

    def test_no_group_crosses_pruning_or_decode_boundary(self):
        key, groups = load_policy(True), {}
        for m in range(1, 4097):
            groups.setdefault(key(m), []).append(m)
        for members in groups.values():
            self.assertLessEqual(len(members), 32)
            self.assertTrue(
                all(m <= 64 for m in members)
                or all(64 < m < 1024 for m in members)
                or all(m >= 1024 for m in members)
            )


if __name__ == "__main__":
    unittest.main()
