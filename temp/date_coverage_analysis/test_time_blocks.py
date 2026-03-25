#!/usr/bin/env python3
"""Lightweight tests for analyze_time_blocks helpers and smoke-run on CSV."""
import os
import subprocess
import sys
import unittest
from datetime import date

# Same directory as analyze_time_blocks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_time_blocks import consecutive_day_blocks, longest_hour_streak


class TestHelpers(unittest.TestCase):
    def test_consecutive_blocks(self):
        d = [
            date(2024, 7, 10),
            date(2024, 7, 11),
            date(2024, 7, 12),
            date(2024, 7, 14),
            date(2024, 7, 15),
        ]
        b = consecutive_day_blocks(d)
        self.assertEqual(len(b), 2)
        self.assertEqual(b[0]["start_date"], date(2024, 7, 10))
        self.assertEqual(b[0]["end_date"], date(2024, 7, 12))
        self.assertEqual(b[1]["start_date"], date(2024, 7, 14))
        self.assertEqual(b[1]["end_date"], date(2024, 7, 15))

    def test_longest_hour_streak(self):
        self.assertEqual(longest_hour_streak({8, 9, 10, 14, 15}), 3)
        self.assertEqual(longest_hour_streak(set()), 0)
        self.assertEqual(longest_hour_streak({0}), 1)


class TestSmoke(unittest.TestCase):
    def test_script_runs_on_parsed_csv(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        csv_path = os.path.join(root, "DO_lakedata_parsed.csv")
        if not os.path.isfile(csv_path):
            self.skipTest("DO_lakedata_parsed.csv not present")
        script = os.path.join(os.path.dirname(__file__), "analyze_time_blocks.py")
        out = subprocess.run(
            [sys.executable, script, csv_path, "-o", os.path.dirname(script)],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("Wrote:", out.stdout)
        daily = os.path.join(os.path.dirname(script), "daily_time_coverage.csv")
        blocks = os.path.join(os.path.dirname(script), "consecutive_day_blocks.csv")
        self.assertTrue(os.path.isfile(daily))
        self.assertTrue(os.path.isfile(blocks))


if __name__ == "__main__":
    unittest.main()
