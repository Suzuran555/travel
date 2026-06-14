import json
import os
import tempfile
import unittest

import pandas as pd

from chinatravel.agent.UrbanTrip.segment_index import SegmentIndex
from scripts.build_urbantrip_segments import write_jsonl


class SegmentIndexTest(unittest.TestCase):
    def test_missing_files_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index = SegmentIndex(lang="en", segment_dir=tmpdir)
            df = pd.DataFrame(
                [
                    {"BeginTime": "09:00", "Cost": 200, "FlightID": "F2"},
                    {"BeginTime": "08:00", "Cost": 100, "FlightID": "F1"},
                ]
            )
            self.assertEqual(index.rank_intercity({}, "go", df), [0, 1])
            self.assertIsNone(index.get_intracity_route("X", "A", "B", "walk", "08:00", 2))

    def test_intercity_ranking_honors_must_transport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "from_city": "A",
                    "to_city": "B",
                    "mode": "airplane",
                    "id": "F1",
                    "from": "A Airport",
                    "to": "B Airport",
                    "begin": "09:00",
                    "end": "10:00",
                    "cost": 300,
                    "duration": 60,
                    "base_score": 100,
                    "score_text": "airplane cheap",
                },
                {
                    "from_city": "A",
                    "to_city": "B",
                    "mode": "train",
                    "id": "T1",
                    "from": "A Station",
                    "to": "B Station",
                    "begin": "08:00",
                    "end": "12:00",
                    "cost": 100,
                    "duration": 240,
                    "base_score": 50,
                    "score_text": "train cheap",
                },
            ]
            write_jsonl(os.path.join(tmpdir, "intercity_segments.jsonl"), rows)
            index = SegmentIndex(lang="en", segment_dir=tmpdir)
            df = pd.DataFrame(
                [
                    {"BeginTime": "09:00", "Cost": 300, "FlightID": "F1"},
                    {"BeginTime": "08:00", "Cost": 100, "TrainID": "T1"},
                ]
            )
            ranked = index.rank_intercity({"must_depart_transport": ["train"]}, "go", df)
            self.assertEqual(ranked[0], 1)

    def test_intracity_route_recomputes_costs_for_people(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {
                    "city": "A",
                    "start": "Hotel",
                    "end": "Airport",
                    "start_type": "hotel",
                    "end_type": "terminal",
                    "mode": "metro",
                    "route": [
                        {
                            "start": "Hotel",
                            "end": "Station A",
                            "mode": "walk",
                            "cost": 0,
                            "distance": 0.5,
                            "duration": 6,
                        },
                        {
                            "start": "Station A",
                            "end": "Station B",
                            "mode": "metro",
                            "cost": 5,
                            "distance": 10,
                            "duration": 20,
                        },
                    ],
                    "cost": 5,
                    "distance": 10.5,
                    "duration": 26,
                    "base_score": 126,
                    "score_text": "hotel airport metro cheap",
                }
            ]
            write_jsonl(os.path.join(tmpdir, "intracity_segments.jsonl"), rows)
            index = SegmentIndex(lang="en", segment_dir=tmpdir)
            route = index.get_intracity_route("A", "Hotel", "Airport", "metro", "08:00", 3)
            self.assertEqual(route[0]["start_time"], "08:00")
            self.assertEqual(route[0]["end_time"], "08:06")
            self.assertEqual(route[1]["tickets"], 3)
            self.assertEqual(route[1]["cost"], 15)

    def test_write_jsonl_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "rows.jsonl")
            write_jsonl(path, [{"b": 2}, {"a": 1}])
            with open(path, encoding="utf-8") as f:
                first = f.read()
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            with open(path, encoding="utf-8") as f:
                second = f.read()
            self.assertEqual(first, second)
            self.assertEqual([json.loads(line) for line in first.splitlines()], [{"a": 1}, {"b": 2}])


if __name__ == "__main__":
    unittest.main()
