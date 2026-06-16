import unittest
from unittest.mock import MagicMock

from chinatravel.agent.UrbanTrip.plan_graph import (
    activity_position,
    apply_activity_time_rules,
    clamp_activity_start_to_open_hours,
    forward_time_chain,
    incremental_space_time_ok,
    needs_transport,
    repair_activity_edge,
    repair_full_itinerary,
)


class PlanGraphTest(unittest.TestCase):
    def test_needs_transport(self):
        self.assertFalse(needs_transport("Hotel A", "Hotel A"))
        self.assertTrue(needs_transport("Hotel A", "Museum"))

    def test_same_position_breakfast_space_ok(self):
        query = {
            "people_number": 2,
            "start_city": "A",
            "target_city": "B",
            "days": 2,
        }
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel A",
                        "type": "accommodation",
                        "price": 100,
                        "cost": 100,
                        "start_time": "20:00",
                        "end_time": "24:00",
                        "transports": [],
                        "room_type": 2,
                        "rooms": 1,
                    },
                    {
                        "position": "Hotel A",
                        "type": "breakfast",
                        "price": 0,
                        "cost": 0,
                        "start_time": "08:00",
                        "end_time": "08:30",
                        "transports": [],
                    },
                ],
            }
        ]
        forward_time_chain(itinerary)
        self.assertTrue(incremental_space_time_ok(query, itinerary))

    def test_repair_activity_edge_fills_missing_transport(self):
        query = {
            "people_number": 2,
            "start_city": "A",
            "target_city": "B",
            "days": 2,
        }
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel A",
                        "type": "attraction",
                        "price": 0,
                        "cost": 0,
                        "start_time": "10:00",
                        "end_time": "11:00",
                        "transports": [],
                    },
                    {
                        "position": "Museum",
                        "type": "attraction",
                        "price": 0,
                        "cost": 0,
                        "start_time": "11:00",
                        "end_time": "12:00",
                        "transports": [],
                    },
                ],
            }
        ]
        agent = MagicMock()
        agent.innercity_transports_ranking = ["walk"]
        agent.segment_index = None
        agent.collect_innercity_transport.return_value = [
            {
                "start": "Hotel A",
                "end": "Museum",
                "mode": "walk",
                "start_time": "11:00",
                "end_time": "11:10",
                "cost": 0,
                "distance": 0.5,
                "price": 0,
            }
        ]
        self.assertTrue(repair_activity_edge(agent, query, itinerary, 0, 1))
        self.assertEqual(len(itinerary[0]["activities"][1]["transports"]), 1)
        self.assertEqual(activity_position(itinerary[0]["activities"][1]), "Museum")

    def test_repair_activity_edge_respects_attraction_open_hours(self):
        query = {
            "people_number": 1,
            "start_city": "A",
            "target_city": "B",
            "days": 2,
        }
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel",
                        "type": "breakfast",
                        "price": 0,
                        "cost": 0,
                        "start_time": "08:00",
                        "end_time": "08:30",
                        "transports": [],
                    },
                    {
                        "position": "Museum",
                        "type": "attraction",
                        "price": 0,
                        "cost": 0,
                        "start_time": "10:00",
                        "end_time": "11:30",
                        "transports": [],
                        "tickets": 1,
                    },
                ],
            }
        ]
        agent = MagicMock()
        agent.innercity_transports_ranking = ["walk"]
        agent.segment_index = None
        agent.memory = {
            "attractions": __import__("pandas").DataFrame(
                [
                    {
                        "name": "Museum",
                        "opentime": "10:00",
                        "endtime": "18:00",
                        "price": 0,
                    }
                ]
            )
        }
        agent.collect_innercity_transport.return_value = [
            {
                "start": "Hotel",
                "end": "Museum",
                "mode": "walk",
                "start_time": "08:30",
                "end_time": "08:58",
                "cost": 0,
                "distance": 0.5,
                "price": 0,
            }
        ]
        self.assertTrue(repair_activity_edge(agent, query, itinerary, 0, 1))
        museum = itinerary[0]["activities"][1]
        self.assertEqual(museum["start_time"], "10:00")
        self.assertEqual(museum["end_time"], "11:30")

    def test_clamp_activity_start_to_open_hours(self):
        act = {
            "type": "attraction",
            "start_time": "08:58",
            "end_time": "11:30",
        }
        clamp_activity_start_to_open_hours(act, "10:00", "21:00")
        self.assertEqual(act["start_time"], "10:00")

    def test_lunch_meal_time_rules(self):
        act = {
            "type": "lunch",
            "start_time": "10:45",
            "end_time": "11:45",
        }
        clamp_activity_start_to_open_hours(act, "10:00", "22:00")
        self.assertEqual(act["start_time"], "11:00")

    def test_repair_keeps_existing_transport_when_aligned(self):
        query = {
            "people_number": 1,
            "start_city": "A",
            "target_city": "B",
            "days": 2,
        }
        transports = [
            {
                "start": "Hotel",
                "end": "Museum",
                "mode": "metro",
                "start_time": "08:30",
                "end_time": "08:58",
                "cost": 3,
                "price": 3,
                "tickets": 1,
            }
        ]
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel",
                        "type": "breakfast",
                        "price": 0,
                        "cost": 0,
                        "start_time": "08:00",
                        "end_time": "08:30",
                        "transports": [],
                    },
                    {
                        "position": "Museum",
                        "type": "attraction",
                        "price": 0,
                        "cost": 0,
                        "start_time": "10:00",
                        "end_time": "11:30",
                        "transports": transports,
                        "tickets": 1,
                    },
                ],
            }
        ]
        agent = MagicMock()
        agent.innercity_transports_ranking = ["walk"]
        agent.segment_index = None
        agent.memory = {
            "attractions": __import__("pandas").DataFrame(
                [
                    {
                        "name": "Museum",
                        "opentime": "10:00",
                        "endtime": "18:00",
                        "price": 0,
                    }
                ]
            )
        }
        self.assertTrue(repair_activity_edge(agent, query, itinerary, 0, 1))
        agent.collect_innercity_transport.assert_not_called()
        self.assertIs(itinerary[0]["activities"][1]["transports"], transports)

    def test_forward_time_chain_fixes_late_accommodation(self):
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel",
                        "type": "accommodation",
                        "price": 100,
                        "cost": 100,
                        "start_time": "27:22",
                        "end_time": "24:00",
                        "transports": [
                            {
                                "start": "A",
                                "end": "B",
                                "mode": "metro",
                                "start_time": "25:16",
                                "end_time": "27:13",
                                "cost": 0,
                                "distance": 1,
                                "price": 0,
                            }
                        ],
                        "room_type": 1,
                        "rooms": 1,
                    }
                ],
            }
        ]
        forward_time_chain(itinerary)
        act = itinerary[0]["activities"][0]
        self.assertEqual(act["transports"][0]["end_time"], "27:13")
        self.assertEqual(act["start_time"], "23:00")
        self.assertEqual(act["end_time"], "24:00")

    def test_repair_full_itinerary(self):
        query = {
            "people_number": 2,
            "start_city": "A",
            "target_city": "B",
            "days": 1,
        }
        itinerary = [
            {
                "day": 1,
                "activities": [
                    {
                        "position": "Hotel A",
                        "type": "breakfast",
                        "price": 0,
                        "cost": 0,
                        "start_time": "08:00",
                        "end_time": "08:00",
                        "transports": [],
                    }
                ],
            }
        ]
        agent = MagicMock()
        agent.innercity_transports_ranking = ["walk"]
        agent.segment_index = None
        forward_time_chain(itinerary)
        self.assertNotEqual(
            itinerary[0]["activities"][0]["start_time"],
            itinerary[0]["activities"][0]["end_time"],
        )


if __name__ == "__main__":
    unittest.main()
