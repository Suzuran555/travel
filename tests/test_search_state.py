import unittest

from chinatravel.agent.UrbanTrip.search_state import (
    PlanPool,
    SearchState,
    VisitingSnapshot,
    ensure_day_plan,
    _meals_bitmap,
)


class TestSearchState(unittest.TestCase):
    def test_plan_pool_and_search_state(self):
        pool = PlanPool()
        plan = [{"day": 1, "activities": [{"type": "lunch"}]}]
        state = SearchState(
            day=0,
            time="12:00",
            pos="Hotel",
            visited_attr=frozenset(),
            visited_res=frozenset({"R1"}),
            meals_taken=_meals_bitmap(plan, 0),
            plan_snapshot_id=pool.store(plan),
            total_days=2,
        )
        self.assertEqual(state.pos, "Hotel")
        self.assertIn("R1", state.visited_res)
        self.assertEqual(state.plan(pool), plan)

    def test_visiting_snapshot_roundtrip(self):
        class Agent:
            restaurants_visiting = [1]
            food_type_visiting = ["川菜"]
            restaurant_names_visiting = ["A"]
            attractions_visiting = [2]
            spot_type_visiting = ["博物馆"]
            attraction_names_visiting = ["B"]

        agent = Agent()
        snap = VisitingSnapshot.capture(agent)
        agent.restaurant_names_visiting.append("C")
        snap.restore(agent)
        self.assertEqual(agent.restaurant_names_visiting, ["A"])

    def test_ensure_day_plan(self):
        plan = []
        ensure_day_plan(plan, 1)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[1]["day"], 2)


if __name__ == "__main__":
    unittest.main()
