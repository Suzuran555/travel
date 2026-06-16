"""Immutable search state for dfs_poi / future best-first frontier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class VisitingSnapshot:
    """Point-in-time copy of dfs visiting trackers for push/pop backtracking."""

    restaurants_visiting: tuple
    food_type_visiting: tuple
    restaurant_names_visiting: tuple
    attractions_visiting: tuple
    spot_type_visiting: tuple
    attraction_names_visiting: tuple

    @classmethod
    def capture(cls, agent) -> "VisitingSnapshot":
        return cls(
            restaurants_visiting=tuple(agent.restaurants_visiting),
            food_type_visiting=tuple(agent.food_type_visiting),
            restaurant_names_visiting=tuple(agent.restaurant_names_visiting),
            attractions_visiting=tuple(agent.attractions_visiting),
            spot_type_visiting=tuple(agent.spot_type_visiting),
            attraction_names_visiting=tuple(agent.attraction_names_visiting),
        )

    def restore(self, agent) -> None:
        agent.restaurants_visiting = list(self.restaurants_visiting)
        agent.food_type_visiting = list(self.food_type_visiting)
        agent.restaurant_names_visiting = list(self.restaurant_names_visiting)
        agent.attractions_visiting = list(self.attractions_visiting)
        agent.spot_type_visiting = list(self.spot_type_visiting)
        agent.attraction_names_visiting = list(self.attraction_names_visiting)


@dataclass(frozen=True)
class SearchState:
    """Compact dfs position; plan body lives in PlanPool by plan_snapshot_id."""

    day: int
    time: str
    pos: str
    visited_attr: frozenset
    visited_res: frozenset
    meals_taken: int
    plan_snapshot_id: int
    total_days: int

    @classmethod
    def from_dfs(
        cls,
        agent,
        query: Dict[str, Any],
        day: int,
        time: str,
        pos: str,
        plan: List[Dict[str, Any]],
    ) -> "SearchState":
        pool = agent._plan_pool
        snapshot_id = pool.store(plan)
        return cls(
            day=day,
            time=time or "",
            pos=pos or "",
            visited_attr=frozenset(agent.attraction_names_visiting),
            visited_res=frozenset(agent.restaurant_names_visiting),
            meals_taken=_meals_bitmap(plan, day),
            plan_snapshot_id=snapshot_id,
            total_days=query["days"],
        )

    def plan(self, pool: PlanPool) -> List[Dict[str, Any]]:
        return pool.get(self.plan_snapshot_id)


class PlanPool:
    """Shallow plan registry so SearchState stays small."""

    def __init__(self) -> None:
        self._plans: Dict[int, List[Dict[str, Any]]] = {}
        self._next_id = 0

    def reset(self) -> None:
        self._plans.clear()
        self._next_id = 0

    def store(self, plan: List[Dict[str, Any]]) -> int:
        plan_id = self._next_id
        self._next_id += 1
        self._plans[plan_id] = plan
        return plan_id

    def get(self, plan_id: int) -> Optional[List[Dict[str, Any]]]:
        return self._plans.get(plan_id)


def _meals_bitmap(plan: List[Dict[str, Any]], up_to_day: int) -> int:
    """Two bits per day: lunch (even), dinner (odd)."""
    bits = 0
    for day_idx in range(min(up_to_day + 1, len(plan))):
        for act in plan[day_idx].get("activities", []):
            act_type = act.get("type")
            if act_type == "lunch":
                bits |= 1 << (day_idx * 2)
            elif act_type == "dinner":
                bits |= 1 << (day_idx * 2 + 1)
    return bits


def ensure_day_plan(plan: List[Dict[str, Any]], day: int) -> None:
    while len(plan) <= day:
        plan.append({"day": len(plan) + 1, "activities": []})
