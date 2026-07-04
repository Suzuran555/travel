import sys
import os
import time
import argparse
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import random
import re
import ast
from geopy.distance import geodesic

sys.path.append("./../../../")
project_root_path = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.agent.base import AbstractAgent, BaseAgent
from chinatravel.agent.UrbanTrip.utils import (
    time_compare_if_earlier_equal,
    calc_cost_from_itinerary_wo_intercity,
    add_time_delta,
    get_time_delta,
    TimeOutError,
    clamp_time_to_day_end,
    time_to_minutes,
)
from chinatravel.agent.UrbanTrip.segment_index import SegmentIndex
from chinatravel.agent.UrbanTrip.plan_graph import (
    forward_time_chain,
    incremental_commonsense_ok,
    incremental_space_time_ok,
    repair_full_itinerary,
    sync_itinerary_commonsense,
)
from chinatravel.agent.UrbanTrip.search_state import PlanPool, SearchState, VisitingSnapshot, ensure_day_plan
from chinatravel.agent.UrbanTrip.dfs_search import (
    arrive_leave_violated,
    arrived_time,
    dfs_log,
    filter_open_at_time,
    flatten_visiting_indices,
    is_closed_at_arrival,
    iterate_transports,
    rank_poi_dataframe,
    transport_rules_violated,
    transports_ranking_to,
)

# from chinatravel.eval.utils import load_json_file, validate_json, save_json_file
from chinatravel.data.load_datasets import load_json_file, save_json_file
from chinatravel.agent.utils import Logger
from chinatravel.symbol_verification.commonsense_constraint import (
    func_commonsense_constraints,
)
from chinatravel.symbol_verification.hard_constraint import (
    get_symbolic_concepts,
    evaluate_constraints,
    evaluate_constraints_py,
    normalize_hard_logic_constraint,
)
from chinatravel.symbol_verification.preference import evaluate_preference_py
from chinatravel.environment.tools.poi.apis import Poi

from chinatravel.agent.nesy_verifier.verifier.commonsense_constraint_nl import collect_commonsense_constraints_error
from chinatravel.agent.nesy_verifier.verifier.personal_constraint_nl import collect_personal_error

from chinatravel.symbol_verification.concept_func import *
from chinatravel.agent.nesy_agent.nl2sl_hybrid import nl2sl_reflect
from copy import deepcopy


class UrbanTripOptimizedV6(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="TPC", **kwargs)
        cache_dir = kwargs.get("cache_dir", "cache/")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        self.cache_dir = cache_dir

        self.method = kwargs["method"]
        self.memory = {}
        external_timeout = kwargs.get("external_timeout")
        self.TIME_CUT = external_timeout if external_timeout is not None else (60 * 5 - 10)
        self.EXTERNAL_TIMEOUT_BUFFER = 60  # leave enough time for repair/fallback before the 330s outer timeout
        self.top_k_candidates = kwargs.get("top_k_candidates", 50)  # 候选截断数，提速后放宽以缓解"正解排第 n+1 不可达"
        self.enable_dav_postprocess = kwargs.get("enable_dav_postprocess", True)
        self.dav_postprocess_seconds = max(0.0, float(kwargs.get("dav_postprocess_seconds", 14.0)))
        self.dav_postprocess_max_attempts = max(0, int(kwargs.get("dav_postprocess_max_attempts", 96)))
        self.dav_postprocess_candidates_per_gap = max(
            1, int(kwargs.get("dav_postprocess_candidates_per_gap", 5))
        )
        self.dav_postprocess_target_per_day = max(
            1, int(kwargs.get("dav_postprocess_target_per_day", 4))
        )
        self.dav_postprocess_attraction_minutes = max(
            30, int(kwargs.get("dav_postprocess_attraction_minutes", 60))
        )
        # Gap-fill scan order: scanning day-by-day, front-to-back, spends the
        # fixed attempt/time budget above on small early-day gaps before ever
        # reaching the (usually much larger) last-day tail gap before the
        # return intercity transport. Default on: both DAV and meal
        # postprocess visit gaps largest-first instead (the accept-only-if-
        # not-worse validation is unchanged). Validated on the full 1000-query
        # phase1 set with zero FPR/C-LPR/MicEPR regression and DAV +6.6 vs the
        # prior day-order scan (still overridable for A/B via kwargs).
        self.dav_postprocess_prioritize_large_gaps = kwargs.get(
            "dav_postprocess_prioritize_large_gaps", True
        )
        self.require_complete_daily_meals = bool(
            kwargs.get("require_complete_daily_meals", False)
        )
        self.enable_meal_postprocess = kwargs.get("enable_meal_postprocess", True)
        self.meal_postprocess_seconds = max(0.0, float(kwargs.get("meal_postprocess_seconds", 10.0)))
        self.meal_postprocess_max_attempts = max(0, int(kwargs.get("meal_postprocess_max_attempts", 120)))
        self.meal_postprocess_candidates_per_gap = max(
            1, int(kwargs.get("meal_postprocess_candidates_per_gap", 8))
        )
        self.meal_postprocess_minutes = max(
            30, int(kwargs.get("meal_postprocess_minutes", 30))
        )
        # Compact dwell: shrink the *default* attraction/meal visit duration
        # used during the main DFS search (POIs with an explicit hard minimum
        # -- activities_stay_time_dict / activities_leave_time_dict -- already
        # take max(default, required) downstream, so they are unaffected).
        # Commonsense has no minimum-stay check; the 90/60-minute defaults are
        # this agent's own convention, and shortening them frees mid-day slack
        # for more POIs. Gated; default keeps the legacy 90/60 minutes.
        self.enable_compact_dwell = kwargs.get("enable_compact_dwell", False)
        self.compact_attraction_minutes = max(
            15, int(kwargs.get("compact_attraction_minutes", 30))
        )
        self.compact_meal_minutes = max(
            15, int(kwargs.get("compact_meal_minutes", 30))
        )
        self.debug = kwargs.get("debug", False)
        self.use_llm_dynamic_weights = kwargs.get(
            "use_llm_dynamic_weights",
            kwargs.get("llm_dynamic_weights", False),
        )
        self.dynamic_weight_debug = kwargs.get("dynamic_weight_debug", False)
        self._dynamic_weight_cache = {}
        self._dynamic_ranking_context = None
        self._candidate_static_cache = {}
        self.too_many_backtrack = False
        self.lang = kwargs.get("lang", "zh")
        self.poi_search = Poi(lang=self.lang)
        self._distance_cache = {}  # (city, frozenset({start, end})) -> 球面距离，单次搜索内复用
        # v6 default: off. Full 1000-query A/B (v14 base flags, segments on
        # vs off) showed the precomputed segment graph both fixes and
        # introduces failures -- net FPR -0.2 but ATT +9.1 (likely stale/
        # suboptimal edges skewing hotel & intercity ranking), so this is kept
        # a class-default divergence from v5 pending root-cause triage of the
        # newly-introduced failures (still overridable via kwargs for A/B).
        self.use_segments = kwargs.get("use_segments", False)
        self.segment_top_k = kwargs.get("segment_top_k", 50)
        self.segment_index = None
        if self.use_segments:
            self.segment_index = SegmentIndex(
                lang=self.lang,
                segment_dir=kwargs.get("segment_dir"),
                top_k=self.segment_top_k,
            )
        # Phase D: when use_segments=False, agent.segment_index is None so
        # rank_poi_dataframe's fallback sort has no notion of pending
        # must-visit names/types (that logic otherwise lives entirely inside
        # SegmentIndex.rank_poi). This flag re-applies the same coverage/
        # order-block bonuses via a segment-independent scorer (crude
        # distance standing in for route_cost). No-op whenever segments are
        # on. Default off pending A/B; v6 has segments off by default so this
        # is the flag that actually matters here.
        self.enable_segment_independent_must_rank = kwargs.get(
            "enable_segment_independent_must_rank", False
        )

        # ----- Geo-aware bundle search (all default off for safe A/B) -----
        # Anchor hotel ranking on must-visit POIs, not just intercity terminals.
        self.geo_anchor_must = kwargs.get("geo_anchor_must", True)
        # Estimated taxi RMB per km, used as a fallback when a segment edge is
        # missing so far-away must POIs still incur a geographic penalty.
        self.taxi_per_km_est = float(kwargs.get("taxi_per_km_est", 4.0))
        # Deterministic bundle scorer + top-N beam over (go, back, hotel).
        self.use_bundle_search = kwargs.get("use_bundle_search", False)
        self.bundle_go_top = int(kwargs.get("bundle_go_top", 12))
        self.bundle_back_top = int(kwargs.get("bundle_back_top", 12))
        self.bundle_hotel_top = int(kwargs.get("bundle_hotel_top", 20))
        self.bundle_top_n = int(kwargs.get("bundle_top_n", 20))
        self.bundle_dfs_top = int(kwargs.get("bundle_dfs_top", 3))
        # LLM rerank of the deterministic top-N (rerank only, never feasibility).
        self.use_llm_bundle_rerank = kwargs.get("use_llm_bundle_rerank", False)
        self._bundle_rerank_cache = {}
        # Stability extras (Section 6), each independently gated.
        self.enable_fallback_hard_repair = kwargs.get("enable_fallback_hard_repair", False)
        # Budget drop repair: when the final plan exceeds total_cost/innercity
        # budget, drop the non-must attraction with the priciest inbound leg
        # until the budget line flips (gated; only accepts strict improvements).
        self.enable_budget_drop_repair = kwargs.get("enable_budget_drop_repair", False)
        self.budget_drop_max = int(kwargs.get("budget_drop_max", 6))
        self.enable_dfs_memoization = kwargs.get("enable_dfs_memoization", False)
        self.enable_dynamic_top_k = kwargs.get("enable_dynamic_top_k", False)
        self.dynamic_top_k_pending = int(kwargs.get("dynamic_top_k_pending", 8))
        self.enable_metro_only_prune = kwargs.get("enable_metro_only_prune", False)
        # TF-IDF semantic weight for POI ranking (0 keeps legacy behavior; only
        # meaningful once must POIs are tagged into segment score_text).
        self.rank_semantic_weight = float(kwargs.get("rank_semantic_weight", 0.0))
        self._dfs_state_seen = set()

        self.visited_attractions = set()
        self.visited_restaurants = set()

    def _public_query(self, query):
        return {
            key: deepcopy(value)
            for key, value in query.items()
            if not str(key).startswith("_urbantrip_")
        }

    def _normalize_transport_rules(self):
        if self.transport_rules_by_distance is None:
            return
        if isinstance(self.transport_rules_by_distance, str):
            self.transport_rules_by_distance = json.loads(self.transport_rules_by_distance)
        elif isinstance(self.transport_rules_by_distance, dict):
            self.transport_rules_by_distance = [self.transport_rules_by_distance]
        elif isinstance(self.transport_rules_by_distance, list):
            self.transport_rules_by_distance = [
                rule for rule in self.transport_rules_by_distance if isinstance(rule, dict)
            ]

    def _install_constraint_state(self, constraints_json, requirement_list):
        self.all_satisfy = constraints_json.get("all_satisfy", None)

        self.must_see_attraction = constraints_json.get("must_see_attraction", None)
        self.must_see_attraction_type = constraints_json.get("must_see_attraction_type", None)
        self.must_not_see_attraction = constraints_json.get("must_not_see_attraction", None)
        self.must_not_see_attraction_type = constraints_json.get("must_not_see_attraction_type", None)
        self.only_free_attractions = constraints_json.get("only_free_attractions", None)

        self.must_visit_restaurant = constraints_json.get("must_visit_restaurant", None)
        self.must_visit_restaurant_type = constraints_json.get("must_visit_restaurant_type", None)
        self.must_not_visit_restaurant = constraints_json.get("must_not_visit_restaurant", None)
        self.must_not_visit_restaurant_type = constraints_json.get("must_not_visit_restaurant_type", None)

        # DSL `{...} & set` means "any one of" (disjunction); `{...} <= set` means
        # "all of" (conjunction). These flags let check_constraint pick intersection
        # vs subset accordingly. Default False keeps the original subset (all) behavior.
        self.must_see_attraction_type_match_any = constraints_json.get("attraction_type_match_any", False)
        self.must_visit_restaurant_type_match_any = constraints_json.get("restaurant_type_match_any", False)

        self.activities_stay_time_dict = constraints_json.get("activities_stay_time_dict", None)
        self.activities_arrive_time_dict = constraints_json.get("activities_arrive_time_dict", None)
        self.activities_leave_time_dict = constraints_json.get("activities_leave_time_dict", None)

        self.must_live_hotel = constraints_json.get("must_live_hotel", None)
        self.must_not_live_hotel = constraints_json.get("must_not_live_hotel", None)
        self.must_live_hotel_feature = constraints_json.get("must_live_hotel_feature", None)
        self.must_not_live_hotel_feature = constraints_json.get(
            "must_not_live_hotel_feature", None
        )
        self.must_live_hotel_location_limit = constraints_json.get("must_live_hotel_location_limit", None)
        self.bed_number = constraints_json.get("bed_number", None)
        self.room_number = constraints_json.get("room_number", None)
        self.must_visit_order = constraints_json.get("must_visit_order", None)

        self.must_innercity_transport = constraints_json.get("must_innercity_transport", None)
        self.must_not_innercity_transport = constraints_json.get("must_not_innercity_transport", None)
        self.transport_rules_by_distance = constraints_json.get("transport_rules_by_distance", None)

        self.must_depart_transport = constraints_json.get("must_depart_transport", None)
        self.must_return_transport = constraints_json.get("must_return_transport", None)
        self.must_not_depart_transport = constraints_json.get("must_not_depart_transport", None)
        self.must_not_return_transport = constraints_json.get("must_not_return_transport", None)

        self.attraction_budget = constraints_json.get("attraction_budget", None)
        self.restaurant_budget = constraints_json.get("restaurant_budget", None)
        self.hotel_budget = constraints_json.get("hotel_budget", None)
        self.innercity_budget = constraints_json.get("innercity_budget", None)
        self.intercity_budget = constraints_json.get("intercity_budget", None)
        self.overall_budget = constraints_json.get("overall_budget", None)

        self.requirement_list = requirement_list
        self._bundle_loc_anchor_cache = None
        self._normalize_transport_rules()
        self._force_arrive_time_pois()

    def _force_arrive_time_pois(self):
        """An arrive-time constraint (`if pos==X and start<=T: result=True`) requires
        the plan to actually VISIT X before T, but the parser only records the time
        window in activities_arrive_time_dict, so the named POI was never forced as a
        must-visit and the search could skip it. Add each such POI to the matching
        must-visit set (by its DB type) so the search includes it."""
        arrive = getattr(self, "activities_arrive_time_dict", None) or {}
        if not arrive:
            return
        mem = getattr(self, "memory", {}) or {}

        def _names(kind):
            df = mem.get(kind)
            try:
                return set(df["name"].astype(str)) if df is not None else set()
            except Exception:
                return set()

        rest_names, attr_names, acc_names = _names("restaurants"), _names("attractions"), _names("accommodations")

        def _add(attr, name):
            cur = getattr(self, attr, None) or []
            if not isinstance(cur, list):
                cur = [cur]
            if name not in cur:
                setattr(self, attr, cur + [name])

        for name in arrive:
            if name in rest_names:
                _add("must_visit_restaurant", name)
            elif name in attr_names:
                _add("must_see_attraction", name)
            elif name in acc_names:
                _add("must_live_hotel", name)

    def _pending_early_arrive_restaurant(self, current_time):
        """True if an unplaced must-visit restaurant has an 'early' arrive-time
        deadline still ahead of current_time. Used to serve it as an early lunch
        before morning attractions push the schedule past the deadline."""
        arrive = getattr(self, "activities_arrive_time_dict", None) or {}
        if not arrive:
            return False
        visiting = getattr(self, "restaurant_names_visiting", []) or []
        for name in (self.must_visit_restaurant or []):
            info = arrive.get(name)
            if not info or info[0] != "early":
                continue
            if self._visited_contains(visiting, name):
                continue
            # only worth forcing lunch if we can still make the deadline (<= deadline)
            if time_compare_if_earlier_equal(current_time, info[1]):
                return True
        return False

    def _constraint_branch_has_guidance(self, constraints_json):
        return any(
            key != "all_satisfy" and value is not None
            for key, value in constraints_json.items()
        )

    def _constraint_search_branches(self, results_main, requirement_list):
        if results_main.get("all_satisfy", True):
            return [(deepcopy(results_main), deepcopy(requirement_list), "all")]

        branches = []
        for idx, branch_constraints in enumerate(requirement_list or []):
            branch_constraints = deepcopy(branch_constraints)
            branch_constraints["all_satisfy"] = True
            if not self._constraint_branch_has_guidance(branch_constraints):
                continue
            branches.append((branch_constraints, [deepcopy(branch_constraints)], f"or_branch_{idx}"))

        legacy = deepcopy(results_main)
        branches.append((legacy, deepcopy(requirement_list), "legacy_merged_or"))
        return branches

    def _plan_passes_original_hard_logic(self, query, plan):
        if not isinstance(plan, dict) or not plan.get("itinerary"):
            return False
        public_query = self._public_query(query)
        try:
            if not func_commonsense_constraints(public_query, plan, verbose=False):
                return False
            logical_result = evaluate_constraints_py(
                public_query["hard_logic_py"], plan, verbose=False
            )
        except Exception:
            return False
        return bool(logical_result) and all(logical_result)

    def _search_with_installed_constraints(self, query, constraints_json, requirement_list):
        branch_query = self._public_query(query)
        branch_query["_urbantrip_constraints_override"] = (
            deepcopy(constraints_json),
            deepcopy(requirement_list),
        )
        branch_query["_urbantrip_search_start"] = self.time_before_search
        return self.generate_plan_with_search(branch_query)

    def _run_constraint_search_branches(self, query, constraints_json, requirement_list):
        fallback_success = None
        fallback_failure = None
        for branch_constraints, branch_requirements, branch_name in self._constraint_search_branches(
            constraints_json, requirement_list
        ):
            print(f"Trying constraint branch: {branch_name}")
            success, plan = self._search_with_installed_constraints(
                query, branch_constraints, branch_requirements
            )
            if success and self._plan_passes_original_hard_logic(query, plan):
                return True, plan
            if success and fallback_success is None:
                fallback_success = plan
            elif not success and fallback_failure is None:
                fallback_failure = plan

        if fallback_success is not None:
            return True, fallback_success
        return False, fallback_failure or {"error_info": "No solution found."}

    def run(self, query, prob_idx, oralce_translation=True):
        method_name = self.method + "_" + self.backbone_llm.name
        if oralce_translation:
            method_name = method_name + "_oracletranslation"

        self.log_dir = os.path.join(self.cache_dir, method_name)
        os.makedirs(self.log_dir, exist_ok=True)

        sys.stdout = Logger(
            "{}/{}.log".format(
                self.log_dir, query["uid"]
            ),
            sys.stdout,
            self.debug,
        )
        sys.stderr = Logger(
            "{}/{}.error".format(
                self.log_dir, query["uid"]
            ),
            sys.stderr,
            self.debug,
        )

        self.backbone_llm.input_token_count = 0
        self.backbone_llm.output_token_count = 0
        self.backbone_llm.input_token_maxx = 0

        # natural language -> symoblic language -> plan
        # if not oralce_translation:
        #     query = self.translate_nl2sl(query, load_cache=load_cache)

        succ, plan = self.symbolic_search(query)

        if succ:
            plan_out = plan
        else:
            plan_out = None
            if self.least_plan_logic is not None and self.least_plan_logic.get("itinerary"):
                plan_out = deepcopy(self.least_plan_logic)
                print("The least plan with logic constraints: ", plan_out)
                succ = True
            else:
                for cand in (self.least_plan_comm, self.least_plan_schema):
                    if cand is not None and cand.get("itinerary"):
                        plan_out = self._finalize_best_effort_plan(self.query, cand)
                        if plan_out is None:
                            plan_out = deepcopy(cand)
                        break
            if plan_out is None and isinstance(plan, dict) and plan.get("itinerary"):
                plan_out = plan
            elif plan_out is None:
                plan_out = plan if isinstance(plan, dict) else {}

        if isinstance(plan_out, dict) and plan_out.get("itinerary"):
            plan_out = self._eprsafe_output_plan(query, plan_out)
            plan_out = self._postprocess_insert_missing_meals(query, plan_out)
            plan_out = self._postprocess_insert_attractions_for_dav(query, plan_out)

        return succ, plan_out

    def symbolic_search(self, symoblic_query):
        if (symoblic_query["target_city"] in self.env.support_cities) and (
                symoblic_query["start_city"] in self.env.support_cities
        ):
            pass
        else:
            return False, {
                "error_info": f"Unsupported cities {symoblic_query['start_city']} -> {symoblic_query['target_city']}."}

        self.memory["accommodations"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "accommodation"
        )
        self.memory["attractions"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "attraction"
        )
        self.memory["restaurants"] = self.collect_poi_info_all(
            symoblic_query["target_city"], "restaurant"
        )

        self.query = symoblic_query

        success, plan = self.generate_plan_with_search(symoblic_query)

        return success, plan

    def generate_plan_with_search(self, query):
        # 初始化计时器和计数器
        self._distance_cache = {}  # 每条 query 重置距离缓存
        self._candidate_static_cache = {}
        self.time_before_search = query.get("_urbantrip_search_start", time.time())  # 记录搜索开始时间
        self._plan_pool = PlanPool()
        self.llm_inference_time_count = 0  # llm推理时间

        # reset the cache before searching
        poi_plan = {}  # 存储当前计划的 POI 信息
        self._dynamic_weight_cache = {}
        self._dynamic_ranking_context = None
        self.restaurants_visiting = []  # 正在访问的餐厅列表
        self.attractions_visiting = []  # 正在访问的景点列表
        self.food_type_visiting = []  # 正在访问的食物类型列表
        self.spot_type_visiting = []  # 正在访问的景点类型列表
        self.attraction_names_visiting = []  # 正在访问的景点名称列表
        self.restaurant_names_visiting = []  # 正在访问的餐厅名称列表

        self.llm_rec_format_error = 0  # llm推荐格式错误计数
        self.llm_rec_count = 0  # llm推荐计数
        self.search_nodes = 0  # 搜索节点计数
        self.backtrack_count = 0  # 回溯计数

        self.constraints_validation_count = 0  # 约束验证计数
        self.commonsense_pass_count = 0  # 常识通过计数
        self.logical_pass_count = 0  # 逻辑通过计数
        self.all_constraints_pass = 0  # 所有约束通过计数

        # 存储通过逻辑检查的次优计划
        self.least_plan_schema, self.least_plan_comm, self.least_plan_logic = None, None, None
        self.least_plan_logical_pass = -1
        self.least_plan_hard_pass = -1
        self.least_plan_objective_consistency_pass = False
        self.least_plan_activity_count = -1
        self._current_dfs_plan = None
        self._current_poi_plan = None
        # Keep DFS state transitions local: candidate generation performs pruning
        # before append, while final output polishing handles whole-plan repairs.
        self._enable_plan_sync = False
        self.FALLBACK_COMPLETE_SEC = 20
        # 提取用户需求
        # 获取用户约束信息
        # constraints_json = self.extract_user_constraints(query)
        override = query.get("_urbantrip_constraints_override")
        if override is not None:
            constraints_json, requirement_list = deepcopy(override[0]), deepcopy(override[1])
        else:
            constraints_json, requirement_list = self.extract_user_constraints_by_DSL(query)
            if constraints_json.get("all_satisfy") is False:
                return self._run_constraint_search_branches(query, constraints_json, requirement_list)

        self._install_constraint_state(constraints_json, requirement_list)
        self.all_satisfy_flag = False # 是否满足用户需求
        self.too_many_backtrack = False
        self.stop_search = False
        self.default_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": [],
        }

        source_city = query["start_city"] # 获取出发城市
        target_city = query["target_city"] # 获取目标城市

        print(source_city, "->", target_city)

        print("User's Constraints:")
        print(query['nature_language'])

        print("Formatted Expression:")
        for key, value in constraints_json.items():
            if value is not None:
                print(f"{key}: {value}")

        print("By list:")
        print(f"{self.requirement_list}")

        query_room_number = self.room_number
        query_room_numbed = self.bed_number

        print(f"query room number: {query_room_number}")
        print(f"query room numbed: {query_room_numbed}")

        # 收集去程和返程的城际火车交通选项
        train_go = self.collect_intercity_transport(source_city, target_city, "train")
        train_back = self.collect_intercity_transport(target_city, source_city, "train")

        # 收集去程和返程的城际飞机交通选项
        flight_go = self.collect_intercity_transport(
            source_city, target_city, "airplane"
        )
        flight_back = self.collect_intercity_transport(
            target_city, source_city, "airplane"
        )

        # must_not_depart_transport: 去程不允许的方式
        if self.must_not_depart_transport is not None:
            if "train" in self.must_not_depart_transport:
                train_go = pd.DataFrame()  # 置空
            if "airplane" in self.must_not_depart_transport:
                flight_go = pd.DataFrame()

        # must_not_return_transport: 返程不允许的方式
        if self.must_not_return_transport is not None:
            if "train" in self.must_not_return_transport:
                train_back = pd.DataFrame()
            if "airplane" in self.must_not_return_transport:
                flight_back = pd.DataFrame()

        # must_depart_transport: 去程必须的方式
        if self.must_depart_transport is not None:
            if "train" in self.must_depart_transport:
                flight_go = pd.DataFrame()
            if "airplane" in self.must_depart_transport:
                train_go = pd.DataFrame()

        # must_return_transport: 返程必须的方式
        if self.must_return_transport is not None:
            if "train" in self.must_return_transport:
                flight_back = pd.DataFrame()
            if "airplane" in self.must_return_transport:
                train_back = pd.DataFrame()

        # 计算可用航班和火车的数量，如果为 None 则设为 0
        flight_go_num = 0 if flight_go is None else flight_go.shape[0]
        train_go_num = 0 if train_go is None else train_go.shape[0]
        flight_back_num = 0 if flight_back is None else flight_back.shape[0]
        train_back_num = 0 if train_back is None else train_back.shape[0]

        # 合并最终的去程与返程交通选项
        go_info = pd.concat([train_go, flight_go], axis=0)
        back_info = pd.concat([train_back, flight_back], axis=0)

        # 打印调试信息，显示交通选项数量
        if self.debug:
            print(
                "from {} to {}: {} flights, {} trains".format(
                    source_city, target_city, flight_go_num, train_go_num
                )
            )
            print(
                "from {} to {}: {} flights, {} trains".format(
                    target_city, source_city, flight_back_num, train_back_num
                )
            )

            print(go_info.head())
            print(back_info.head())

        # 对去程城际交通进行排序
        ranking_go = self.ranking_intercity_transport_go(go_info, query)

        # 对酒店进行排序
        ranking_hotel = self.ranking_hotel(self.memory["accommodations"], query)

        default_hotel = (
            self.memory["accommodations"]
            .sort_values(by="price")
            .index
            .tolist()
        )

        # 根据查询对市内交通进行排序
        self.innercity_transports_ranking = ["metro", "taxi", "walk"]
        if self.must_innercity_transport is not None:
            self.innercity_transports_ranking = self.must_innercity_transport[:]
        if self.must_not_innercity_transport is not None:
            self.innercity_transports_ranking = [
                t for t in self.innercity_transports_ranking
                if t not in self.must_not_innercity_transport
            ]
        if self._innercity_budget_saver_enabled():
            budget_first = ["walk", "metro", "taxi"]
            self.innercity_transports_ranking = [
                t for t in budget_first if t in self.innercity_transports_ranking
            ]

        # Geo-aware bundle search: score (go, back, hotel) bundles up front and
        # only run DFS on the most promising few. Legacy nested loop kept below.
        if self.use_bundle_search:
            return self._bundle_search(
                query, go_info, back_info, ranking_go, ranking_hotel, poi_plan
            )

        # 遍历排序后的去程交通
        intercity_budget_msg = "intercity budget not satisfied, backtrack..."
        intercity_budget_count = 0
        for go_i in ranking_go:
            go_info_i = go_info.iloc[go_i]  # 获取当前去程交通信息
            if pd.isna(go_info_i["Cost"]):
                continue
            poi_plan["go_transport"] = go_info_i  # 将其添加到计划中
            self.search_nodes += 1

            # 对返程城际交通进行排序（依赖于去程信息）
            ranking_back = self.ranking_intercity_transport_back(
                back_info, query, go_info_i
            )
            # 遍历排序后的返程交通
            for back_i in ranking_back:
                if self._search_time_exceeded():
                    self.default_plan["backtrack_count"] = self.backtrack_count
                    return self._best_effort_search_result(query, None, poi_plan)

                back_info_i = back_info.iloc[back_i]  # 获取当前返程交通信息
                if pd.isna(back_info_i["Cost"]):
                    continue
                poi_plan["back_transport"] = back_info_i  # 将其添加到计划中

                self.search_nodes += 1

                # 检查城际交通预算
                if not self.too_many_backtrack:
                    self.intercity_cost = (poi_plan["go_transport"]["Cost"] + poi_plan["back_transport"]["Cost"]) * query["people_number"]
                    if self.intercity_budget is not None and self.intercity_budget < self.intercity_cost:
                        intercity_budget_count += 1
                        print(f"{intercity_budget_msg}（{intercity_budget_count}次）".ljust(80), end='\r', flush=True)
                        print("intercity budget not satisfied, backtrack...")
                        self.backtrack_count += 1
                        continue

                if query["days"] > 1:  # 如果天数大于 1，则需要考虑酒店
                    # cnt = 0
                    # 遍历排序后的酒店选项
                    ranking_hotel_for_trip = self._rank_hotels_for_innercity_budget(
                        ranking_hotel,
                        self.memory["accommodations"],
                        query,
                        poi_plan,
                    )
                    for hotel_i in ranking_hotel_for_trip:
                        # 获取当前酒店信息
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                        # 获取酒店房间类型（床位数）
                        room_type = poi_plan["accommodation"]["numbed"]
                        self.search_nodes += 1

                        # 计算所需的房间数量
                        required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                        # 检查查询中的房间类型是否与当前酒店匹配，不匹配则回溯
                        if not self.too_many_backtrack:
                            if query_room_numbed != None and query_room_numbed != room_type:
                                self.backtrack_count += 1
                                print("room_type not match, backtrack...")
                                continue

                        # 如果查询中指定了房间数量，则使用该数量
                        if query_room_number != None:
                            required_rooms = query_room_number

                        # 检查房间数量和类型是否满足人数要求
                        if query_room_number != None and query_room_numbed != None:
                            pass  # 如果同时指定了房间数量和类型，则直接通过
                        else:
                            if (
                                    room_type * required_rooms >= query["people_number"]
                            ) and (
                                    room_type * required_rooms < query["people_number"] + room_type
                            ):
                                pass  # 如果房间足够容纳人数且不过分多余，则通过
                            else:
                                if query_room_number != None and room_type == 2:
                                    pass   # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                else:
                                    if not self.too_many_backtrack:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                        # 所需的房间数量
                        self.required_rooms = required_rooms

                        # 检查酒店的预算
                        if not self.too_many_backtrack:
                            self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                            if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                self.backtrack_count += 1
                                #print("hotel budget not satisfied, backtrack...")
                                continue

                        # 检查当前总预算
                        if not self.too_many_backtrack:
                            self.overall_cost = self.intercity_cost + self.hotel_cost
                            if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                                self.backtrack_count += 1
                                print("overall cost < intercity + hotel, backtrack...")
                                continue

                        print("search: ...")
                        self._current_poi_plan = poi_plan
                        # 尝试通过 DFS 搜索 POI 计划
                        try:
                            success, plan = self.dfs_poi(
                                query,
                                poi_plan, # go、back、accommodation
                                plan=[],
                                current_time="",
                                current_position="",
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}
                        # exit(0)

                        # print(success, plan)
                        if success:
                            return True, plan
                        if self._is_terminal_plan_failure(plan):
                            return False, plan
                        else:
                            if self._search_time_exceeded():
                                self.default_plan["backtrack_count"] = self.backtrack_count
                                return self._best_effort_search_result(query, plan, poi_plan)

                            self.backtrack_count += 1
                            print("search failed given the intercity-transport and hotels, backtrack...")
                    if self._has_hard_hotel_constraints():
                        # A relaxed hotel cannot satisfy an explicit hotel constraint.
                        # Try the next intercity combination instead of wasting DFS budget.
                        continue
                    # 都不满足，则从所有酒店中选
                    print("No Hotel satisfies constraint")
                    rnbc = 0 # room number bed count
                    ohbc = 0 # over hotel budget count
                    oobc = 0 # over overall budget count
                    default_hotel_for_trip = self._rank_hotels_for_innercity_budget(
                        default_hotel,
                        self.memory["accommodations"],
                        query,
                        poi_plan,
                    )
                    for hotel_i in default_hotel_for_trip:
                        # 获取当前酒店信息
                        poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                        # 获取酒店房间类型（床位数）
                        room_type = poi_plan["accommodation"]["numbed"]
                        self.search_nodes += 1

                        # 计算所需的房间数量
                        required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                        # 检查查询中的房间类型是否与当前酒店匹配，不匹配则回溯
                        if not self.too_many_backtrack:
                            if query_room_numbed != None and query_room_numbed != room_type:
                                self.backtrack_count += 1
                                rnbc += 1
                                print("room_type not match, backtrack...")
                                continue

                        # 如果查询中指定了房间数量，则使用该数量
                        if query_room_number != None:
                            required_rooms = query_room_number

                        # 检查房间数量和类型是否满足人数要求
                        if query_room_number != None and query_room_numbed != None:
                            pass  # 如果同时指定了房间数量和类型，则直接通过
                        else:
                            if (
                                    room_type * required_rooms >= query["people_number"]
                            ) and (
                                    room_type * required_rooms < query["people_number"] + room_type
                            ):
                                pass  # 如果房间足够容纳人数且不过分多余，则通过
                            else:
                                if query_room_number != None and room_type == 2:
                                    pass   # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                else:
                                    if not self.too_many_backtrack:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                        # 所需的房间数量
                        self.required_rooms = required_rooms

                        # 检查酒店的预算
                        if not self.too_many_backtrack:
                            self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                            if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                self.backtrack_count += 1
                                ohbc += 1
                                #print("hotel budget not satisfied, backtrack...")
                                continue

                        # 检查当前总预算
                        if not self.too_many_backtrack:
                            self.overall_cost = self.intercity_cost + self.hotel_cost
                            if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                                self.backtrack_count += 1
                                oobc += 1
                                print("overall cost < intercity + hotel, backtrack...")
                                continue

                        print("search: ...")
                        self._current_poi_plan = poi_plan
                        # 尝试通过 DFS 搜索 POI 计划
                        try:
                            success, plan = self.dfs_poi(
                                query,
                                poi_plan, # go、back、accommodation
                                plan=[],
                                current_time="",
                                current_position="",
                            )
                        except TimeOutError as e:
                            print("TimeOutError")
                            return False, {"error_info": "TimeOutError"}
                        # exit(0)


                        # print(success, plan)
                        if success:
                            return True, plan
                        if self._is_terminal_plan_failure(plan):
                            return False, plan
                        else:
                            if self._search_time_exceeded():
                                self.default_plan["backtrack_count"] = self.backtrack_count
                                return self._best_effort_search_result(query, plan, poi_plan)

                            self.backtrack_count += 1
                            print("search failed given the intercity-transport and hotels, backtrack...")
                    # 放弃约束
                    if rnbc == len(default_hotel) or ohbc == len(default_hotel) or oobc == len(default_hotel):
                        for hotel_i in default_hotel_for_trip:
                            # 获取当前酒店信息
                            poi_plan["accommodation"] = self.memory["accommodations"].iloc[hotel_i]

                            # 获取酒店房间类型（床位数）
                            room_type = poi_plan["accommodation"]["numbed"]
                            self.search_nodes += 1

                            # 计算所需的房间数量
                            required_rooms = (int((query["people_number"] - 1) / room_type) + 1)

                            if rnbc != len(default_hotel):
                                if query_room_numbed != None and query_room_numbed != room_type:
                                    self.backtrack_count += 1
                                    rnbc += 1
                                    print("room_type not match, backtrack...")
                                    continue

                            # 如果查询中指定了房间数量，则使用该数量
                            if query_room_number != None:
                                required_rooms = query_room_number

                            # 检查房间数量和类型是否满足人数要求
                            if query_room_number != None and query_room_numbed != None:
                                pass  # 如果同时指定了房间数量和类型，则直接通过
                            else:
                                if (
                                        room_type * required_rooms >= query["people_number"]
                                ) and (
                                        room_type * required_rooms < query["people_number"] + room_type
                                ):
                                    pass  # 如果房间足够容纳人数且不过分多余，则通过
                                else:
                                    if query_room_number != None and room_type == 2:
                                        pass  # 特殊情况，如果指定了房间数量且房间类型为2，则通过
                                    else:
                                        self.backtrack_count += 1
                                        # print("room_number * room_type not match, backtrack...")
                                        continue  # 不满足要求则回溯

                            # 所需的房间数量
                            self.required_rooms = required_rooms

                            if ohbc != len(default_hotel):
                                # 检查酒店的预算
                                self.hotel_cost = poi_plan["accommodation"]["price"] * required_rooms * (query["days"] - 1)
                                if self.hotel_budget is not None and self.hotel_budget < self.hotel_cost:
                                    self.backtrack_count += 1
                                    ohbc += 1
                                    # print("hotel budget not satisfied, backtrack...")
                                    continue

                            if oobc != len(default_hotel):
                                # 检查当前总预算
                                self.overall_cost = self.intercity_cost + self.hotel_cost
                                if self.overall_budget is not None and self.overall_budget < self.overall_cost:
                                    self.backtrack_count += 1
                                    oobc += 1
                                    print("overall cost < intercity + hotel, backtrack...")
                                    continue

                            print("search: ...")
                            # 尝试通过 DFS 搜索 POI 计划
                            try:
                                success, plan = self.dfs_poi(
                                    query,
                                    poi_plan,  # go、back、accommodation
                                    plan=[],
                                    current_time="",
                                    current_position="",
                                )
                            except TimeOutError as e:
                                print("TimeOutError")
                                return False, {"error_info": "TimeOutError"}
                            # exit(0)

                            # print(success, plan)
                            if success:
                                return True, plan
                            if self._is_terminal_plan_failure(plan):
                                return False, plan
                            else:
                                if self._search_time_exceeded():
                                    self.default_plan["backtrack_count"] = self.backtrack_count
                                    return self._best_effort_search_result(query, None, poi_plan)

                                self.backtrack_count += 1
                                print("search failed given the intercity-transport and hotels, backtrack...")

                else:  # 如果旅行天数只有 1 天，则不需要考虑酒店
                    self.hotel_cost = 0
                    # 检查返程交通的开始时间是否早于或等于去程交通的结束时间
                    if time_compare_if_earlier_equal(
                            poi_plan["back_transport"]["BeginTime"],
                            poi_plan["go_transport"]["EndTime"],
                    ):
                        self.backtrack_count += 1
                        print("back_transport BeginTime earlier than go_transport EndTime, backtrack...")
                        continue

                    # 计算城际交通的总成本（无酒店）
                    if not self.too_many_backtrack:
                        self.intercity_cost = (poi_plan["go_transport"]["Cost"] + poi_plan["back_transport"]["Cost"]) * query["people_number"]
                        if self.intercity_budget is not None and self.intercity_budget < self.intercity_cost:
                            self.backtrack_count += 1
                            print("[one-day-trip]intercity budget < cost, backtrack...")
                            continue

                    print("search: ...")
                    # 尝试通过 DFS 搜索 POI 计划
                    try:
                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan=[],
                            current_time="",
                            current_position="",
                        )
                    except TimeOutError as e:
                        print("TimeOutError")
                        return False, {"error_info": "TimeOutError"}

                    # print(success, plan)
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan
                    else:
                        if self._search_time_exceeded():
                            self.default_plan["backtrack_count"] = self.backtrack_count
                            return self._best_effort_search_result(query, None, poi_plan)

                        self.backtrack_count += 1
                        print("search failed given the intercity-transport and hotels, backtrack...")

        return False, {"error_info": "No solution found."}

    def _plan_activity_count(self, itinerary):
        if not itinerary:
            return 0
        return sum(len(day.get("activities", [])) for day in itinerary)

    def _is_better_plan_score(self, hard_pass, objective_consistency_pass, activity_count):
        """Rank fallback plans without changing the search objective.

        objective_consistency_pass is the boolean returned by
        func_commonsense_constraints(): all objective/commonsense checkers
        passed, including transport, POI, time, and space consistency.
        """
        if hard_pass > self.least_plan_hard_pass:
            return True
        if hard_pass < self.least_plan_hard_pass:
            return False
        if objective_consistency_pass and not self.least_plan_objective_consistency_pass:
            return True
        if not objective_consistency_pass and self.least_plan_objective_consistency_pass:
            return False
        return activity_count > self.least_plan_activity_count

    def _update_best_plan(self, query, res_plan, objective_consistency_pass, logical_result):
        itinerary = res_plan.get("itinerary")
        activity_count = self._plan_activity_count(itinerary)
        if activity_count == 0:
            return

        hard_pass = int(np.sum(logical_result))
        logical_pass = bool(logical_result) and all(logical_result)

        self.least_plan_schema = deepcopy(res_plan)

        if self._is_better_plan_score(hard_pass, objective_consistency_pass, activity_count):
            self.least_plan_comm = deepcopy(res_plan)
            self.least_plan_hard_pass = hard_pass
            self.least_plan_logical_pass = hard_pass
            self.least_plan_objective_consistency_pass = objective_consistency_pass
            self.least_plan_activity_count = activity_count
            self.least_plan_comm["hard_pass_count"] = hard_pass
            self.least_plan_comm["commonsense_pass"] = objective_consistency_pass

        if objective_consistency_pass and logical_pass and self.least_plan_logic is None:
            self.least_plan_logic = deepcopy(res_plan)

    def _update_best_plan_from_partial(self, query, plan):
        if not plan or not isinstance(plan, list):
            return
        if self._plan_activity_count(plan) == 0:
            return

        itinerary = deepcopy(plan)
        repair_full_itinerary(self, query, itinerary)
        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": itinerary,
        }
        logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=False)
        commonsense_ok = bool(func_commonsense_constraints(query, res_plan, verbose=False))
        self._update_best_plan(query, res_plan, commonsense_ok, logical_result)

    def _polish_output_plan(self, query, plan):
        """Repair time chain on every emitted JSON so eval never sees 24:xx/27:xx."""
        if not plan or not isinstance(plan, dict) or not plan.get("itinerary"):
            return plan
        polished = deepcopy(plan)
        itinerary = polished["itinerary"]
        repair_full_itinerary(self, query, itinerary)
        polished["itinerary"] = itinerary
        shell = self._res_plan_shell(query, itinerary)
        polished["commonsense_pass"] = bool(
            func_commonsense_constraints(query, shell, verbose=False)
        )
        return self._annotate_daily_meal_check(polished)

    def _strip_poi_types(self, itinerary, poi_types):
        for day in itinerary:
            day["activities"] = [
                act for act in day.get("activities", [])
                if act.get("type") not in poi_types
            ]

    def _eprsafe_output_plan(self, query, plan):
        """Polish plan; if commonsense still fails, strip risky POIs and re-complete skeleton."""
        if not plan or not isinstance(plan, dict) or not plan.get("itinerary"):
            return plan
        plan = self._polish_output_plan(query, plan)
        if plan.get("commonsense_pass"):
            return plan

        stripped = deepcopy(plan)
        itinerary = stripped["itinerary"]
        strip_groups = (("attraction",), ("lunch", "dinner"), ("breakfast",))
        for poi_types in strip_groups:
            self._strip_poi_types(itinerary, poi_types)
            self._drop_broken_activities(itinerary)
            self._ensure_day_count(query, itinerary)
            repair_full_itinerary(self, query, itinerary)
            shell = self._res_plan_shell(query, itinerary)
            if self._commonsense_passes(query, shell):
                stripped["itinerary"] = itinerary
                stripped["commonsense_pass"] = True
                return stripped

        poi_plan = getattr(self, "_current_poi_plan", None)
        if poi_plan:
            completed = self._fallback_complete_for_commonsense(query, poi_plan, stripped)
            if completed is not None:
                return self._polish_output_plan(query, completed)

        stripped["commonsense_pass"] = self._commonsense_passes(
            query, self._res_plan_shell(query, itinerary)
        )
        return stripped

    def _arrived_time_too_late_for_hotel(self, arrived_time):
        if not arrived_time:
            return True
        return time_to_minutes(str(arrived_time).split("次日")[-1]) >= time_to_minutes("24:00")

    def _innercity_transports_valid(self, transports):
        if not isinstance(transports, list):
            return False
        day_end = time_to_minutes("24:00")
        for tr in transports:
            for key in ("start_time", "end_time"):
                t = tr.get(key)
                if t and time_to_minutes(str(t).split("次日")[-1]) >= day_end:
                    return False
        return True

    def _time_minutes(self, time_str):
        if not time_str:
            return 0
        time_str = str(time_str).split("次日")[-1]
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    def _is_intercity_activity(self, activity):
        if not isinstance(activity, dict):
            return False
        if activity.get("type") in ("train", "airplane"):
            return True
        return "TrainID" in activity or "FlightID" in activity

    def _activity_position(self, activity):
        if not isinstance(activity, dict):
            return ""
        if activity.get("position"):
            return activity["position"]
        if self._is_intercity_activity(activity):
            return activity.get("end", "")
        return ""

    def _last_activity_matches(self, plan, day_idx, poi_type=None, position=None):
        if not isinstance(plan, list) or not plan or day_idx < 0 or day_idx >= len(plan):
            return False
        activities = plan[day_idx].get("activities", [])
        if not activities:
            return False
        last_act = activities[-1]
        if poi_type is not None and last_act.get("type") != poi_type:
            return False
        if position is not None and self._activity_position(last_act) != position:
            return False
        return True

    def _pop_last_activity_if_matches(self, plan, day_idx, poi_type=None, position=None):
        if not self._last_activity_matches(plan, day_idx, poi_type, position):
            return False
        plan[day_idx]["activities"].pop()
        return True

    def _is_terminal_plan_failure(self, plan):
        return isinstance(plan, dict) and not plan.get("itinerary")

    def _res_plan_shell(self, query, itinerary):
        return {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": itinerary,
        }

    def _effective_time_cut(self):
        return self.TIME_CUT - getattr(self, "EXTERNAL_TIMEOUT_BUFFER", 15)

    def _search_time_exceeded(self, slack=0):
        return time.time() > self.time_before_search + self._effective_time_cut() + slack

    def _commonsense_passes(self, query, res_plan):
        return bool(func_commonsense_constraints(query, res_plan, verbose=False))

    def _hard_logic_results(self, query, res_plan):
        try:
            return evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=False)
        except Exception:
            return []

    def _hard_pass_count(self, query, res_plan):
        return int(np.sum(self._hard_logic_results(query, res_plan)))

    def _plan_passes_official_constraints(self, query, res_plan):
        if not isinstance(res_plan, dict) or not res_plan.get("itinerary"):
            return False
        public_query = self._public_query(query)
        try:
            if not func_commonsense_constraints(public_query, res_plan, verbose=False):
                return False
            logical_result = evaluate_constraints_py(
                public_query["hard_logic_py"], res_plan, verbose=False
            )
        except Exception:
            return False
        return bool(logical_result) and all(logical_result)

    def _plan_passes_all_constraints(self, query, res_plan):
        if not self._plan_passes_official_constraints(query, res_plan):
            return False
        if self.require_complete_daily_meals:
            meal_report = self._daily_meal_report(res_plan)
            if not meal_report["pass"]:
                return False
        return True

    def _daily_meal_report(self, res_plan):
        meal_types = ("breakfast", "lunch", "dinner")
        report = {
            "pass": False,
            "target_meals_per_day": len(meal_types),
            "total_meals": 0,
            "missing": [],
            "days": [],
        }
        if not isinstance(res_plan, dict) or not isinstance(res_plan.get("itinerary"), list):
            report["missing"].append({"day": None, "meals": list(meal_types)})
            return report

        for fallback_day, day in enumerate(res_plan.get("itinerary", []), start=1):
            day_no = day.get("day", fallback_day)
            present = []
            for act in day.get("activities", []):
                act_type = act.get("type")
                if act_type in meal_types and act_type not in present:
                    present.append(act_type)
            missing = [meal for meal in meal_types if meal not in present]
            report["total_meals"] += len(present)
            day_report = {
                "day": day_no,
                "present": present,
                "missing": missing,
                "pass": not missing,
            }
            report["days"].append(day_report)
            if missing:
                report["missing"].append({"day": day_no, "meals": missing})

        report["pass"] = not report["missing"]
        return report

    def _annotate_daily_meal_check(self, plan):
        if isinstance(plan, dict) and plan.get("itinerary"):
            report = self._daily_meal_report(plan)
            plan["daily_meal_check"] = report
            plan["meal_completeness_pass"] = report["pass"]
        return plan

    def _meal_time_valid(self, meal_type, start_time, end_time):
        try:
            start_min = time_to_minutes(str(start_time).split("次日")[-1])
            end_min = time_to_minutes(str(end_time).split("次日")[-1])
        except (TypeError, ValueError):
            return False
        windows = {
            "breakfast": (time_to_minutes("06:00"), time_to_minutes("09:00")),
            "lunch": (time_to_minutes("11:00"), time_to_minutes("14:00")),
            "dinner": (time_to_minutes("17:00"), time_to_minutes("20:00")),
        }
        if meal_type not in windows:
            return False
        earliest_end, latest_start = windows[meal_type]
        return start_min < latest_start and end_min > earliest_end and start_min < end_min

    def _candidate_plan_with_itinerary(self, plan, itinerary):
        candidate = deepcopy(plan)
        candidate["itinerary"] = itinerary
        return candidate

    def _accept_meal_candidate(self, query, plan, candidate_itinerary):
        candidate = self._candidate_plan_with_itinerary(plan, candidate_itinerary)
        repair_full_itinerary(self, query, candidate["itinerary"])
        if not self._plan_passes_official_constraints(query, candidate):
            return None
        return self._annotate_daily_meal_check(candidate)

    def _hotel_name_for_breakfast_day(self, itinerary, day_idx):
        if day_idx > 0:
            for act in reversed(itinerary[day_idx - 1].get("activities", [])):
                if act.get("type") == "accommodation" and act.get("position"):
                    return act["position"]
        for act in itinerary[day_idx].get("activities", []):
            if act.get("type") == "accommodation" and act.get("position"):
                return act["position"]
        poi_plan = getattr(self, "_current_poi_plan", None) or {}
        hotel = poi_plan.get("accommodation")
        if isinstance(hotel, dict):
            return hotel.get("name")
        return None

    def _try_insert_hotel_breakfast_at_day_start(self, query, plan, day_idx):
        updated = deepcopy(plan.get("itinerary", []))
        if day_idx < 0 or day_idx >= len(updated):
            return None
        activities = updated[day_idx].setdefault("activities", [])
        if any(act.get("type") == "breakfast" for act in activities):
            return None
        hotel_name = self._hotel_name_for_breakfast_day(updated, day_idx)
        if not hotel_name:
            return None
        breakfast = {
            "position": hotel_name,
            "type": "breakfast",
            "price": 0,
            "cost": 0,
            "start_time": "06:00",
            "end_time": "06:30",
            "transports": [],
        }

        if not activities:
            activities.insert(0, breakfast)
            return self._accept_meal_candidate(query, plan, updated)

        first = activities[0]
        if not self._is_intercity_activity(first):
            if not time_compare_if_earlier_equal("06:30", first.get("start_time", "00:00")):
                return None
            transports = first.get("transports") or []
            first_start = transports[0].get("start") if transports else self._activity_start_position(first)
            if first_start and first_start != hotel_name:
                return None
            activities.insert(0, breakfast)
            return self._accept_meal_candidate(query, plan, updated)

        if day_idx == 0 and len(activities) > 1:
            return self._try_insert_hotel_breakfast_between(
                query, plan, day_idx, 1, hotel_name
            )
        return None

    def _try_insert_hotel_breakfast_between(self, query, plan, day_idx, insert_idx, hotel_name):
        updated = deepcopy(plan.get("itinerary", []))
        activities = updated[day_idx].get("activities", [])
        if insert_idx <= 0 or insert_idx >= len(activities):
            return None
        prev_act = activities[insert_idx - 1]
        next_act = activities[insert_idx]
        prev_position = self._activity_end_position(prev_act)
        next_position = self._activity_start_position(next_act)
        current_time = prev_act.get("end_time", "")
        next_start = next_act.get("start_time", "")
        if not prev_position or not next_position or not current_time or not next_start:
            return None

        for mode_to_hotel in self._postprocess_transport_modes(query, prev_position, hotel_name):
            transports_to_hotel, arrival = self._postprocess_collect_transport(
                query, prev_position, hotel_name, current_time, mode_to_hotel
            )
            if transports_to_hotel is None:
                continue
            start_time = arrival
            if time_compare_if_earlier_equal(start_time, "06:00"):
                start_time = "06:00"
            end_time = add_time_delta(start_time, 30)
            if not self._meal_time_valid("breakfast", start_time, end_time):
                continue
            if not time_compare_if_earlier_equal(end_time, next_start):
                continue

            for mode_to_next in self._postprocess_transport_modes(query, hotel_name, next_position):
                transports_to_next, next_arrival = self._postprocess_collect_transport(
                    query, hotel_name, next_position, end_time, mode_to_next
                )
                if transports_to_next is None:
                    continue
                if not time_compare_if_earlier_equal(next_arrival, next_start):
                    continue
                activities.insert(
                    insert_idx,
                    {
                        "position": hotel_name,
                        "type": "breakfast",
                        "price": 0,
                        "cost": 0,
                        "start_time": start_time,
                        "end_time": end_time,
                        "transports": transports_to_hotel,
                    },
                )
                activities[insert_idx + 1]["transports"] = transports_to_next
                return self._accept_meal_candidate(query, plan, updated)
        return None

    def _meal_restaurant_positions(self, itinerary):
        positions = set()
        restaurants = self.memory.get("restaurants")
        restaurant_names = set(restaurants["name"].astype(str)) if restaurants is not None else set()
        for day in itinerary or []:
            for act in day.get("activities", []):
                if act.get("type") in {"breakfast", "lunch", "dinner"}:
                    pos = act.get("position")
                    if pos in restaurant_names:
                        positions.add(pos)
        return positions

    def _meal_candidates_for_gap(self, meal_type, existing_restaurants):
        candidates = self._filter_restaurant_hard_candidates(self.memory["restaurants"])
        candidates = candidates[~candidates["name"].isin(existing_restaurants)]
        if candidates.empty:
            return candidates
        if meal_type == "breakfast":
            target_time = "08:00"
        elif meal_type == "lunch":
            target_time = "11:30"
        else:
            target_time = "17:30"
        return filter_open_at_time(candidates, target_time)

    def _try_build_meal_insertion(self, query, itinerary, day_idx, insert_idx, meal_type, res_row):
        updated = deepcopy(itinerary)
        activities = updated[day_idx].get("activities", [])
        if insert_idx <= 0 or insert_idx >= len(activities):
            return None
        prev_act = activities[insert_idx - 1]
        next_act = activities[insert_idx]
        prev_position = self._activity_end_position(prev_act)
        next_position = self._activity_start_position(next_act)
        current_time = prev_act.get("end_time", "")
        next_start = next_act.get("start_time", "")
        if not prev_position or not next_position or not current_time or not next_start:
            return None
        if not time_compare_if_earlier_equal(current_time, next_start):
            return None

        for mode_to_res in self._postprocess_transport_modes(query, prev_position, res_row["name"]):
            transports_to_res, res_arrival = self._postprocess_collect_transport(
                query, prev_position, res_row["name"], current_time, mode_to_res
            )
            if transports_to_res is None:
                continue
            scheduled = self._scheduled_poi_times(
                res_row["name"],
                res_arrival,
                res_row.get("opentime", "00:00"),
                res_row.get("endtime", "23:59"),
                self.meal_postprocess_minutes,
                meal_type,
            )
            if scheduled is None:
                continue
            meal_start, meal_end = scheduled
            if not self._meal_time_valid(meal_type, meal_start, meal_end):
                continue
            if not time_compare_if_earlier_equal(meal_end, next_start):
                continue

            for mode_to_next in self._postprocess_transport_modes(query, res_row["name"], next_position):
                transports_to_next, next_arrival = self._postprocess_collect_transport(
                    query, res_row["name"], next_position, meal_end, mode_to_next
                )
                if transports_to_next is None:
                    continue
                if not time_compare_if_earlier_equal(next_arrival, next_start):
                    continue
                activities.insert(
                    insert_idx,
                    {
                        "position": res_row["name"],
                        "type": meal_type,
                        "price": int(res_row["price"]),
                        "cost": int(res_row["price"]) * query["people_number"],
                        "start_time": meal_start,
                        "end_time": meal_end,
                        "transports": transports_to_res,
                    },
                )
                activities[insert_idx + 1]["transports"] = transports_to_next
                return updated
        return None

    def _postprocess_insert_one_restaurant_meal(
        self, query, plan, day_idx, meal_type, deadline, attempts
    ):
        itinerary = plan.get("itinerary", [])
        existing_restaurants = self._meal_restaurant_positions(itinerary)
        meal_candidates = self._meal_candidates_for_gap(meal_type, existing_restaurants)
        if meal_candidates.empty:
            return plan, attempts, False

        activities = itinerary[day_idx].get("activities", [])
        insert_order = range(1, len(activities))
        if self.dav_postprocess_prioritize_large_gaps:
            insert_order = sorted(
                insert_order,
                key=lambda idx: self._gap_minutes(activities[idx - 1], activities[idx]),
                reverse=True,
            )
        for insert_idx in insert_order:
            if attempts >= self.meal_postprocess_max_attempts or time.time() >= deadline:
                return plan, attempts, False
            prev_act = activities[insert_idx - 1]
            next_act = activities[insert_idx]
            if next_act.get("type") == "breakfast":
                continue
            prev_position = self._activity_end_position(prev_act)
            next_position = self._activity_start_position(next_act)
            ranked = self._rank_dav_insertion_candidates(
                meal_candidates, prev_position, next_position
            )
            for _, res_row in ranked.head(self.meal_postprocess_candidates_per_gap).iterrows():
                if attempts >= self.meal_postprocess_max_attempts or time.time() >= deadline:
                    return plan, attempts, False
                attempts += 1
                candidate_itinerary = self._try_build_meal_insertion(
                    query, itinerary, day_idx, insert_idx, meal_type, res_row
                )
                if candidate_itinerary is None:
                    continue
                candidate = self._accept_meal_candidate(query, plan, candidate_itinerary)
                if candidate is None:
                    continue
                print(
                    f"Meal postprocess inserted {meal_type}: {res_row['name']} "
                    f"(day={day_idx + 1}, attempts={attempts})"
                )
                return candidate, attempts, True
        return plan, attempts, False

    def _postprocess_insert_missing_meals(self, query, plan):
        if (
            not self.enable_meal_postprocess
            or self.meal_postprocess_seconds <= 0
            or self.meal_postprocess_max_attempts <= 0
            or not isinstance(plan, dict)
            or not plan.get("itinerary")
        ):
            return self._annotate_daily_meal_check(plan)
        if not self._plan_passes_official_constraints(query, plan):
            return self._annotate_daily_meal_check(plan)

        timeout_guard = self.time_before_search + max(1, self.TIME_CUT - 5)
        deadline = min(time.time() + self.meal_postprocess_seconds, timeout_guard)
        improved = self._annotate_daily_meal_check(deepcopy(plan))
        attempts = 0
        insertions = 0

        while time.time() < deadline and attempts < self.meal_postprocess_max_attempts:
            report = self._daily_meal_report(improved)
            if report["pass"]:
                break
            changed = False
            for day_report in report["days"]:
                day_idx = int(day_report["day"]) - 1
                if "breakfast" in day_report["missing"]:
                    candidate = self._try_insert_hotel_breakfast_at_day_start(
                        query, improved, day_idx
                    )
                    attempts += 1
                    if candidate is not None:
                        improved = candidate
                        insertions += 1
                        changed = True
                        print(f"Meal postprocess inserted breakfast at hotel (day={day_idx + 1})")
                        break
                    improved, attempts, accepted = self._postprocess_insert_one_restaurant_meal(
                        query, improved, day_idx, "breakfast", deadline, attempts
                    )
                    if accepted:
                        insertions += 1
                        changed = True
                        break
                for meal_type in ("lunch", "dinner"):
                    if meal_type not in day_report["missing"]:
                        continue
                    improved, attempts, accepted = self._postprocess_insert_one_restaurant_meal(
                        query, improved, day_idx, meal_type, deadline, attempts
                    )
                    if accepted:
                        insertions += 1
                        changed = True
                        break
                if changed:
                    break
            if not changed:
                break

        improved = self._annotate_daily_meal_check(improved)
        improved["meal_postprocess_insertions"] = insertions
        improved["meal_postprocess_attempts"] = attempts
        return improved

    def _activity_start_position(self, activity):
        if not isinstance(activity, dict):
            return ""
        if self._is_intercity_activity(activity):
            return activity.get("start", "")
        return activity.get("position", "")

    def _activity_end_position(self, activity):
        return self._activity_position(activity)

    def _gap_minutes(self, prev_act, next_act):
        """Wall-clock minutes between ``prev_act.end_time`` and
        ``next_act.start_time``. Used only to *order* gap-fill scanning
        (largest-idle-window-first); returns -1 (sorts last) when either
        timestamp is missing/unparseable so callers fall back to encounter
        order instead of raising.
        """
        try:
            prev_end = prev_act.get("end_time")
            next_start = next_act.get("start_time")
            if not prev_end or not next_start:
                return -1
            gap = time_to_minutes(str(next_start)) - time_to_minutes(str(prev_end))
            return gap if gap >= 0 else -1
        except (TypeError, ValueError):
            return -1

    def _poi_coordinate(self, name):
        if not name:
            return None
        for key in ("attractions", "restaurants", "accommodations"):
            info = self.memory.get(key)
            if info is None or "name" not in info.columns:
                continue
            match = info[info["name"] == name]
            if not match.empty and {"lat", "lon"}.issubset(match.columns):
                row = match.iloc[0]
                try:
                    return float(row["lat"]), float(row["lon"])
                except (TypeError, ValueError):
                    return None
        return None

    def _rank_dav_insertion_candidates(self, candidates, prev_position, next_position):
        if candidates.empty:
            return candidates
        prev_coord = self._poi_coordinate(prev_position)
        next_coord = self._poi_coordinate(next_position)
        if prev_coord is None or next_coord is None:
            return candidates.sort_values(by=["price", "name"], ascending=[True, True])

        direct_distance = geodesic(prev_coord, next_coord).kilometers

        def added_distance(row):
            try:
                attr_coord = (float(row["lat"]), float(row["lon"]))
            except (TypeError, ValueError):
                return float("inf")
            return (
                geodesic(prev_coord, attr_coord).kilometers
                + geodesic(attr_coord, next_coord).kilometers
                - direct_distance
            )
        # Use the distance to order candidates
        ranked = candidates.copy()
        ranked["_dav_added_distance"] = ranked.apply(added_distance, axis=1)
        return ranked.sort_values(
            by=["_dav_added_distance", "price", "name"], ascending=[True, True, True]
        ).drop(columns=["_dav_added_distance"], errors="ignore")

    def _postprocess_transport_modes(self, query, start, end):
        modes = list(self.innercity_transports_ranking)
        if self.transport_rules_by_distance is not None:
            distance = self.calculate_distance(query, start, end)
            modes = self.get_transport_by_distance(distance)
        return modes

    def _postprocess_collect_transport(self, query, start, end, start_time, mode):
        transports = self.collect_innercity_transport(
            query["target_city"], start, end, start_time, mode
        )
        if not isinstance(transports, list):
            return None, None
        if transport_rules_violated(self, transports):
            return None, None
        if not self._innercity_transports_valid(transports):
            return None, None
        return transports, arrived_time(start_time, transports) if transports else start_time

    def _try_build_dav_insertion(
        self, query, itinerary, day_idx, insert_idx, attr_row
    ):
        """Build one candidate itinerary with an extra attraction inserted.

        The insertion point is the gap before ``activities[insert_idx]`` on
        ``day_idx``:

            previous activity -> candidate attraction -> next activity

        This helper only checks local schedule/transport feasibility for that
        gap and rewires the next activity's incoming transport.  It returns a
        deep-copied itinerary candidate, or ``None`` if the attraction cannot
        fit.  Whole-plan commonsense and hard-logic validation happen later in
        ``_validated_dav_insertion_plan`` before the candidate can be accepted.
        """
        updated = deepcopy(itinerary)
        activities = updated[day_idx].get("activities", [])
        if insert_idx <= 0 or insert_idx >= len(activities):
            return None

        prev_act = activities[insert_idx - 1]
        next_act = activities[insert_idx]
        prev_position = self._activity_end_position(prev_act)
        next_position = self._activity_start_position(next_act)
        current_time = prev_act.get("end_time", "")
        next_start = next_act.get("start_time", "")
        if not prev_position or not next_position or not current_time or not next_start:
            return None
        if not time_compare_if_earlier_equal(current_time, next_start):
            return None

        for mode_to_attr in self._postprocess_transport_modes(query, prev_position, attr_row["name"]):
            transports_to_attr, attr_arrival = self._postprocess_collect_transport(
                query, prev_position, attr_row["name"], current_time, mode_to_attr
            )
            if transports_to_attr is None:
                continue
            scheduled = self._scheduled_poi_times(
                attr_row["name"],
                attr_arrival,
                attr_row.get("opentime", "00:00"),
                attr_row.get("endtime", "23:59"),
                self.dav_postprocess_attraction_minutes,
                "attraction",
            )
            if scheduled is None:
                continue
            attr_start, attr_end = scheduled
            if not time_compare_if_earlier_equal(attr_end, next_start):
                continue

            for mode_to_next in self._postprocess_transport_modes(query, attr_row["name"], next_position):
                transports_to_next, next_arrival = self._postprocess_collect_transport(
                    query, attr_row["name"], next_position, attr_end, mode_to_next
                )
                if transports_to_next is None:
                    continue
                if not time_compare_if_earlier_equal(next_arrival, next_start):
                    continue

                activities.insert(
                    insert_idx,
                    {
                        "position": attr_row["name"],
                        "type": "attraction",
                        "price": int(attr_row["price"]),
                        "cost": int(attr_row["price"]) * query["people_number"],
                        "tickets": query["people_number"],
                        "start_time": attr_start,
                        "end_time": attr_end,
                        "transports": transports_to_attr,
                    },
                )
                activities[insert_idx + 1]["transports"] = transports_to_next
                return updated
        return None

    def _validated_dav_insertion_plan(self, query, plan, candidate_itinerary):
        candidate = deepcopy(plan)
        candidate["itinerary"] = candidate_itinerary
        repair_full_itinerary(self, query, candidate["itinerary"])
        if not self._plan_passes_all_constraints(query, candidate):
            return None
        candidate["commonsense_pass"] = True
        candidate["hard_pass_count"] = self._hard_pass_count(query, candidate)
        return candidate

    def _postprocess_insert_attractions_for_dav(self, query, plan):
        """Conservatively insert extra along-route attractions for DAV.

        This runs only after a full valid plan exists.  Each proposed insertion
        is accepted only if whole-plan commonsense and all hard-logic
        constraints still pass, so the 85% hard-score block remains protected.
        """
        if (
            not self.enable_dav_postprocess
            or self.dav_postprocess_seconds <= 0
            or self.dav_postprocess_max_attempts <= 0
            or not isinstance(plan, dict)
            or not plan.get("itinerary")
        ):
            return self._annotate_daily_meal_check(plan)
        if not self._plan_passes_all_constraints(query, plan):
            return self._annotate_daily_meal_check(plan)

        timeout_guard = self.time_before_search + max(1, self.TIME_CUT - 5)
        deadline = min(time.time() + self.dav_postprocess_seconds, timeout_guard)
        accepted = 0
        attempts = 0
        improved = deepcopy(plan)
        day_count = max(1, len(improved.get("itinerary", [])) or int(query.get("days", 1)))
        current_attractions = sum(
            1
            for day in improved.get("itinerary", [])
            for act in day.get("activities", [])
            if act.get("type") == "attraction"
        )
        target_attractions = self.dav_postprocess_target_per_day * day_count
        max_insertions = max(0, target_attractions - current_attractions)
        if max_insertions <= 0:
            return self._annotate_daily_meal_check(improved)

        while (
            accepted < max_insertions
            and attempts < self.dav_postprocess_max_attempts
            and time.time() < deadline
        ):
            itinerary = improved.get("itinerary", [])
            existing_positions = self._plan_positions(itinerary)
            attr_candidates = self._filter_attraction_hard_candidates(
                self.memory["attractions"]
            )
            attr_candidates = attr_candidates[
                ~attr_candidates["name"].isin(existing_positions)
            ]
            if attr_candidates.empty:
                break

            inserted_this_round = False
            gap_slots = []
            for day_idx, day in enumerate(itinerary):
                activities = day.get("activities", [])
                for insert_idx in range(1, len(activities)):
                    prev_act = activities[insert_idx - 1]
                    next_act = activities[insert_idx]
                    if prev_act.get("type") == "accommodation" or next_act.get("type") == "breakfast":
                        continue
                    gap_slots.append((day_idx, insert_idx))
            if self.dav_postprocess_prioritize_large_gaps:
                gap_slots.sort(
                    key=lambda slot: self._gap_minutes(
                        itinerary[slot[0]]["activities"][slot[1] - 1],
                        itinerary[slot[0]]["activities"][slot[1]],
                    ),
                    reverse=True,
                )
            for day_idx, insert_idx in gap_slots:
                if attempts >= self.dav_postprocess_max_attempts or time.time() >= deadline:
                    break
                activities = itinerary[day_idx].get("activities", [])
                prev_act = activities[insert_idx - 1]
                next_act = activities[insert_idx]
                prev_position = self._activity_end_position(prev_act)
                next_position = self._activity_start_position(next_act)
                ranked = self._rank_dav_insertion_candidates(
                    attr_candidates, prev_position, next_position
                )
                for _, attr_row in ranked.head(self.dav_postprocess_candidates_per_gap).iterrows():
                    if attempts >= self.dav_postprocess_max_attempts or time.time() >= deadline:
                        break
                    attempts += 1
                    candidate_itinerary = self._try_build_dav_insertion(
                        query, itinerary, day_idx, insert_idx, attr_row
                    )
                    if candidate_itinerary is None:
                        continue
                    candidate = self._validated_dav_insertion_plan(
                        query, improved, candidate_itinerary
                    )
                    if candidate is None:
                        continue
                    improved = candidate
                    accepted += 1
                    inserted_this_round = True
                    print(
                        f"DAV postprocess inserted attraction: {attr_row['name']} "
                        f"(accepted={accepted}, attempts={attempts})"
                    )
                    break
                if inserted_this_round:
                    break
            if not inserted_this_round:
                break

        if accepted:
            improved["dav_postprocess_insertions"] = accepted
            improved["dav_postprocess_attempts"] = attempts
            return self._annotate_daily_meal_check(improved)
        return self._annotate_daily_meal_check(plan)

    def _plan_total_cost(self, plan):
        """Verifier-style total: activity_cost + inner-city transport cost."""
        total = 0.0
        for day in plan.get("itinerary", []) or []:
            for act in day.get("activities", []):
                total += float(act.get("cost", 0) or 0)
                for tr in act.get("transports", []) or []:
                    total += float(tr.get("cost", 0) or 0)
        return total

    def _budget_trim_repair(self, query, repaired):
        """Downgrade the most expensive non-must meals toward the cheapest option
        when the plan exceeds overall_budget (targets budget-fail FPRs, 6a).

        Only accepted when commonsense still passes AND hard_pass_count increases
        (getting under budget flips the budget line), so it never regresses.
        """
        if self.overall_budget is None:
            return repaired
        res_info = self.memory.get("restaurants")
        if res_info is None or res_info.empty:
            return repaired
        protected = set(self.must_visit_restaurant or [])
        cheapest = res_info.sort_values(by=["price", "name"]).iloc[0]
        people = query["people_number"]
        for _ in range(6):  # bounded passes
            if self._plan_total_cost(repaired) <= self.overall_budget:
                break
            target = None
            for di, day in enumerate(repaired.get("itinerary", [])):
                for ai, act in enumerate(day.get("activities", [])):
                    if act.get("type") not in {"lunch", "dinner"}:
                        continue
                    if act.get("position") in protected:
                        continue
                    cost = float(act.get("cost", 0) or 0)
                    if cost <= float(cheapest["price"]) * people:
                        continue
                    if target is None or cost > target[2]:
                        target = (di, ai, cost)
            if target is None:
                break
            candidate_itin = deepcopy(repaired["itinerary"])
            act = candidate_itin[target[0]]["activities"][target[1]]
            sched = self._replacement_activity_times(act, cheapest, "restaurant")
            if sched is None:
                break
            act["position"] = cheapest["name"]
            act["price"] = int(cheapest["price"])
            act["cost"] = int(cheapest["price"]) * people
            act["start_time"], act["end_time"] = sched
            repaired, accepted = self._try_accept_repair(query, repaired, candidate_itin)
            if not accepted:
                break
        return repaired

    def _plan_innercity_cost(self, plan):
        total = 0.0
        for day in plan.get("itinerary", []) or []:
            for act in day.get("activities", []):
                for tr in act.get("transports", []) or []:
                    total += float(tr.get("cost", 0) or 0)
        return total

    def _budget_over(self, plan):
        """True when the plan violates a total_cost or inner-city budget line."""
        if self.overall_budget is not None and self._plan_total_cost(plan) > self.overall_budget:
            return True
        if self.innercity_budget is not None and self._plan_innercity_cost(plan) > self.innercity_budget:
            return True
        return False

    def _enforce_budget_by_dropping(self, query, repaired):
        """Best-effort construction step: keep the assembled plan within its
        total_cost / inner-city budget by dropping the non-must attraction with
        the most expensive inbound transport.

        Over-budget best-effort plans are typically driven by a couple of
        far-flung attractions that force pricey taxi legs (metro is too slow to
        keep them in the day, so a mode downgrade breaks commonsense). Removing
        such an optional attraction shortens the day (never pushes activities
        past their open windows) and cuts its taxi cost outright. Each drop is
        accepted only when commonsense still passes and the hard-pass count does
        not drop, so it can only ever improve the plan; the whole pass is
        discarded unless the final plan is strictly better than the input.
        """
        if self.overall_budget is None and self.innercity_budget is None:
            return repaired
        if not self._budget_over(repaired):
            return repaired
        if not self._commonsense_passes(query, repaired):
            return repaired
        protected = set(self.must_see_attraction or [])
        baseline_count = self._hard_pass_count(query, repaired)
        current = repaired
        current_count = baseline_count
        for _ in range(max(1, self.budget_drop_max)):
            if not self._budget_over(current):
                break
            target = None  # (day_idx, act_idx, inbound_cost)
            for di, day in enumerate(current.get("itinerary", [])):
                for ai, act in enumerate(day.get("activities", [])):
                    if act.get("type") != "attraction":
                        continue
                    if act.get("position") in protected:
                        continue
                    inbound = sum(
                        float(t.get("cost", 0) or 0)
                        for t in act.get("transports", []) or []
                    )
                    if target is None or inbound > target[2]:
                        target = (di, ai, inbound)
            if target is None:
                break
            candidate = deepcopy(current)
            del candidate["itinerary"][target[0]]["activities"][target[1]]
            repair_full_itinerary(self, query, candidate["itinerary"])
            candidate = self._res_plan_shell(query, candidate["itinerary"])
            if not self._commonsense_passes(query, candidate):
                break
            if self._hard_pass_count(query, candidate) < current_count:
                break
            current = candidate
            current_count = self._hard_pass_count(query, current)
        if current_count > baseline_count:
            return current
        return repaired

    def _try_accept_repair(self, query, current_plan, candidate_itinerary):
        candidate = self._res_plan_shell(query, deepcopy(candidate_itinerary))
        repair_full_itinerary(self, query, candidate["itinerary"])
        if not self._commonsense_passes(query, candidate):
            return current_plan, False
        current_count = self._hard_pass_count(query, current_plan)
        candidate_count = self._hard_pass_count(query, candidate)
        if candidate_count > current_count:
            return candidate, True
        return current_plan, False

    def _plan_positions(self, itinerary):
        positions = set()
        for day in itinerary or []:
            for act in day.get("activities", []):
                pos = self._activity_position(act)
                if pos:
                    positions.add(pos)
        return positions

    def _hotel_repair_candidates(self):
        hotel_info = self.memory["accommodations"]
        return hotel_info[
            hotel_info.apply(self._hotel_satisfies_hard_constraints, axis=1)
        ]

    def _replace_hotel_in_itinerary(self, itinerary, hotel_row):
        updated = deepcopy(itinerary)
        for day in updated:
            for act in day.get("activities", []):
                if act.get("type") in {"accommodation", "breakfast"}:
                    act["position"] = hotel_row["name"]
                    rooms = act.get("rooms", self.required_rooms)
                    if act.get("type") == "accommodation":
                        act["price"] = int(hotel_row["price"])
                        act["cost"] = int(hotel_row["price"]) * rooms
                        act["room_type"] = hotel_row["numbed"]
                        act["rooms"] = rooms
                    else:
                        act["cost"] = int(hotel_row["price"]) * self.query["people_number"]
        return updated

    def _replacement_activity_times(self, activity, target_row, target_kind):
        """Schedule a repair replacement instead of retaining stale slot times."""
        return self._scheduled_poi_times(
            target_row["name"],
            activity.get("start_time", "00:00"),
            target_row.get("opentime", "00:00"),
            target_row.get("endtime", "23:59"),
            90 if target_kind == "attraction" else 60,
            activity.get("type"),
        )

    def _replace_early_window_activity_in_itinerary(
        self, itinerary, target_row, target_kind
    ):
        """Use the breakfast-adjacent first POI slot for an early replacement.

        Keeping an afternoon slot cannot repair ``start <= 08:10``.  This
        fallback creates the same 06:00 breakfast anchor as DFS, but leaves
        transport generation to ``repair_full_itinerary``.  The candidate is
        accepted only if that repair preserves both commonsense and hard logic.
        """
        if target_kind != "attraction":
            return None
        start_constraint = (getattr(self, "activities_arrive_time_dict", None) or {}).get(
            target_row["name"]
        )
        if (
            not start_constraint
            or start_constraint[0] != "early"
            or not time_compare_if_earlier_equal(start_constraint[1], "09:00")
        ):
            return None

        updated = deepcopy(itinerary)
        protected = set(self.must_see_attraction or []) | set(self.must_visit_restaurant or [])
        for day in updated:
            activities = day.get("activities", [])
            breakfast_idx = next(
                (idx for idx, act in enumerate(activities) if act.get("type") == "breakfast"),
                None,
            )
            if breakfast_idx is None or breakfast_idx + 1 >= len(activities):
                continue
            activity = activities[breakfast_idx + 1]
            if activity.get("type") != "attraction" or activity.get("position") in protected:
                continue
            scheduled_times = self._scheduled_poi_times(
                target_row["name"],
                "06:30",
                target_row.get("opentime", "00:00"),
                target_row.get("endtime", "23:59"),
                90,
                "attraction",
            )
            if scheduled_times is None:
                continue
            breakfast = activities[breakfast_idx]
            breakfast["start_time"], breakfast["end_time"] = "06:00", "06:30"
            activity["position"] = target_row["name"]
            activity["price"] = int(target_row["price"])
            activity["cost"] = int(target_row["price"]) * self.query["people_number"]
            activity["tickets"] = self.query["people_number"]
            activity["start_time"], activity["end_time"] = scheduled_times
            return updated
        return None

    def _replace_activity_in_itinerary(self, itinerary, target_row, target_kind):
        updated = deepcopy(itinerary)
        protected = set(self.must_see_attraction or []) | set(self.must_visit_restaurant or [])
        for day in updated:
            for act in day.get("activities", []):
                act_type = act.get("type")
                if target_kind == "attraction" and act_type != "attraction":
                    continue
                if target_kind == "restaurant" and act_type not in {"lunch", "dinner"}:
                    continue
                if act.get("position") in protected:
                    continue
                scheduled_times = self._replacement_activity_times(
                    act, target_row, target_kind
                )
                if scheduled_times is None:
                    continue
                act["position"] = target_row["name"]
                act["price"] = int(target_row["price"])
                act["cost"] = int(target_row["price"]) * self.query["people_number"]
                act["start_time"], act["end_time"] = scheduled_times
                if target_kind == "attraction":
                    act["tickets"] = self.query["people_number"]
                return updated
        return self._replace_early_window_activity_in_itinerary(
            itinerary, target_row, target_kind
        )

    def _replace_windowed_activity_in_itinerary(
        self, itinerary, target_row, target_kind
    ):
        """Place a required POI directly in its verifier time window.

        Search and ordinary replacement preserve the old activity slot.  That
        cannot repair a named POI whose verifier requires ``start <= T`` and
        ``end >= U`` when the closest generated slot starts after ``T``.  Pick
        the existing target (when present), otherwise the closest compatible
        slot, schedule it at the latest allowed start, and remove only
        same-day activities that overlap the required interval.  Transport and
        the remaining time chain are rebuilt by ``_try_accept_repair``.
        """
        name = target_row["name"]
        arrive_info = (self.activities_arrive_time_dict or {}).get(name)
        leave_info = (self.activities_leave_time_dict or {}).get(name)
        if not arrive_info and not leave_info:
            return None

        desired_start = (
            arrive_info[1]
            if arrive_info and arrive_info[0] == "early"
            else None
        )
        if desired_start is None:
            return None
        scheduled = self._scheduled_poi_times(
            name,
            desired_start,
            target_row.get("opentime", "00:00"),
            target_row.get("endtime", "23:59"),
            90 if target_kind == "attraction" else 60,
            "attraction" if target_kind == "attraction" else "dinner",
        )
        if scheduled is None:
            return None
        desired_start, desired_end = scheduled

        protected = set(self.must_see_attraction or []) | set(
            self.must_visit_restaurant or []
        )
        candidates = []
        for day_idx, day in enumerate(itinerary or []):
            for act_idx, activity in enumerate(day.get("activities", [])):
                act_type = activity.get("type")
                if target_kind == "attraction" and act_type != "attraction":
                    continue
                if target_kind == "restaurant" and act_type not in {"lunch", "dinner"}:
                    continue
                position = self._activity_position(activity)
                if position in protected and position != name:
                    continue
                distance = abs(
                    self._time_minutes(activity.get("start_time", desired_start))
                    - self._time_minutes(desired_start)
                )
                day_size = len(day.get("activities", []))
                candidates.append(
                    (position != name, day_size, distance, day_idx, act_idx)
                )
        if not candidates:
            return None

        _, _, _, day_idx, act_idx = min(candidates)
        updated = deepcopy(itinerary)
        activity = updated[day_idx]["activities"][act_idx]
        target_was_present = self._activity_position(activity) == name
        activity["position"] = name
        activity["price"] = int(target_row["price"])
        activity["cost"] = int(target_row["price"]) * self.query["people_number"]
        activity["start_time"], activity["end_time"] = desired_start, desired_end
        if target_kind == "attraction":
            activity["type"] = "attraction"
            activity["tickets"] = self.query["people_number"]

        start_min = self._time_minutes(desired_start)
        end_min = self._time_minutes(desired_end)
        cleaned = []
        for idx, other in enumerate(updated[day_idx]["activities"]):
            if idx == act_idx:
                cleaned.append(other)
                continue
            if self._is_intercity_activity(other):
                cleaned.append(other)
                continue
            if other.get("type") == "accommodation":
                other_start = self._time_minutes(other.get("start_time"))
                other_end = self._time_minutes(other.get("end_time"))
                if other_start < end_min and start_min < other_end:
                    other["start_time"] = desired_end
                cleaned.append(other)
                continue
            position = self._activity_position(other)
            if position in protected:
                cleaned.append(other)
                continue
            other_start = self._time_minutes(other.get("start_time"))
            other_end = self._time_minutes(other.get("end_time"))
            if other_start < end_min and start_min < other_end:
                continue
            # A newly inserted fixed-window stop can make the next attraction
            # unreachable before closing even when its old clock does not
            # overlap.  Leave a three-hour travel margin; later attractions
            # and meals remain available for the rest of the day.
            if (
                not target_was_present
                and other.get("type") == "attraction"
                and end_min <= other_start < end_min + 180
            ):
                continue
            cleaned.append(other)
        updated[day_idx]["activities"] = cleaned
        return updated

    def _swap_order_pair_in_itinerary(self, itinerary, before_name, after_name):
        updated = deepcopy(itinerary)
        before_ref = None
        after_ref = None
        for day_idx, day in enumerate(updated):
            for act_idx, act in enumerate(day.get("activities", [])):
                pos = self._activity_position(act)
                if pos == before_name:
                    before_ref = (day_idx, act_idx)
                elif pos == after_name:
                    after_ref = (day_idx, act_idx)
        if before_ref is None or after_ref is None:
            return None
        if before_ref < after_ref:
            return None
        b_day, b_idx = before_ref
        a_day, a_idx = after_ref
        updated[b_day]["activities"][b_idx], updated[a_day]["activities"][a_idx] = (
            updated[a_day]["activities"][a_idx],
            updated[b_day]["activities"][b_idx],
        )
        return updated

    def _hard_aware_repair_plan(self, query, completed):
        if not completed or not completed.get("itinerary"):
            return completed
        if not self._commonsense_passes(query, completed):
            return completed

        repaired = deepcopy(completed)
        positions = self._plan_positions(repaired.get("itinerary"))

        hotel_info = self.memory["accommodations"]
        hotel_candidates = self._hotel_repair_candidates()
        if not hotel_candidates.empty:
            accommodation_names = [
                act.get("position")
                for day in repaired.get("itinerary", [])
                for act in day.get("activities", [])
                if act.get("type") == "accommodation" and act.get("position")
            ]
            current_hotels = hotel_info[hotel_info["name"].isin(accommodation_names)]
            needs_hotel = not accommodation_names or current_hotels.empty or not all(
                self._hotel_satisfies_hard_constraints(row)
                for _, row in current_hotels.iterrows()
            )
            if needs_hotel:
                for _, hotel_row in hotel_candidates.iterrows():
                    candidate_itinerary = self._replace_hotel_in_itinerary(
                        repaired["itinerary"], hotel_row
                    )
                    repaired, accepted = self._try_accept_repair(query, repaired, candidate_itinerary)
                    if accepted:
                        break

        positions = self._plan_positions(repaired.get("itinerary"))
        attr_info = self.memory["attractions"]
        for name in self.must_see_attraction or []:
            match = attr_info[attr_info["name"] == name]
            if match.empty:
                continue
            candidate_itinerary = self._replace_windowed_activity_in_itinerary(
                repaired["itinerary"], match.iloc[0], "attraction"
            )
            if candidate_itinerary is None and name not in positions:
                candidate_itinerary = self._replace_activity_in_itinerary(
                    repaired["itinerary"], match.iloc[0], "attraction"
                )
            if candidate_itinerary is not None:
                repaired, _ = self._try_accept_repair(query, repaired, candidate_itinerary)
                positions = self._plan_positions(repaired.get("itinerary"))

        for attr_type in self.must_see_attraction_type or []:
            itinerary = repaired.get("itinerary", [])
            present = False
            for day in itinerary:
                for act in day.get("activities", []):
                    if act.get("type") != "attraction":
                        continue
                    match = attr_info[attr_info["name"] == act.get("position")]
                    if not match.empty and self._canon_type("attraction", match.iloc[0].get("type")) == self._canon_type("attraction", attr_type):
                        present = True
            if present:
                continue
            match = attr_info[self._type_match_mask(attr_info["type"], attr_type, "attraction")]
            if match.empty:
                continue
            candidate_itinerary = self._replace_activity_in_itinerary(
                repaired["itinerary"], match.iloc[0], "attraction"
            )
            if candidate_itinerary is not None:
                repaired, _ = self._try_accept_repair(query, repaired, candidate_itinerary)

        positions = self._plan_positions(repaired.get("itinerary"))
        res_info = self.memory["restaurants"]
        for name in self.must_visit_restaurant or []:
            match = res_info[res_info["name"] == name]
            if match.empty:
                continue
            candidate_itinerary = self._replace_windowed_activity_in_itinerary(
                repaired["itinerary"], match.iloc[0], "restaurant"
            )
            if candidate_itinerary is None and name not in positions:
                candidate_itinerary = self._replace_activity_in_itinerary(
                    repaired["itinerary"], match.iloc[0], "restaurant"
                )
            if candidate_itinerary is not None:
                repaired, _ = self._try_accept_repair(query, repaired, candidate_itinerary)
                positions = self._plan_positions(repaired.get("itinerary"))

        for cuisine in self.must_visit_restaurant_type or []:
            itinerary = repaired.get("itinerary", [])
            present = False
            for day in itinerary:
                for act in day.get("activities", []):
                    if act.get("type") not in {"lunch", "dinner"}:
                        continue
                    match = res_info[res_info["name"] == act.get("position")]
                    if not match.empty and self._canon_type("restaurant", match.iloc[0].get("cuisine")) == self._canon_type("restaurant", cuisine):
                        present = True
            if present:
                continue
            match = res_info[self._type_match_mask(res_info["cuisine"], cuisine, "restaurant")]
            if match.empty:
                continue
            candidate_itinerary = self._replace_activity_in_itinerary(
                repaired["itinerary"], match.iloc[0], "restaurant"
            )
            if candidate_itinerary is not None:
                repaired, _ = self._try_accept_repair(query, repaired, candidate_itinerary)

        for before, after in self.must_visit_order or []:
            candidate_itinerary = self._swap_order_pair_in_itinerary(
                repaired["itinerary"], before, after
            )
            if candidate_itinerary is not None:
                repaired, _ = self._try_accept_repair(query, repaired, candidate_itinerary)

        # Budget-aware trim (6a): if the plan still exceeds overall_budget, swap
        # non-must meals down toward the cheapest option to flip the budget line.
        if self.enable_fallback_hard_repair:
            repaired = self._budget_trim_repair(query, repaired)

        repaired["hard_pass_count"] = self._hard_pass_count(query, repaired)
        repaired["commonsense_pass"] = True
        repaired["fallback_hard_repaired"] = repaired["hard_pass_count"] > completed.get("hard_pass_count", -1)
        return repaired

    def _after_append_activity(self, query, plan, day_idx):
        if not getattr(self, "_enable_plan_sync", False):
            return True
        return sync_itinerary_commonsense(self, query, plan, day_idx)

    def _repair_itinerary_times(self, itinerary):
        forward_time_chain(itinerary)

    def _drop_broken_activities(self, itinerary):
        for day in itinerary:
            cleaned = []
            for act in day.get("activities", []):
                if self._is_intercity_activity(act):
                    cleaned.append(act)
                    continue
                st = act.get("start_time")
                et = act.get("end_time")
                transports = act.get("transports") or []
                broken = False
                if not st or not et:
                    broken = True
                elif self._time_minutes(st) >= self._time_minutes(et) and act.get("type") != "accommodation":
                    broken = True
                elif transports:
                    te = transports[-1].get("end_time")
                    if te and self._time_minutes(te) > self._time_minutes(st):
                        broken = True
                if not broken:
                    cleaned.append(act)
            day["activities"] = cleaned

    def _ensure_day_count(self, query, itinerary):
        target_days = query["days"]
        while len(itinerary) < target_days:
            itinerary.append({"day": len(itinerary) + 1, "activities": []})
        while len(itinerary) > target_days:
            itinerary.pop()
        for idx, day in enumerate(itinerary):
            day["day"] = idx + 1

    def _extract_tail_state(self, itinerary):
        last_day_idx = -1
        last_act = None
        for d_idx, day in enumerate(itinerary):
            acts = day.get("activities", [])
            if acts:
                last_day_idx = d_idx
                last_act = acts[-1]
        if last_act is None:
            return 0, "", ""
        return last_day_idx, last_act.get("end_time", ""), self._activity_position(last_act)

    def _required_rooms_for_hotel(self, query, hotel_sel):
        room_type = hotel_sel["numbed"]
        if self.room_number is not None:
            return self.room_number
        return int((query["people_number"] - 1) / room_type) + 1

    def _add_breakfast_on_day(self, itinerary, day_idx, poi_plan, transports_sel=None):
        if "accommodation" not in poi_plan:
            return False
        hotel_name = poi_plan["accommodation"]["name"]
        prev_pos = ""
        acts = itinerary[day_idx].get("activities", [])
        if acts:
            prev_pos = self._activity_position(acts[-1])
        innercity = transports_sel or []
        if prev_pos and prev_pos != hotel_name:
            innercity = None
        itinerary[day_idx]["activities"] = self.add_poi(
            activities=itinerary[day_idx]["activities"],
            position=hotel_name,
            poi_type="breakfast",
            price=0,
            cost=0,
            start_time="06:00",
            end_time="06:30",
            innercity_transports=innercity if innercity is not None else [],
        )
        if getattr(self, "query", None) is not None:
            if not self._after_append_activity(self.query, itinerary, day_idx):
                itinerary[day_idx]["activities"].pop()
                return False
        return True

    def _try_append_hotel_stay(self, query, poi_plan, itinerary, day_idx, current_time, current_position, deadline):
        if time.time() > deadline:
            return None
        if "accommodation" not in poi_plan:
            return None
        hotel_sel = poi_plan["accommodation"]
        required_rooms = self._required_rooms_for_hotel(query, hotel_sel)
        transports_ranking = self.innercity_transports_ranking
        if self.transport_rules_by_distance is not None:
            temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
            transports_ranking = self.get_transport_by_distance(temp_distance)
        for trans_type_sel in transports_ranking:
            if time.time() > deadline:
                return None
            if current_position == hotel_sel["name"]:
                transports_sel = []
                arrived_time = current_time
            else:
                transports_sel = self.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    hotel_sel["name"],
                    current_time,
                    trans_type_sel,
                )
                if not isinstance(transports_sel, list):
                    continue
                arrived_time = transports_sel[-1]["end_time"] if transports_sel else current_time
            arrived_time = clamp_time_to_day_end(arrived_time)
            if not self._innercity_transports_valid(transports_sel):
                continue
            if self._arrived_time_too_late_for_hotel(arrived_time):
                continue
            self.add_accommodation(
                current_plan=itinerary,
                hotel_sel=hotel_sel,
                current_day=day_idx,
                arrived_time=arrived_time,
                required_rooms=required_rooms,
                transports_sel=transports_sel,
            )
            return day_idx, "00:00", hotel_sel["name"]
        return None

    def _try_append_return_transport(self, query, poi_plan, itinerary, day_idx, current_time, current_position, deadline):
        if time.time() > deadline or "back_transport" not in poi_plan:
            return False
        back_transport = poi_plan["back_transport"]
        transports_ranking = self.innercity_transports_ranking
        if self.transport_rules_by_distance is not None:
            temp_distance = self.calculate_distance(query, current_position, back_transport["From"])
            transports_ranking = self.get_transport_by_distance(temp_distance)
        for trans_type_sel in transports_ranking:
            if time.time() > deadline:
                return False
            transports_sel = self.collect_innercity_transport(
                query["target_city"],
                current_position,
                back_transport["From"],
                current_time,
                trans_type_sel,
            )
            if not isinstance(transports_sel, list):
                continue
            arrived_time = transports_sel[-1]["end_time"] if transports_sel else current_time
            if not time_compare_if_earlier_equal(arrived_time, back_transport["BeginTime"]):
                continue
            acts = itinerary[day_idx]["activities"]
            if acts and self._is_intercity_activity(acts[-1]):
                acts.pop()
            itinerary[day_idx]["activities"] = self.add_intercity_transport(
                itinerary[day_idx]["activities"],
                back_transport,
                innercity_transports=transports_sel,
                tickets=query["people_number"],
            )
            return True
        return False

    def _fallback_complete_for_commonsense(self, query, poi_plan, seed_plan):
        if not seed_plan or not poi_plan or "go_transport" not in poi_plan or "back_transport" not in poi_plan:
            return None

        deadline = time.time() + self.FALLBACK_COMPLETE_SEC
        itinerary = deepcopy(seed_plan.get("itinerary", []))
        if self._plan_activity_count(itinerary) == 0:
            return None

        self._repair_itinerary_times(itinerary)
        self._drop_broken_activities(itinerary)
        self._ensure_day_count(query, itinerary)

        if not incremental_commonsense_ok(query, itinerary, lang=getattr(self, "lang", None)):
            repair_full_itinerary(self, query, itinerary)

        if self._commonsense_passes(query, self._res_plan_shell(query, itinerary)):
            completed = self._res_plan_shell(query, itinerary)
            completed.update({
                k: v for k, v in seed_plan.items()
                if k not in completed and k != "itinerary"
            })
            completed["commonsense_pass"] = True
            completed["fallback_completed"] = True
            return self._hard_aware_repair_plan(query, completed)

        day_idx, current_time, current_position = self._extract_tail_state(itinerary)
        last_day_idx = query["days"] - 1

        if day_idx < 0:
            day_idx = 0
            current_time = poi_plan["go_transport"]["EndTime"]
            current_position = poi_plan["go_transport"]["To"]

        while day_idx < last_day_idx:
            if time.time() > deadline:
                break
            acts = itinerary[day_idx].get("activities", [])
            last_act = acts[-1] if acts else None
            if last_act and last_act.get("type") == "accommodation":
                day_idx += 1
                if not self._add_breakfast_on_day(itinerary, day_idx, poi_plan):
                    break
                current_time = "06:30"
                current_position = poi_plan["accommodation"]["name"]
                continue
            state = self._try_append_hotel_stay(
                query, poi_plan, itinerary, day_idx, current_time, current_position, deadline
            )
            if state is None:
                break
            day_idx, current_time, current_position = state
            if day_idx < last_day_idx:
                day_idx += 1
                if not self._add_breakfast_on_day(itinerary, day_idx, poi_plan):
                    break
                current_time = "06:30"
                current_position = poi_plan["accommodation"]["name"]

        if day_idx == last_day_idx:
            acts = itinerary[day_idx].get("activities", [])
            if current_time in ("", "00:00") and "accommodation" in poi_plan:
                has_breakfast = any(a.get("type") == "breakfast" for a in acts)
                if not has_breakfast:
                    self._add_breakfast_on_day(itinerary, day_idx, poi_plan)
                    current_time = "06:30"
                    current_position = poi_plan["accommodation"]["name"]
            elif current_time == "00:00" and acts:
                last_act = acts[-1]
                current_time = last_act.get("end_time", current_time)
                current_position = self._activity_position(last_act) or current_position

            if not (acts and self._is_intercity_activity(acts[-1])):
                self._try_append_return_transport(
                    query, poi_plan, itinerary, day_idx, current_time, current_position, deadline
                )

        self._repair_itinerary_times(itinerary)
        self._drop_broken_activities(itinerary)
        self._ensure_day_count(query, itinerary)
        repair_full_itinerary(self, query, itinerary)

        completed = self._res_plan_shell(query, itinerary)
        if not self._commonsense_passes(query, completed):
            last_day = itinerary[last_day_idx]
            acts = last_day.get("activities", [])
            while len(acts) > 1 and time.time() < deadline:
                acts.pop()
                self._try_append_return_transport(
                    query, poi_plan, itinerary, last_day_idx,
                    self._extract_tail_state(itinerary)[1],
                    self._extract_tail_state(itinerary)[2],
                    deadline,
                )
                self._repair_itinerary_times(itinerary)
                self._drop_broken_activities(itinerary)
                completed = self._res_plan_shell(query, itinerary)
                if self._commonsense_passes(query, completed):
                    break

        if not self._commonsense_passes(query, completed):
            return None

        completed = self._hard_aware_repair_plan(query, completed)
        logical_result = evaluate_constraints_py(query["hard_logic_py"], completed, verbose=False)
        completed["hard_pass_count"] = int(np.sum(logical_result))
        completed["commonsense_pass"] = True
        completed["fallback_completed"] = True
        completed["backtrack_count"] = self.backtrack_count
        return completed

    def _finalize_best_effort_plan(self, query, seed_plan, poi_plan=None):
        poi_plan = poi_plan or self._current_poi_plan
        if seed_plan is None:
            return None
        if poi_plan and seed_plan.get("itinerary"):
            completed = self._fallback_complete_for_commonsense(query, poi_plan, seed_plan)
            if completed is not None:
                # Budget feasibility is part of *constructing* the best-effort
                # plan, not a post-hoc repair: a timed-out search often surfaces a
                # partial that is complete but a hair over the total_cost budget
                # because of one far-flung attraction's taxi leg. Drop the priciest
                # non-must attraction until the budget line holds (only accepts
                # strict hard-pass improvements, so it never regresses).
                if self.enable_budget_drop_repair:
                    completed = self._enforce_budget_by_dropping(query, completed)
                    completed["hard_pass_count"] = self._hard_pass_count(query, completed)
                print("Timeout fallback: completed plan passes commonsense checks.")
                return completed
            print("Timeout fallback: could not repair plan to commonsense; returning partial seed.")
            seed = deepcopy(seed_plan)
            itinerary = seed.get("itinerary", [])
            if itinerary:
                repair_full_itinerary(self, query, itinerary)
            seed["commonsense_pass"] = (
                self._commonsense_passes(query, self._res_plan_shell(query, itinerary))
                if itinerary
                else False
            )
            seed["fallback_completed"] = False
            seed["backtrack_count"] = self.backtrack_count
            return seed
        seed_plan = deepcopy(seed_plan)
        seed_plan["backtrack_count"] = self.backtrack_count
        seed_plan["commonsense_pass"] = False
        seed_plan["fallback_completed"] = False
        return seed_plan

    def _best_effort_plan(self, query=None, plan=None, poi_plan=None):
        """
        超时兜底：取 hard constraint 通过最多的 partial plan，再 fallback 补全为
        commonsense 合法路径；避免直接返回缺天数/缺返程的残缺行程。
        """
        query = query if query is not None else self.query
        poi_plan = poi_plan or self._current_poi_plan
        plan = plan if plan is not None else self._current_dfs_plan
        if query is not None and plan is not None:
            self._update_best_plan_from_partial(query, plan)

        for cand in (self.least_plan_logic, self.least_plan_comm, self.least_plan_schema):
            if cand is not None and cand.get("itinerary"):
                finalized = self._finalize_best_effort_plan(query, cand, poi_plan)
                if finalized is not None:
                    return finalized
        if isinstance(plan, list) and self._plan_activity_count(plan) > 0:
            shell = self._res_plan_shell(query, deepcopy(plan))
            finalized = self._finalize_best_effort_plan(query, shell, poi_plan)
            if finalized is not None:
                return finalized
        fallback = deepcopy(self.default_plan)
        fallback["backtrack_count"] = self.backtrack_count
        return fallback

    def _is_search_cutoff_error(self, plan):
        return (
            isinstance(plan, dict)
            and not plan.get("itinerary")
            and plan.get("error") == "No solution found before search cutoff"
        )

    def _best_effort_search_result(self, query=None, plan=None, poi_plan=None):
        fallback_plan = self._best_effort_plan(query, plan, poi_plan)
        if isinstance(fallback_plan, dict) and fallback_plan.get("itinerary"):
            return True, fallback_plan
        return False, fallback_plan

    def _visited_contains(self, visited_items, value):
        for item in visited_items:
            if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
                try:
                    if value in list(item):
                        return True
                except TypeError:
                    pass
            elif item == value:
                return True
        return False

    def _dfs_log(self, *args, **kwargs):
        dfs_log(self, *args, **kwargs)

    def _cache_value_key(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            return tuple(
                sorted((key, self._cache_value_key(val)) for key, val in value.items())
            )
        if isinstance(value, (list, tuple, set)):
            return tuple(self._cache_value_key(item) for item in value)
        return value

    def _candidate_frame_key(self, candidates):
        name_key = None
        if "name" in candidates.columns:
            name_key = tuple(candidates["name"].astype(str).tolist())
        return (len(candidates), tuple(candidates.index.tolist()), name_key)

    def _dfs_search_state(self, query, plan, current_day, current_time, current_position):
        return SearchState.from_dfs(self, query, current_day, current_time, current_position, plan)

    def _try_append_back_transport(
        self, query, poi_plan, plan, current_day, current_time, current_position
    ):
        """Try inner-city legs to the return station, append back intercity, validate."""
        ensure_day_plan(plan, current_day)
        destination = poi_plan["back_transport"]["From"]
        back_begin = poi_plan["back_transport"]["BeginTime"]

        for trans_type in transports_ranking_to(self, query, current_position, destination):
            self.search_nodes += 1
            self._dfs_log("collecting innercity transport to back-transport")
            if current_position == destination:
                transports_sel = []
                arrival = current_time
            else:
                transports_sel = self.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    destination,
                    current_time,
                    trans_type,
                )
                if not isinstance(transports_sel, list):
                    self.backtrack_count += 1
                    self._dfs_log("inner-city transport error, backtrack...")
                    continue
                arrival = arrived_time(current_time, transports_sel)

            if not self.too_many_backtrack and not time_compare_if_earlier_equal(
                arrival, back_begin
            ):
                self.backtrack_count += 1
                self._dfs_log("Fail to catch the back transport")
                continue

            if transport_rules_violated(self, transports_sel):
                if not self.too_many_backtrack:
                    self.backtrack_count += 1
                continue

            plan[current_day]["activities"] = self.add_intercity_transport(
                plan[current_day]["activities"],
                poi_plan["back_transport"],
                innercity_transports=transports_sel,
                tickets=query["people_number"],
            )

            if not self.too_many_backtrack and self.check_budgets(plan):
                plan[current_day]["activities"].pop()
                return False, plan

            repair_full_itinerary(self, query, plan)
            res_bool, res_plan = self.constraints_validation(query, plan, poi_plan)
            if res_bool:
                return True, res_plan
            plan[current_day]["activities"].pop()
            self.backtrack_count += 1

        return False, plan

    def _filter_restaurant_hard_candidates(self, res_info):
        cache_key = (
            "restaurant_hard",
            self._candidate_frame_key(res_info),
            self._cache_value_key(self.must_not_visit_restaurant),
            self._cache_value_key(self.must_not_visit_restaurant_type),
            self._cache_value_key(self.must_visit_restaurant),
            self._cache_value_key(self.must_visit_restaurant_type),
        )
        cached = self._candidate_static_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        candidate_res_list = res_info.copy()
        if self.must_not_visit_restaurant is not None:
            candidate_res_list = candidate_res_list[
                ~candidate_res_list["name"].isin(self.must_not_visit_restaurant)
            ]
        if self.must_not_visit_restaurant_type is not None:
            # Canonicalize DB cuisine the same way the verifier does (e.g. DB "cafe"
            # -> "coffee shop"), then match case-insensitively, so alias/case variants
            # are excluded consistently with the official eval.
            from chinatravel.symbol_verification.concept_func import normalize_concept_value
            _excluded = {normalize_concept_value("restaurant", str(t)).lower() for t in self.must_not_visit_restaurant_type}
            _canon = candidate_res_list["cuisine"].astype(str).map(
                lambda c: str(normalize_concept_value("restaurant", c)).lower()
            )
            candidate_res_list = candidate_res_list[~_canon.isin(_excluded)]
        if self.must_visit_restaurant is not None:
            for must_name in self.must_visit_restaurant:
                if must_name not in candidate_res_list["name"].values:
                    must_res = res_info[res_info["name"] == must_name]
                    if not must_res.empty:
                        candidate_res_list = pd.concat(
                            [candidate_res_list, must_res]
                        ).drop_duplicates()
        if self.must_visit_restaurant_type is not None:
            found_types = candidate_res_list["cuisine"].unique()
            missing_types = [
                t for t in self.must_visit_restaurant_type if t not in found_types
            ]
            if missing_types:
                self._dfs_log(
                    f"[Warning] must visit restaurant type:{missing_types} is not in candidates"
                )
        self._candidate_static_cache[cache_key] = candidate_res_list.copy()
        return candidate_res_list

    def _filter_dynamic_poi_candidates(
        self,
        query,
        current_position,
        current_time,
        candidates,
        visited_indices,
        poi_kind,
        *,
        use_attraction_budget_key,
    ):
        drop_idx = flatten_visiting_indices(visited_indices)
        open_cache_key = (
            poi_kind,
            "open_at",
            self._candidate_frame_key(candidates),
            current_time,
        )
        open_candidates = self._candidate_static_cache.get(open_cache_key)
        if open_candidates is None:
            open_candidates = filter_open_at_time(candidates, current_time)
            self._candidate_static_cache[open_cache_key] = open_candidates.copy()
        candidates = open_candidates.drop(index=drop_idx, errors="ignore")
        previous_context = self._dynamic_ranking_context
        self._current_rank_time = current_time
        self._dynamic_ranking_context = self._build_dynamic_ranking_context(
            query, current_time, poi_kind
        )
        try:
            return rank_poi_dataframe(
                self,
                query,
                current_position,
                candidates,
                poi_kind,
                use_attraction_budget_key=use_attraction_budget_key,
            )
        finally:
            self._dynamic_ranking_context = previous_context

    def _required_restaurant_name_candidates(self, res_info):
        must_candidates = pd.DataFrame()
        if self.must_visit_restaurant is None:
            return must_candidates
        for must_name in self.must_visit_restaurant:
            if self._visited_contains(self.restaurant_names_visiting, must_name):
                continue
            must_res = res_info[res_info["name"] == must_name]
            if not must_res.empty:
                must_candidates = pd.concat([must_candidates, must_res]).drop_duplicates()
        return must_candidates

    def _required_restaurant_type_candidates(self, res_info):
        must_type_candidates = pd.DataFrame()
        if self.must_visit_restaurant_type is None:
            return must_type_candidates
        for cuisine in self.must_visit_restaurant_type:
            if self._visited_contains(self.food_type_visiting, cuisine):
                continue
            must_type = res_info[self._type_match_mask(res_info["cuisine"], cuisine, "restaurant")]
            if not must_type.empty:
                must_type_candidates = pd.concat(
                    [must_type_candidates, must_type]
                ).drop_duplicates()
        return must_type_candidates

    def _prepare_restaurant_candidates(
        self, query, current_position, current_time, res_info
    ):
        candidate_res_list = self._filter_restaurant_hard_candidates(res_info)
        candidate_res_ranked = self._filter_dynamic_poi_candidates(
            query,
            current_position,
            current_time,
            candidate_res_list,
            self.restaurants_visiting,
            "restaurant",
            use_attraction_budget_key=False,
        )
        n = self._effective_top_k()
        must_candidates = self._required_restaurant_name_candidates(res_info)
        must_type_candidates = self._required_restaurant_type_candidates(res_info)
        top_candidates = candidate_res_ranked.iloc[
            : min(n, len(candidate_res_ranked))
        ].copy()
        return must_candidates, must_type_candidates, top_candidates

    def _try_restaurant_candidate(
        self,
        query,
        poi_plan,
        plan,
        current_day,
        current_time,
        current_position,
        poi_type,
        poi_sel,
        res_info,
        *,
        skip_name_dup: bool,
        skip_cuisine_dup: bool,
    ):
        if skip_name_dup and poi_sel["name"] in self.restaurant_names_visiting:
            return False, plan
        if skip_cuisine_dup and poi_sel["cuisine"] in self.food_type_visiting:
            return False, plan

        destination = poi_sel["name"]
        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]

        for trans_type in transports_ranking_to(self, query, current_position, destination):
            self.search_nodes += 1
            transports_sel = self.collect_innercity_transport(
                query["target_city"],
                current_position,
                destination,
                current_time,
                trans_type,
            )
            if not isinstance(transports_sel, list):
                self.backtrack_count += 1
                self._dfs_log("inner-city transport error, backtrack...")
                continue

            arrival = arrived_time(current_time, transports_sel)
            if is_closed_at_arrival(opentime, endtime, arrival, allow_overnight=True):
                self.backtrack_count += 1
                self._dfs_log("The restaurant is closed now...")
                continue

            if transport_rules_violated(self, transports_sel):
                if not self.too_many_backtrack:
                    self.backtrack_count += 1
                continue

            scheduled_times = self._scheduled_poi_times(
                poi_sel["name"], arrival, opentime, endtime,
                self._default_meal_minutes(60), poi_type
            )
            if scheduled_times is None:
                self.backtrack_count += 1
                continue
            act_start_time, act_end_time = scheduled_times

            visiting = VisitingSnapshot.capture(self)
            try:
                plan = self.add_restaurant(
                    plan,
                    poi_type,
                    poi_sel,
                    current_day,
                    arrival,
                    transports_sel,
                    start_time=act_start_time,
                    end_time=act_end_time,
                )
                if (
                    len(plan[current_day]["activities"]) == 0
                    or plan[current_day]["activities"][-1].get("position") != poi_sel["name"]
                ):
                    self.backtrack_count += 1
                    continue
            except Exception:
                self.backtrack_count += 1
                self._dfs_log("add_restaurant failed, backtrack...")
                continue

            self._dfs_log(
                f"add restaurant: {poi_sel['name']}, type: {poi_sel['cuisine']}"
            )
            res_idx = res_info[res_info["name"] == poi_sel["name"]].index
            self.restaurants_visiting.append(res_idx)
            self.food_type_visiting.append(poi_sel["cuisine"])
            self.restaurant_names_visiting.append(poi_sel["name"])

            new_time = plan[current_day]["activities"][-1]["end_time"]
            success, plan = self.dfs_poi(
                query, poi_plan, plan, new_time, poi_sel["name"], current_day
            )
            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan

            self.backtrack_count += 1
            self._dfs_log("add_restaurant failed, backtrack...")
            self._pop_last_activity_if_matches(
                plan, current_day, poi_type, poi_sel["name"]
            )
            visiting.restore(self)

        return False, plan

    def _dfs_explore_restaurants(
        self,
        query,
        poi_plan,
        plan,
        current_day,
        current_time,
        current_position,
        poi_type,
    ):
        res_info = self.memory["restaurants"]
        must_candidates, must_type_candidates, top_candidates = (
            self._prepare_restaurant_candidates(
                query, current_position, current_time, res_info
            )
        )

        if self.must_visit_restaurant is not None:
            flag, _ = self.check_constraint(
                plan, {"must_visit_restaurant": self.must_visit_restaurant}
            )
            if not flag:
                for _, poi_sel in must_candidates.iterrows():
                    success, plan = self._try_restaurant_candidate(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                        poi_sel,
                        res_info,
                        skip_name_dup=True,
                        skip_cuisine_dup=False,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

        if self.must_visit_restaurant_type is not None:
            flag, _ = self.check_constraint(
                plan, {"must_visit_restaurant_type": self.must_visit_restaurant_type}
            )
            if not flag:
                for _, poi_sel in must_type_candidates.iterrows():
                    success, plan = self._try_restaurant_candidate(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                        poi_sel,
                        res_info,
                        skip_name_dup=True,
                        skip_cuisine_dup=False,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

        for _, poi_sel in top_candidates.iterrows():
            success, plan = self._try_restaurant_candidate(
                query,
                poi_plan,
                plan,
                current_day,
                current_time,
                current_position,
                poi_type,
                poi_sel,
                res_info,
                skip_name_dup=True,
                skip_cuisine_dup=True,
            )
            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan

        return False, plan

    def _attraction_stage(self, query, current_day, current_time, candidates_type):
        if (
            current_day == 0
            and time_compare_if_earlier_equal("14:00", current_time)
            and "dinner" in candidates_type
        ):
            return 2
        if "lunch" in candidates_type and "dinner" in candidates_type:
            return 1
        if "lunch" not in candidates_type and "dinner" in candidates_type:
            return 2
        return 0

    def _attraction_stage_allows(self, stage, act_end_time):
        if stage == 1:
            return time_compare_if_earlier_equal(act_end_time, "12:00")
        if stage == 2:
            return time_compare_if_earlier_equal(act_end_time, "19:00")
        return True

    def _filter_attraction_hard_candidates(self, attr_info):
        cache_key = (
            "attraction_hard",
            self._candidate_frame_key(attr_info),
            self._cache_value_key(self.only_free_attractions),
            self._cache_value_key(self.must_not_see_attraction),
            self._cache_value_key(self.must_not_see_attraction_type),
            self._cache_value_key(self.must_see_attraction),
            self._cache_value_key(self.must_see_attraction_type),
        )
        cached = self._candidate_static_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        candidate_attr_list = attr_info.copy()
        if self.only_free_attractions is not None and self.only_free_attractions:
            candidate_attr_list = candidate_attr_list[candidate_attr_list["price"] == 0]
        if self.must_not_see_attraction is not None:
            candidate_attr_list = candidate_attr_list[
                ~candidate_attr_list["name"].isin(self.must_not_see_attraction)
            ]
        if self.must_not_see_attraction_type is not None:
            # Canonicalize DB types the same way the verifier does (alias + case),
            # so forbidden types are excluded consistently regardless of the DB's
            # inconsistent casing/aliases across cities.
            from chinatravel.symbol_verification.concept_func import normalize_concept_value
            _excluded_types = {normalize_concept_value("attraction", str(t)).lower() for t in self.must_not_see_attraction_type}
            _canon_attr = candidate_attr_list["type"].astype(str).map(
                lambda t: str(normalize_concept_value("attraction", t)).lower()
            )
            candidate_attr_list = candidate_attr_list[~_canon_attr.isin(_excluded_types)]
        if self.must_see_attraction is not None:
            for must_name in self.must_see_attraction:
                if must_name not in candidate_attr_list["name"].values:
                    must_attr = attr_info[attr_info["name"] == must_name]
                    if not must_attr.empty:
                        candidate_attr_list = pd.concat(
                            [candidate_attr_list, must_attr]
                        ).drop_duplicates()
        if self.must_see_attraction_type is not None:
            found_types = candidate_attr_list["type"].unique()
            missing_types = [
                t for t in self.must_see_attraction_type if t not in found_types
            ]
            if missing_types:
                self._dfs_log(
                    f"[Warning] must see attraction type:{missing_types} is not in attraction candidates"
                )
        self._candidate_static_cache[cache_key] = candidate_attr_list.copy()
        return candidate_attr_list

    def _required_attraction_name_candidates(self, attr_info):
        must_candidates = pd.DataFrame()
        if self.must_see_attraction is None:
            return must_candidates
        for must_name in self.must_see_attraction:
            if self._visited_contains(self.attractions_visiting, must_name):
                continue
            must_attr = attr_info[attr_info["name"] == must_name]
            if not must_attr.empty:
                must_candidates = pd.concat(
                    [must_candidates, must_attr]
                ).drop_duplicates()
        if not must_candidates.empty:
            must_candidates = must_candidates.copy()
            must_candidates["open_duration"] = must_candidates.apply(
                lambda row: get_time_delta(row["opentime"], row["endtime"]),
                axis=1,
            )
            must_candidates = must_candidates.sort_values(
                by="open_duration", ascending=True
            ).reset_index(drop=True)
        return must_candidates

    def _required_attraction_type_candidates(self, attr_info):
        must_type_candidates = pd.DataFrame()
        if self.must_see_attraction_type is None:
            return must_type_candidates
        for must_type in self.must_see_attraction_type:
            if self._visited_contains(self.spot_type_visiting, must_type):
                continue
            must_attr = attr_info[self._type_match_mask(attr_info["type"], must_type, "attraction")]
            if not must_attr.empty:
                must_type_candidates = pd.concat(
                    [must_type_candidates, must_attr]
                ).drop_duplicates()
        return must_type_candidates

    def _prepare_attraction_candidates(
        self, query, current_position, current_time, attr_info
    ):
        candidate_attr_list = self._filter_attraction_hard_candidates(attr_info)
        candidate_attr_ranked = self._filter_dynamic_poi_candidates(
            query,
            current_position,
            current_time,
            candidate_attr_list,
            self.attractions_visiting,
            "attraction",
            use_attraction_budget_key=True,
        )
        n = self._effective_top_k()
        must_candidates = self._required_attraction_name_candidates(attr_info)
        must_type_candidates = self._required_attraction_type_candidates(attr_info)
        top_candidates = candidate_attr_ranked.iloc[
            : min(n, len(candidate_attr_ranked))
        ].copy()
        return must_candidates, must_type_candidates, top_candidates

    def _try_attraction_candidate(
        self,
        query,
        poi_plan,
        plan,
        current_day,
        current_time,
        current_position,
        poi_type,
        poi_sel,
        attr_info,
        stage,
        *,
        skip_name_dup: bool,
        skip_type_dup: bool,
    ):
        if skip_name_dup and poi_sel["name"] in self.attraction_names_visiting:
            return False, plan
        if skip_type_dup and poi_sel["type"] in self.spot_type_visiting:
            return False, plan

        self._dfs_log(f"lookahead to add attraction, candidate: {poi_sel['name']}")
        destination = poi_sel["name"]
        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]

        for trans_type in transports_ranking_to(self, query, current_position, destination):
            self.search_nodes += 1
            transports_sel = self.collect_innercity_transport(
                query["target_city"],
                current_position,
                destination,
                current_time,
                trans_type,
            )
            if not isinstance(transports_sel, list):
                self.backtrack_count += 1
                self._dfs_log("inner-city transport error, backtrack...")
                continue

            arrival = arrived_time(current_time, transports_sel)
            if is_closed_at_arrival(opentime, endtime, arrival, allow_overnight=False):
                self.backtrack_count += 1
                self._dfs_log(
                    f"{poi_sel['name']} closed at {endtime}, start time: {current_time}, "
                    f"arrival time: {arrival}, backtrack..."
                )
                continue

            if transport_rules_violated(self, transports_sel):
                if not self.too_many_backtrack:
                    self.backtrack_count += 1
                continue

            scheduled_times = self._scheduled_poi_times(
                poi_sel["name"], arrival, opentime, endtime,
                self._default_attraction_minutes(90), poi_type
            )
            if scheduled_times is None:
                self.backtrack_count += 1
                continue
            act_start_time, act_end_time = scheduled_times

            if not self._attraction_stage_allows(stage, act_end_time):
                continue

            visiting = VisitingSnapshot.capture(self)
            plan[current_day]["activities"] = self.add_poi(
                activities=plan[current_day]["activities"],
                position=poi_sel["name"],
                poi_type=poi_type,
                price=int(poi_sel["price"]),
                cost=int(poi_sel["price"]) * query["people_number"],
                start_time=act_start_time,
                end_time=act_end_time,
                innercity_transports=transports_sel,
            )
            plan[current_day]["activities"][-1]["tickets"] = query["people_number"]

            if self._enable_plan_sync and not self._after_append_activity(
                query, plan, current_day
            ):
                self._pop_last_activity_if_matches(
                    plan, current_day, poi_type, poi_sel["name"]
                )
                self.backtrack_count += 1
                continue

            self._dfs_log(
                f"add attraction: {poi_sel['name']}, type: {poi_sel['type']}"
            )
            attr_idx = attr_info[attr_info["name"] == poi_sel["name"]].index
            self.attractions_visiting.append(attr_idx)
            self.spot_type_visiting.append(poi_sel["type"])
            self.attraction_names_visiting.append(poi_sel["name"])

            success, plan = self.dfs_poi(
                query, poi_plan, plan, act_end_time, poi_sel["name"], current_day
            )
            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan

            self.backtrack_count += 1
            self._dfs_log("add_attraction failed, backtrack...")
            self._pop_last_activity_if_matches(
                plan, current_day, poi_type, poi_sel["name"]
            )
            visiting.restore(self)

        return False, plan

    def _dfs_explore_attractions(
        self,
        query,
        poi_plan,
        plan,
        current_day,
        current_time,
        current_position,
        poi_type,
        candidates_type,
    ):
        attr_info = self.memory["attractions"]
        must_candidates, must_type_candidates, top_candidates = (
            self._prepare_attraction_candidates(
                query, current_position, current_time, attr_info
            )
        )
        stage = self._attraction_stage(query, current_day, current_time, candidates_type)

        if self.must_see_attraction is not None:
            flag, _ = self.check_constraint(
                plan, {"must_see_attraction": self.must_see_attraction}
            )
            if not flag:
                for _, poi_sel in must_candidates.iterrows():
                    success, plan = self._try_attraction_candidate(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                        poi_sel,
                        attr_info,
                        stage,
                        skip_name_dup=True,
                        skip_type_dup=False,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

        if self.must_see_attraction_type is not None:
            flag, _ = self.check_constraint(
                plan, {"must_see_attraction_type": self.must_see_attraction_type}
            )
            if not flag:
                for _, poi_sel in must_type_candidates.iterrows():
                    success, plan = self._try_attraction_candidate(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                        poi_sel,
                        attr_info,
                        stage,
                        skip_name_dup=True,
                        skip_type_dup=False,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

        for _, poi_sel in top_candidates.iterrows():
            success, plan = self._try_attraction_candidate(
                query,
                poi_plan,
                plan,
                current_day,
                current_time,
                current_position,
                poi_type,
                poi_sel,
                attr_info,
                stage,
                skip_name_dup=True,
                skip_type_dup=True,
            )
            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan

        return False, plan

    def dfs_poi(self, query, poi_plan, plan, current_time, current_position, current_day=0):
        if current_day >= query["days"]:
            return False, plan

        self._current_dfs_plan = plan
        self._current_poi_plan = poi_plan
        self._current_rank_day = current_day
        self._current_rank_time = current_time
        self._dfs_log("----------------------------------calling dfs_poi-----------------------------------------")
        self._dfs_log(f"current_day: {current_day}")
        self._dfs_log(f"current_time: {current_time}")
        self._dfs_log(f"current_position: {current_position}")
        self._dfs_log(self.backtrack_count)
        # if self.backtrack_count > 5800 or time.time() - self.time_before_search + 20 > self.TIME_CUT + self.llm_inference_time_count:
        #     self.too_many_backtrack = True
        if self._search_time_exceeded():
            self.too_many_backtrack = True
            self.stop_search = True
            self.default_plan["backtrack_count"] = self.backtrack_count
            return self._best_effort_search_result(query, plan, poi_plan)

        # Dominance pruning (6b): a state defined by (day, time, position, and the
        # set of still-pending required items) is deterministic; re-entering it
        # after a prior failure cannot succeed, so skip it. pending_must is part
        # of the key so states differing only in required-coverage are distinct.
        if self.enable_dfs_memoization and not self.too_many_backtrack:
            state_key = (
                current_day,
                current_time or "",
                current_position or "",
                self._pending_state_key(),
            )
            if state_key in self._dfs_state_seen:
                self.backtrack_count += 1
                self._dfs_log("memoized state repeat, backtrack...")
                return False, plan
            self._dfs_state_seen.add(state_key)

        if not self.all_satisfy_flag and not self.too_many_backtrack:
            ok, backtrack = self.check_requirement(plan)
            if ok:
                self.all_satisfy_flag = True
            if backtrack:
                self.backtrack_count += 1
                self._dfs_log("requirements can not be satisfied, backtrack...")
                return False, plan

        self.search_nodes += 1
        # 检查是否超时
        if self.stop_search:
            self.default_plan["backtrack_count"] = self.backtrack_count
            return self._best_effort_search_result(query, plan, poi_plan)
        if self._search_time_exceeded():
            self.stop_search = True
            self.default_plan["backtrack_count"] = self.backtrack_count
            return self._best_effort_search_result(query, plan, poi_plan)

        # 检查当前时间是否太晚，无法前往酒店或返程交通
        self._dfs_log("check if too late")
        if not self.too_many_backtrack:
            if self.check_if_too_late(query, current_day, current_time, current_position, poi_plan):
                self.backtrack_count += 1
                self._dfs_log("The current time is too late to go hotel or back-transport, backtrack...")
                return False, plan

        # 处理第一天的去程城际交通
        if current_day == 0 and current_time == "":
            plan = [{"day": current_day + 1, "activities": []}]  # 初始化第一天的活动列表
            # 添加去程城际交通活动
            plan[current_day]["activities"] = self.add_intercity_transport(
                plan[current_day]["activities"],
                poi_plan["go_transport"],
                innercity_transports=[],
                tickets=query["people_number"],
            )

            self._update_best_plan_from_partial(query, plan)

            self._dfs_log(plan)

            new_time = poi_plan["go_transport"]["EndTime"]  # 更新当前时间为去程交通的结束时间
            new_position = poi_plan["go_transport"]["To"]  # 更新当前位置为目的地（车站）

            # 递归调用 dfs_poi 进行后续规划
            success, plan = self.dfs_poi(query, poi_plan, plan, new_time, new_position, current_day)

            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan
            else:
                self.backtrack_count += 1
                self._dfs_log("No solution for the given Go Transport, backtrack...")
                return False, plan

        # breakfast
        if (current_time == "00:00" and current_day != query["days"] - 1) or (current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal("11:30", poi_plan["back_transport"]["BeginTime"])):
            if len(plan) < current_day + 1:
                plan.append({"day": current_day + 1, "activities": []})  # 如果是新的一天，添加新的活动列表

            self.search_nodes += 1
            # 选择并添加早餐活动
            plan = self.select_and_add_breakfast(plan, poi_plan, current_day, current_time, current_position, [])
            if not self._last_activity_matches(plan, current_day, "breakfast"):
                self.backtrack_count += 1
                self._dfs_log("Breakfast append failed, backtrack...")
                return False, plan

            new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为早餐结束时间
            new_position = current_position  # 位置不变

            # 递归调用 dfs_poi 进行后续规划
            success, plan = self.dfs_poi(
                query, poi_plan, plan, new_time, new_position, current_day
            )
            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan

            self._pop_last_activity_if_matches(plan, current_day, "breakfast")

            candidates_type = []
            if current_day == query["days"] - 1 and current_time != "":  # 如果是最后一天，考虑返程交通
                candidates_type.append("back-intercity-transport")
            else:
                self.backtrack_count += 1
                self._dfs_log("No solution for the given Breakfast, backtrack...")
                return False, plan
        elif current_time == "00:00" and current_day == query["days"] - 1 and time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"], "11:30"):
            candidates_type = ["back-intercity-transport"]
        else:  # 如果当前时间不是 "00:00"，说明一天已经开始
            haved_lunch_today, haved_dinner_today = False, False

            for act_i in plan[current_day]["activities"]:
                if act_i["type"] == "lunch":
                    haved_lunch_today = True  # 更新午餐状态
                if act_i["type"] == "dinner":
                    haved_dinner_today = True  # 更新晚餐状态
            if time_compare_if_earlier_equal("20:30", current_time):
                candidates_type = []
            else:
                candidates_type = ["attraction"]
            # 如果今天还没吃午餐，则考虑午餐
            if not haved_lunch_today:
                candidates_type.append("lunch")
            # 如果今天还没吃晚餐，则考虑晚餐
            if not haved_dinner_today:
                candidates_type.append("dinner")
            # 如果有住宿且不是最后一天，则考虑前往酒店
            if ("accommodation" in poi_plan) and (current_day < query["days"] - 1):
                candidates_type.append("hotel")
            # 如果是最后一天且时间不为空，则考虑返程交通
            if current_day == query["days"] - 1 and current_time != "":
                candidates_type.append("back-intercity-transport")

        self._dfs_log("candidates_type: ", candidates_type)

        # 当还有候选类型时
        while len(candidates_type) > 0:
            poi_type, candidates_type = self.select_next_poi_type(
                candidates_type,
                plan,
                poi_plan,
                current_day,
                current_time,
                current_position,
            )

            self._dfs_log(
                "POI planning, day {} {}, {}, next-poi type: {}".format(
                    current_day, current_time, current_position, poi_type
                )
            )

            if poi_type == "back-intercity-transport":
                success, plan = self._try_append_back_transport(
                    query, poi_plan, plan, current_day, current_time, current_position
                )
                if success:
                    return True, plan
            # 如果下一个 POI 类型是酒店
            elif poi_type == "hotel":
                # 获取选定的酒店信息
                hotel_sel = poi_plan["accommodation"]
                # 获取市内交通排名
                # transports_ranking = self.ranking_innercity_transport(current_position, hotel_sel["name"], current_day, current_time)
                transports_ranking = self.innercity_transports_ranking
                if self.transport_rules_by_distance is not None:
                    temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                    transports_ranking = self.get_transport_by_distance(temp_distance)
                # 遍历市内交通类型
                for trans_type_sel in transports_ranking:
                    self.search_nodes += 1
                    # 如果已经在酒店位置，则无需市内交通
                    if hotel_sel["name"] == current_position:
                        transports_sel = []
                        arrived_time = current_time
                    else:
                        # 收集市内交通选项，从当前位置到酒店
                        print("collecting innercity transport to hotel")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        # 没找到则回溯
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            print("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]

                        if not self._innercity_transports_valid(transports_sel):
                            self.backtrack_count += 1
                            continue

                        backtrack_flag = False
                        if self.transport_rules_by_distance is not None:
                            distance = 0
                            for transport in transports_sel:
                                if transport["mode"] is not None:
                                    distance += transport.get("distance", 0)
                            mode = None
                            if len(transports_sel) == 3:
                                mode = transports_sel[1]["mode"]
                            elif len(transports_sel) == 1:
                                mode = transports_sel[0]["mode"]
                            if mode is None:
                                continue
                            for rule in self.transport_rules_by_distance:
                                if rule["min_distance"] is not None:
                                    if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                                if rule["max_distance"] is not None:
                                    if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                        if backtrack_flag and not self.too_many_backtrack:
                            self.backtrack_count += 1
                            continue

                    arrived_time = clamp_time_to_day_end(arrived_time)
                    if self._arrived_time_too_late_for_hotel(arrived_time):
                        self.backtrack_count += 1
                        continue

                    if time_compare_if_earlier_equal("09:00", arrived_time) or (
                        self._innercity_budget_saver_enabled()
                        and current_day < query["days"] - 1
                        and not time_compare_if_earlier_equal(arrived_time, "08:00")
                    ):
                        before = len(plan[current_day]["activities"])
                        # 添加住宿活动
                        plan = self.add_accommodation(
                            current_plan=plan,
                            hotel_sel=hotel_sel,
                            current_day=current_day,
                            arrived_time=arrived_time,
                            required_rooms=self.required_rooms,
                            transports_sel=transports_sel,
                        )
                        if len(plan[current_day]["activities"]) == before:
                            self.backtrack_count += 1
                            continue
                        transports_sel = []

                    if time_compare_if_earlier_equal(arrived_time, "08:00"):
                        breakfast_start = (
                            "06:00"
                            if time_compare_if_earlier_equal(arrived_time, "06:00")
                            else arrived_time
                        )
                        plan = self.select_and_add_breakfast(
                            plan,
                            poi_plan,
                            current_day,
                            current_time,
                            current_position,
                            transports_sel,
                            start_time=breakfast_start,
                            end_time=add_time_delta(breakfast_start, 30),
                        )
                        if not self._last_activity_matches(plan, current_day, "breakfast"):
                            self.backtrack_count += 1
                            continue
                        new_time = plan[current_day]["activities"][-1]["end_time"]  # 更新当前时间为早餐结束时间
                        new_position = hotel_sel["name"]

                        if self._innercity_budget_saver_enabled() and current_day < query["days"] - 1:
                            before = len(plan[current_day]["activities"])
                            plan = self.add_accommodation(
                                current_plan=plan,
                                hotel_sel=hotel_sel,
                                current_day=current_day,
                                arrived_time=new_time,
                                required_rooms=self.required_rooms,
                                transports_sel=[],
                            )
                            if len(plan[current_day]["activities"]) == before:
                                self.backtrack_count += 1
                                continue
                            if not self.too_many_backtrack and self.check_budgets(plan):
                                self._pop_last_activity_if_matches(
                                    plan, current_day, "accommodation", hotel_sel["name"]
                                )
                            else:
                                success, plan = self.dfs_poi(
                                    query, poi_plan, plan, "00:00", new_position, current_day + 1
                                )
                                if success:
                                    return True, plan
                                if self._is_terminal_plan_failure(plan):
                                    return False, plan
                                self._pop_last_activity_if_matches(
                                    plan, current_day, "accommodation", hotel_sel["name"]
                                )

                        success, plan = self.dfs_poi(
                            query, poi_plan, plan, new_time, new_position, current_day
                        )
                        if success:
                            return True, plan
                        if self._is_terminal_plan_failure(plan):
                            return False, plan

                        self._pop_last_activity_if_matches(plan, current_day, "breakfast")

                    new_time = "00:00"  # 新的一天开始
                    new_position = hotel_sel["name"]  # 新位置为酒店

                    self._update_best_plan_from_partial(query, plan)

                    # 递归调用 dfs_poi 进行下一天的规划
                    success, plan = self.dfs_poi(query, poi_plan, plan, new_time, new_position, current_day + 1)

                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

                    self.backtrack_count += 1
                    self._dfs_log("Fail with the given accommodation activity, backtrack...")

                    self._pop_last_activity_if_matches(
                        plan, current_day, "accommodation", hotel_sel["name"]
                    )

                    return False, plan
            # 如果是午餐、晚餐或景点
            elif poi_type in ["lunch", "dinner", "attraction"]:
                if poi_type in ["lunch", "dinner"]:
                    success, plan = self._dfs_explore_restaurants(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan
                elif poi_type == "attraction":
                    success, plan = self._dfs_explore_attractions(
                        query,
                        poi_plan,
                        plan,
                        current_day,
                        current_time,
                        current_position,
                        poi_type,
                        candidates_type,
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan

                if current_day == query["days"] - 1:
                    success, plan = self._try_append_back_transport(
                        query, poi_plan, plan, current_day, current_time, current_position
                    )
                    if success:
                        return True, plan
                    if self._is_terminal_plan_failure(plan):
                        return False, plan
                # 如果不是最后一天且天数大于 1
                elif query["days"] > 1 and current_day < query["days"] - 1:
                    # go to hotel
                    hotel_sel = poi_plan["accommodation"]  # 获取选定的酒店信息
                    self.search_nodes += 1
                    # 获取市内交通排名
                    transports_ranking = self.innercity_transports_ranking
                    if self.transport_rules_by_distance is not None:
                        temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                        transports_ranking = self.get_transport_by_distance(temp_distance)
                    for trans_type_sel in transports_ranking:
                        self.search_nodes += 1
                        # 收集市内交通选项，从当前位置到酒店
                        self._dfs_log("not last day, but last event, collecting innercity transport to hotel")
                        transports_sel = self.collect_innercity_transport(
                            query["target_city"],
                            current_position,
                            hotel_sel["name"],
                            current_time,
                            trans_type_sel,
                        )
                        self._dfs_log(f"from: {current_position} to {hotel_sel['name']}")
                        if not isinstance(transports_sel, list):
                            self.backtrack_count += 1
                            self._dfs_log("inner-city transport error, backtrack...")
                            continue

                        if len(transports_sel) == 0:
                            arrived_time = current_time
                        else:
                            arrived_time = transports_sel[-1]["end_time"]

                        backtrack_flag = False
                        if self.transport_rules_by_distance is not None:
                            distance = 0
                            for transport in transports_sel:
                                if transport["mode"] is not None:
                                    distance += transport.get("distance", 0)
                            mode = None
                            if len(transports_sel) == 3:
                                mode = transports_sel[1]["mode"]
                            elif len(transports_sel) == 1:
                                mode = transports_sel[0]["mode"]
                            if mode is None:
                                continue
                            for rule in self.transport_rules_by_distance:
                                if rule["min_distance"] is not None:
                                    if distance > rule["min_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                                if rule["max_distance"] is not None:
                                    if distance < rule["max_distance"] and mode not in rule["transport_type"]:
                                        print("backtrack")
                                        backtrack_flag = True
                        if backtrack_flag and not self.too_many_backtrack:
                            self.backtrack_count += 1
                            continue

                        # 添加住宿活动
                        plan = self.add_accommodation(
                            current_plan=plan,
                            hotel_sel=hotel_sel,
                            current_day=current_day,
                            arrived_time=arrived_time,
                            required_rooms=self.required_rooms,
                            transports_sel=transports_sel,
                        )
                        if not self._last_activity_matches(
                            plan, current_day, "accommodation", hotel_sel["name"]
                        ):
                            self.backtrack_count += 1
                            continue

                        new_time = "00:00"  # 新的一天开始
                        new_position = hotel_sel["name"]  # 新位置为酒店名称

                        self._update_best_plan_from_partial(query, plan)

                        # 递归调用 dfs_poi 进行后续规划
                        success, plan = self.dfs_poi(
                            query,
                            poi_plan,
                            plan,
                            new_time,
                            new_position,
                            current_day + 1,
                        )

                        if success:
                            return True, plan
                        if self._is_terminal_plan_failure(plan):
                            return False, plan
                        else:
                            self.backtrack_count += 1
                            self._dfs_log("Try the go back hotel, failed, backtrack...")

                            self._pop_last_activity_if_matches(
                                plan, current_day, "accommodation", hotel_sel["name"]
                            )

                            continue
            else:
                # raise Exception("Not Implemented.")
                self._dfs_log("incorrect poi type: {}".format(poi_type))
                continue

            candidates_type.remove(poi_type)
            self._dfs_log(f"remove: {poi_type}, candidate type: {candidates_type}")
            self._dfs_log("try another poi type, backtrack...")

        return False, plan

    def _canon_type(self, kind, value):
        """Canonicalize a POI type/cuisine the same way the verifier does
        (concept alias + lowercase), so the planner matches DB values like
        "hot pot"/"university campus" against constraint labels "Hot pot"/"University campus"."""
        from chinatravel.symbol_verification.concept_func import normalize_concept_value
        return str(normalize_concept_value(kind, str(value))).lower()

    def _type_match_mask(self, series, target, kind):
        """Boolean mask of rows whose canonicalized type/cuisine equals target."""
        t = self._canon_type(kind, target)
        return series.astype(str).map(lambda v: self._canon_type(kind, v) == t)

    def check_constraint(self, plan, constraints):
        # 初始化访问记录
        visited_attractions = set()
        visited_attraction_types = set()
        visited_restaurants = set()
        visited_restaurant_types = set()

        logic_fail = False
        backtrack = False

        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0
        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner"]:
                    overall_cost += activity["cost"]
                    restaurant_cost += activity["cost"]
                elif activity["type"] == "attraction":
                    overall_cost += activity["cost"]
                    attraction_cost += activity["cost"]
                innercity_cost += sum(transport.get("cost", 0) for transport in activity.get("transports", []))
        self.overall_cost = overall_cost + innercity_cost + self.hotel_cost + self.intercity_cost
        # print(f"overall cost: {self.overall_cost}, attraction cost:{attraction_cost}, restaurant cost: {restaurant_cost}, innercity cost: {innercity_cost}")

        if self.attraction_budget is not None and self.attraction_budget < attraction_cost:
            self.backtrack_count += 1
            print("attraction budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.restaurant_budget is not None and self.restaurant_budget < restaurant_cost:
            self.backtrack_count += 1
            print("restaurant budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.innercity_budget is not None and self.innercity_budget < innercity_cost:
            self.backtrack_count += 1
            print("innercity budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False
        if self.overall_budget is not None and self.overall_budget < self.overall_cost:
            self.backtrack_count += 1
            print("overall budget exceeded, backtrack...")
            logic_fail = True
            backtrack = True
            self.all_satisfy_flag = False

        # 遍历活动，统计信息并处理 must_not 类约束 & only_free_attractions
        for day in plan:
            for act in day["activities"]:
                poi_name = act.get("position")
                poi_info = None

                if act.get("type") == "attraction":
                    match = self.memory["attractions"][self.memory["attractions"]["name"] == poi_name]
                    if not match.empty:
                        poi_info = match.iloc[0].to_dict()
                        visited_attractions.add(poi_info["name"])
                        visited_attraction_types.add(poi_info["type"])

                        # must_not_see_attraction
                        if "must_not_see_attraction" in constraints:
                            if poi_info["name"] in constraints["must_not_see_attraction"]:
                                print("visited must_not_see_attraction")
                                backtrack = True

                        # must_not_see_attraction_type (canonicalized like the verifier)
                        if "must_not_see_attraction_type" in constraints:
                            from chinatravel.symbol_verification.concept_func import normalize_concept_value
                            _excl = {normalize_concept_value("attraction", str(t)).lower() for t in constraints["must_not_see_attraction_type"]}
                            if str(normalize_concept_value("attraction", poi_info["type"])).lower() in _excl:
                                print("visited must_not_see_attraction_type")
                                backtrack = True

                        # only_free_attractions
                        if "only_free_attractions" in constraints and poi_info.get("price", 0) > 0:
                            print("only_free_attractions but not free")
                            backtrack = True

                elif act.get("type") in {"lunch", "dinner"}:
                    match = self.memory["restaurants"][self.memory["restaurants"]["name"] == poi_name]
                    if not match.empty:
                        poi_info = match.iloc[0].to_dict()
                        visited_restaurants.add(poi_info["name"])
                        visited_restaurant_types.add(poi_info["cuisine"])

                        # must_not_visit_restaurant
                        if "must_not_visit_restaurant" in constraints:
                            if poi_info["name"] in constraints["must_not_visit_restaurant"]:
                                print("visited must_not_visit_restaurant")
                                backtrack = True

                        # must_not_visit_restaurant_type (canonicalized like the verifier)
                        if "must_not_visit_restaurant_type" in constraints:
                            from chinatravel.symbol_verification.concept_func import normalize_concept_value
                            _excl_cui = {normalize_concept_value("restaurant", str(t)).lower() for t in constraints["must_not_visit_restaurant_type"]}
                            if str(normalize_concept_value("restaurant", poi_info["cuisine"])).lower() in _excl_cui:
                                print("visited must_not_visit_restaurant_type")
                                backtrack = True

        # 遍历 must 类要求，统计是否满足
        # 注意：这些不满足不立即回溯，而是给机会后续补上
        if "must_see_attraction" in constraints:
            required = set(constraints["must_see_attraction"])
            if not required.issubset(visited_attractions):
                logic_fail = True

        if "must_see_attraction_type" in constraints:
            # Canonicalize both sides (alias + case) like the verifier: DB types are
            # inconsistently cased across cities (e.g. "university campus").
            required = {self._canon_type("attraction", t) for t in constraints["must_see_attraction_type"]}
            visited_ct = {self._canon_type("attraction", t) for t in visited_attraction_types}
            if getattr(self, "must_see_attraction_type_match_any", False):
                if not (required & visited_ct):
                    logic_fail = True
            elif not required.issubset(visited_ct):
                logic_fail = True

        if "must_visit_restaurant" in constraints:
            required = set(constraints["must_visit_restaurant"])
            if not required.issubset(visited_restaurants):
                logic_fail = True

        if "must_visit_restaurant_type" in constraints:
            # Canonicalize both sides: DB cuisine e.g. "hot pot" -> "Hot pot".
            required = {self._canon_type("restaurant", t) for t in constraints["must_visit_restaurant_type"]}
            visited_ct = {self._canon_type("restaurant", t) for t in visited_restaurant_types}
            if getattr(self, "must_visit_restaurant_type_match_any", False):
                if not (required & visited_ct):
                    logic_fail = True
            elif not required.issubset(visited_ct):
                logic_fail = True

        if "must_innercity_transport" in constraints:
            if not set(constraints["must_innercity_transport"]).issubset(set(self.innercity_transports_ranking)):
                logic_fail = True

        if "must_not_innercity_transport" in constraints:
            if not set(constraints["must_not_innercity_transport"]) & set(self.innercity_transports_ranking):
                logic_fail = True

        return (not backtrack) and (not logic_fail), backtrack

    def check_requirement(self, plan):
        if self.all_satisfy:
            # 情况1：必须全部满足
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints)
                if backtrack:
                    return False, True  # 立即回溯
                if not ok:
                    return False, False  # 不满足但不强制回溯
            return True, False
        else:
            # 情况2：满足任一要求即可
            for constraints in self.requirement_list:
                ok, backtrack = self.check_constraint(plan, constraints)
                if backtrack:
                    return False, True
                if ok:
                    return True, False  # 有一组满足即可
            return False, False  # 所有组都不满足

    def check_budgets(self, plan):
        overall_cost = 0
        attraction_cost = 0
        restaurant_cost = 0
        innercity_cost = 0
        for day_activities in plan:
            for activity in day_activities["activities"]:
                if activity["type"] in ["breakfast", "lunch", "dinner"]:
                    overall_cost += activity["cost"]
                    restaurant_cost += activity["cost"]
                elif activity["type"] == "attraction":
                    overall_cost += activity["cost"]
                    attraction_cost += activity["cost"]
                innercity_cost += sum(transport.get("cost", 0) for transport in activity.get("transports", []))
        self.overall_cost = overall_cost + innercity_cost + self.hotel_cost + self.intercity_cost

        if self.attraction_budget is not None and self.attraction_budget < attraction_cost:
            self.backtrack_count += 1
            print("attraction budget exceeded, backtrack...")
            return True
        if self.restaurant_budget is not None and self.restaurant_budget < restaurant_cost:
            self.backtrack_count += 1
            print("restaurant budget exceeded, backtrack...")
            return True
        if self.innercity_budget is not None and self.innercity_budget < innercity_cost:
            self.backtrack_count += 1
            print("innercity budget exceeded, backtrack...")
            return True
        if self.overall_budget is not None and self.overall_budget < self.overall_cost:
            self.backtrack_count += 1
            print("overall budget exceeded, backtrack...")
            return True

        return False

    def calculate_distance(self, query, start, end):
        """
        计算两个 POI 之间的球面距离（公里），带缓存。
        """
        city = query["target_city"]

        if start == end:
            return 0.0

        cache_key = (city, frozenset((start, end)))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]

        coordinate_A = self.poi_search.search(city, start)
        coordinate_B = self.poi_search.search(city, end)

        if not coordinate_A or not coordinate_B:
            self._distance_cache[cache_key] = None
            return None

        locationA, locationB = coordinate_A, coordinate_B

        distance = geodesic(locationA, locationB).kilometers
        self._distance_cache[cache_key] = distance
        return distance

    def get_transport_by_distance(self, distance):
        selected_modes = []

        if distance is None:
            return ["metro", "taxi", "walk"]

        for rule in self.transport_rules_by_distance:
            min_d = rule.get("min_distance", 0)
            max_d = rule.get("max_distance", float("inf"))
            transport = rule.get("transport_type")

            if min_d is None and max_d is not None:
                if distance <= max_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])
            if max_d is None and (min_d is not None and min_d != 0):
                if distance >= min_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])
            if min_d is not None and max_d is not None:
                if distance >= min_d and distance <= max_d:
                    selected_modes.extend(transport if isinstance(transport, list) else [transport])

        if not selected_modes:
            selected_modes = ["metro", "taxi", "walk"]

        return selected_modes

    def _innercity_budget_saver_enabled(self):
        if self.innercity_budget is None or self.must_innercity_transport is not None:
            return False
        days = max(1, int(self.query.get("days", 1))) if hasattr(self, "query") else 1
        people = max(1, int(self.query.get("people_number", 1))) if hasattr(self, "query") else 1
        per_day_budget = float(self.innercity_budget) / days
        # Inner-city cost scales with party size (per-ticket * people, taxi cars),
        # so a budget that looks loose in absolute terms can be tight per person.
        # Additive: keep the original trigger and also enable the saver when the
        # per-person-per-day budget is tight (catches large-party tight-budget trips
        # that otherwise waste budget on taxis and exhaust it before completing).
        per_person_per_day = per_day_budget / people
        return per_day_budget <= max(25.0, people * 18.0) or per_person_per_day <= 60.0

    def _budget_pressure(self, budget, spent):
        if budget is None:
            return 0.0
        try:
            budget_value = float(budget)
        except (TypeError, ValueError):
            return 0.0
        if budget_value <= 0:
            return 1.0
        ratio = max(0.0, min(1.5, float(spent) / budget_value))
        return max(0.0, min(1.0, ratio))

    def _normalize_lower_better(self, values):
        if not values:
            return []
        clean = []
        for value in values:
            try:
                clean.append(float(value))
            except (TypeError, ValueError):
                clean.append(10**6)
        finite = [value for value in clean if value < 10**6]
        if not finite:
            return [1.0 for _ in clean]
        low, high = min(finite), max(finite)
        if high <= low:
            return [0.0 if value < 10**6 else 1.0 for value in clean]
        return [
            1.0 if value >= 10**6 else max(0.0, min(1.0, (value - low) / (high - low)))
            for value in clean
        ]

    def _current_spend_snapshot(self, plan=None):
        plan = plan if plan is not None else getattr(self, "_current_dfs_plan", None)
        spend = {
            "overall": 0.0,
            "attraction": 0.0,
            "restaurant": 0.0,
            "innercity": 0.0,
            "hotel": float(getattr(self, "hotel_cost", 0) or 0),
            "intercity": float(getattr(self, "intercity_cost", 0) or 0),
        }
        if plan:
            for day_activities in plan:
                for activity in day_activities.get("activities", []):
                    if activity.get("type") in ["breakfast", "lunch", "dinner"]:
                        cost = float(activity.get("cost", 0) or 0)
                        spend["restaurant"] += cost
                        spend["overall"] += cost
                    elif activity.get("type") == "attraction":
                        cost = float(activity.get("cost", 0) or 0)
                        spend["attraction"] += cost
                        spend["overall"] += cost
                    for transport in activity.get("transports", []) or []:
                        spend["innercity"] += float(transport.get("cost", 0) or 0)
        # overall_budget is parsed only from the DSL ``total_cost`` clause, which
        # across the whole dataset is defined as activity_cost + inner-city
        # transport cost (never hotel or intercity). Match that basis so the
        # ``overall`` budget pressure that drives POI/route ranking is accurate;
        # adding hotel+intercity here overstated spend and over-penalised.
        spend["overall"] += spend["innercity"]
        return spend

    def _pending_required_items(self):
        pending = {
            "attraction_names": [],
            "attraction_types": [],
            "restaurant_names": [],
            "restaurant_types": [],
            "order_predecessors": [],
            "order_blocked": [],
        }
        for name in self.must_see_attraction or []:
            if not self._visited_contains(self.attraction_names_visiting, name):
                pending["attraction_names"].append(name)
        for spot_type in self.must_see_attraction_type or []:
            if not self._visited_contains(self.spot_type_visiting, spot_type):
                pending["attraction_types"].append(spot_type)
        for name in self.must_visit_restaurant or []:
            if not self._visited_contains(self.restaurant_names_visiting, name):
                pending["restaurant_names"].append(name)
        for cuisine in self.must_visit_restaurant_type or []:
            if not self._visited_contains(self.food_type_visiting, cuisine):
                pending["restaurant_types"].append(cuisine)
        visited_names = set(self.attraction_names_visiting) | set(self.restaurant_names_visiting)
        for before, after in self.must_visit_order or []:
            if before not in visited_names:
                pending["order_predecessors"].append(before)
                pending["order_blocked"].append(after)
        return pending

    def _pending_state_key(self):
        """Hashable frozenset of still-pending required items (for 6b memoization)."""
        pending = self._pending_required_items()
        items = []
        for name in pending.get("attraction_names", []):
            items.append(("a", name))
        for t in pending.get("attraction_types", []):
            items.append(("at", t))
        for name in pending.get("restaurant_names", []):
            items.append(("r", name))
        for t in pending.get("restaurant_types", []):
            items.append(("rt", t))
        for name in pending.get("order_predecessors", []):
            items.append(("ord", name))
        return frozenset(items)

    def _pending_pressure(self, pending):
        total = (
            len(self.must_see_attraction or [])
            + len(self.must_see_attraction_type or [])
            + len(self.must_visit_restaurant or [])
            + len(self.must_visit_restaurant_type or [])
            + 2 * len(self.must_visit_order or [])
        )
        remaining = sum(len(items) for items in pending.values())
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, remaining / total))

    def _time_pressure(self, query, current_day, current_time, poi_plan=None):
        days = max(1, int(query.get("days", 1)))
        try:
            current_minutes = time_to_minutes(current_time) if current_time else 0
        except (TypeError, ValueError):
            current_minutes = 0
        day_pressure = max(0.0, min(1.0, (current_day + current_minutes / (24 * 60)) / days))
        clock_pressure = max(0.0, min(1.0, (current_minutes - 12 * 60) / (10 * 60)))
        back_pressure = 0.0
        if poi_plan and current_day == days - 1 and poi_plan.get("back_transport") is not None:
            try:
                back_minutes = time_to_minutes(poi_plan["back_transport"].get("BeginTime"))
                remaining = max(0, back_minutes - current_minutes)
                back_pressure = max(0.0, min(1.0, 1.0 - remaining / 360.0))
            except (TypeError, ValueError, AttributeError):
                back_pressure = 0.0
        return max(day_pressure, clock_pressure, back_pressure)

    def _default_dynamic_weights(self, context):
        budget = context.get("budget_pressure", {})
        pending_pressure = float(context.get("pending_pressure", 0.0) or 0.0)
        time_pressure = float(context.get("time_pressure", 0.0) or 0.0)
        route_pressure = max(
            budget.get("innercity", 0.0),
            budget.get("intercity", 0.0),
            budget.get("overall", 0.0),
        )
        poi_price_pressure = max(
            budget.get("overall", 0.0),
            budget.get("intercity", 0.0),
            budget.get("attraction", 0.0),
            budget.get("restaurant", 0.0),
        )
        pending_late = pending_pressure * time_pressure
        price_weight = 0.20 + 1.10 * poi_price_pressure
        if pending_late > 0.25:
            price_weight *= max(0.35, 1.0 - 0.75 * pending_late)
        weights = {
            "must_coverage": 0.40 + 2.60 * pending_pressure * (0.45 + time_pressure),
            "route_cost": 0.35 + 1.40 * route_pressure,
            "route_distance": 0.12 + 0.18 * (1.0 - route_pressure),
            "base_score": 0.10,
            "poi_price": price_weight,
            "time_margin": 0.20 + 1.30 * time_pressure + 0.90 * pending_late,
            "order_block": 1.20,
            "semantic": self.rank_semantic_weight,
        }
        if sum(len(items) for items in (context.get("pending") or {}).values()) > 0:
            weights["must_coverage"] = max(weights["must_coverage"], 1.10)
        return weights

    def _llm_refine_dynamic_weights(self, context, default_weights):
        if not self.use_llm_dynamic_weights:
            return default_weights
        llm_name = getattr(getattr(self, "backbone_llm", None), "name", "")
        if not self.backbone_llm or llm_name == "EmptyLLM":
            return default_weights
        cache_key = json.dumps(
            {
                "kind": context.get("kind"),
                "day": context.get("current_day"),
                "time": context.get("current_time"),
                "budget": context.get("budget_pressure"),
                "pending": context.get("pending"),
                "time_pressure": round(float(context.get("time_pressure", 0.0)), 2),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._dynamic_weight_cache:
            return self._dynamic_weight_cache[cache_key]
        prompt_payload = {
            "task": "Return deterministic ranking weights for DSL-to-plan search. Do not choose POIs.",
            "constraints": self._segment_constraints(include_dynamic=False),
            "context": context,
            "default_weights": default_weights,
            "allowed_keys": [
                "must_coverage",
                "route_cost",
                "route_distance",
                "base_score",
                "poi_price",
                "time_margin",
                "order_block",
                "semantic",
            ],
            "rules": [
                "Increase route_cost and poi_price when budgets are tight.",
                "Increase must_coverage and time_margin when required POIs remain and the day/trip is late.",
                "Decrease poi_price priority when pending must/time are urgent.",
                "Return only JSON: {\"weights\": {...}} with numeric values from 0 to 3.",
            ],
        }
        messages = [
            {"role": "system", "content": "You tune deterministic search weights for a travel planner."},
            {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
        ]
        try:
            start = time.time()
            response = self.backbone_llm(messages, one_line=False, json_mode=True)
            self.llm_inference_time_count += time.time() - start
            parsed = json.loads(response)
            raw_weights = parsed.get("weights", parsed)
            refined = dict(default_weights)
            for key in default_weights:
                if key in raw_weights:
                    refined[key] = max(0.0, min(3.0, float(raw_weights[key])))
            self._dynamic_weight_cache[cache_key] = refined
            self.llm_rec_count += 1
            return refined
        except Exception:
            self.llm_rec_format_error += 1
            self._dynamic_weight_cache[cache_key] = default_weights
            return default_weights

    def _build_dynamic_ranking_context(self, query, current_time, poi_kind):
        plan = getattr(self, "_current_dfs_plan", None)
        poi_plan = getattr(self, "_current_poi_plan", None)
        current_day = getattr(self, "_current_rank_day", 0)
        spend = self._current_spend_snapshot(plan)
        pending = self._pending_required_items()
        budget_pressure = {
            "overall": self._budget_pressure(self.overall_budget, spend["overall"]),
            "innercity": self._budget_pressure(self.innercity_budget, spend["innercity"]),
            "attraction": self._budget_pressure(self.attraction_budget, spend["attraction"]),
            "restaurant": self._budget_pressure(self.restaurant_budget, spend["restaurant"]),
            "hotel": self._budget_pressure(self.hotel_budget, spend["hotel"]),
            "intercity": self._budget_pressure(self.intercity_budget, spend["intercity"]),
        }
        context = {
            "kind": poi_kind,
            "current_day": current_day,
            "current_time": current_time,
            "spend": spend,
            "budget_pressure": budget_pressure,
            "pending": pending,
            "pending_pressure": self._pending_pressure(pending),
            "time_pressure": self._time_pressure(query, current_day, current_time, poi_plan),
        }
        weights = self._default_dynamic_weights(context)
        context["weights"] = self._llm_refine_dynamic_weights(context, weights)
        if self.dynamic_weight_debug:
            self._dfs_log("[dynamic-rank]", context)
        return context

    def _build_hotel_ranking_weights(self, query):
        spend = self._current_spend_snapshot(getattr(self, "_current_dfs_plan", None))
        pending = self._pending_required_items()
        budget_pressure = {
            "overall": self._budget_pressure(self.overall_budget, spend["overall"]),
            "innercity": self._budget_pressure(self.innercity_budget, spend["innercity"]),
            "intercity": self._budget_pressure(self.intercity_budget, spend["intercity"]),
            "hotel": self._budget_pressure(self.hotel_budget, spend["hotel"]),
        }
        context = {
            "kind": "hotel",
            "current_day": getattr(self, "_current_rank_day", 0),
            "current_time": getattr(self, "_current_rank_time", ""),
            "spend": spend,
            "budget_pressure": budget_pressure,
            "pending": pending,
            "pending_pressure": self._pending_pressure(pending),
            "time_pressure": self._time_pressure(query, 0, "", getattr(self, "_current_poi_plan", None)),
        }
        route_pressure = max(
            budget_pressure["innercity"],
            budget_pressure["intercity"],
            budget_pressure["overall"],
        )
        pending_late = context["pending_pressure"] * context["time_pressure"]
        price_weight = 0.35 + 1.25 * max(
            budget_pressure["hotel"],
            budget_pressure["intercity"],
            budget_pressure["overall"],
        )
        if pending_late > 0.25:
            price_weight *= max(0.45, 1.0 - 0.55 * pending_late)
        weights = {
            "route_cost": 0.45 + 1.25 * route_pressure,
            "route_distance": 0.20 + 0.35 * (1.0 - route_pressure),
            "poi_price": price_weight,
        }
        context["weights"] = self._llm_refine_dynamic_weights(context, weights)
        return context

    def _hotel_matches_required_feature(self, hotel_row):
        if not self.must_live_hotel_feature:
            return True
        normalized_features = self._normalized_hotel_features(hotel_row)
        for required in self.must_live_hotel_feature:
            req = self._normalize_hotel_feature(required)
            if req not in normalized_features:
                return False
        return True

    @staticmethod
    def _normalize_hotel_feature(value):
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def _normalized_hotel_features(self, hotel_row):
        # Apply the same concept alias the official verifier uses (accommodation_type
        # -> normalize_concept_value), so a DB feature like "Bed and breakfast"
        # matches a "homestay" constraint. Without this the planner compares the raw
        # DB value against the (alias-normalized) constraint and finds no hotel,
        # failing the whole search.
        try:
            from chinatravel.symbol_verification.concept_func import normalize_concept_value
        except Exception:
            normalize_concept_value = None
        features = []
        for item in str(hotel_row.get("featurehoteltype", "")).split(","):
            item = item.strip()
            if not item:
                continue
            if normalize_concept_value is not None:
                item = normalize_concept_value("accommodation", item)
            features.append(self._normalize_hotel_feature(item))
        return features

    def _hotel_matches_forbidden_feature(self, hotel_row):
        if not self.must_not_live_hotel_feature:
            return False
        hotel_features = self._normalized_hotel_features(hotel_row)
        for forbidden in self.must_not_live_hotel_feature:
            forbidden = self._normalize_hotel_feature(forbidden)
            if forbidden in hotel_features:
                return True
        return False

    def _hotel_satisfies_hard_constraints(self, hotel_row):
        name = hotel_row.get("name")
        if self.must_live_hotel and name not in set(self.must_live_hotel):
            return False
        if self.must_not_live_hotel and name in set(self.must_not_live_hotel):
            return False
        # Required bed number (``room_type(activity)!=N`` in the DSL) is a hard
        # constraint on the hotel's ``numbed``. The legacy triple-loop path
        # backtracked on a mismatch; the bundle path must reject the hotel here
        # too, otherwise it happily books a double-bed room for a single-bed
        # requirement and fails the hard verifier.
        bed_number = getattr(self, "bed_number", None)
        if bed_number is not None:
            try:
                if int(hotel_row["numbed"]) != int(bed_number):
                    return False
            except (TypeError, ValueError, KeyError):
                return False
        return (
            self._hotel_matches_required_feature(hotel_row)
            and not self._hotel_matches_forbidden_feature(hotel_row)
        )

    def _has_hard_hotel_constraints(self):
        return any(
            (
                self.must_live_hotel,
                self.must_not_live_hotel,
                self.must_live_hotel_feature,
                self.must_not_live_hotel_feature,
                self.must_live_hotel_location_limit,
            )
        )

    def _hotel_hard_bonus(self, hotel_row):
        bonus = 0.0
        name = hotel_row.get("name")
        if self.must_live_hotel and name in set(self.must_live_hotel):
            bonus += 1000.0
        if self._hotel_matches_required_feature(hotel_row):
            if self.must_live_hotel_feature:
                bonus += 500.0
        return bonus

    def _has_pending_required_poi(self):
        if self.must_see_attraction:
            for name in self.must_see_attraction:
                if not self._visited_contains(self.attraction_names_visiting, name):
                    return True
        if self.must_see_attraction_type:
            for spot_type in self.must_see_attraction_type:
                if not self._visited_contains(self.spot_type_visiting, spot_type):
                    return True
        if self.must_visit_restaurant:
            for name in self.must_visit_restaurant:
                if not self._visited_contains(self.restaurant_names_visiting, name):
                    return True
        if self.must_visit_restaurant_type:
            for cuisine in self.must_visit_restaurant_type:
                if not self._visited_contains(self.food_type_visiting, cuisine):
                    return True
        if self.must_visit_order:
            visited_names = set(self.attraction_names_visiting) | set(self.restaurant_names_visiting)
            for before, after in self.must_visit_order:
                if before not in visited_names or after not in visited_names:
                    return True
        return False

    def _segment_query(self, query):
        segment_query = dict(query)
        segment_query.update(self._segment_constraints())
        return segment_query

    def _segment_constraints(self, include_dynamic=True):
        constraints = {
            "must_see_attraction": self.must_see_attraction,
            "must_see_attraction_type": self.must_see_attraction_type,
            "must_visit_restaurant": self.must_visit_restaurant,
            "must_visit_restaurant_type": self.must_visit_restaurant_type,
            "must_depart_transport": self.must_depart_transport,
            "must_return_transport": self.must_return_transport,
            "must_not_depart_transport": self.must_not_depart_transport,
            "must_not_return_transport": self.must_not_return_transport,
            "innercity_budget": self.innercity_budget,
            "must_visit_order": self.must_visit_order,
            "must_live_hotel": self.must_live_hotel,
            "must_not_live_hotel": self.must_not_live_hotel,
            "must_live_hotel_feature": self.must_live_hotel_feature,
            "must_not_live_hotel_feature": self.must_not_live_hotel_feature,
        }
        if include_dynamic and self._dynamic_ranking_context is not None:
            constraints["dynamic_ranking"] = self._dynamic_ranking_context
        return constraints

    def _geo_route_lb(self, city, origin, targets):
        """Sum of min one-way route cost origin->each target.

        Uses the segment index when an edge exists; otherwise falls back to a
        geodesic-distance estimate (km * taxi_per_km_est) so a far-away must POI
        still incurs a geographic penalty even when the segment graph misses it.
        """
        if not origin or not targets:
            return 0.0
        total = 0.0
        for target in targets:
            if not target or target == origin:
                continue
            seg = (
                self.segment_index._best_intracity_segment(city, origin, target)
                if self.segment_index is not None
                else None
            )
            if seg is not None:
                total += float(seg.get("cost", 0) or 0)
            else:
                dist = self.calculate_distance({"target_city": city}, origin, target)
                total += (dist if dist is not None else 30.0) * self.taxi_per_km_est
        return total

    def _must_poi_anchor_names(self):
        """Named must-visit POIs used as geographic anchors for hotel ranking."""
        names = []
        for name in (self.must_see_attraction or []):
            if name not in names:
                names.append(name)
        for name in (self.must_visit_restaurant or []):
            if name not in names:
                names.append(name)
        return names

    # ------------------------------------------------------------------
    # Deterministic bundle scorer (Section 3 of the geo bundle search plan)
    # ------------------------------------------------------------------
    BUNDLE_LATE_ARRIVAL_MIN = 14 * 60  # arrivals after 14:00 compress a 2-day trip

    def _effective_top_k(self):
        """Narrow candidate width when required POIs are still pending (6c)."""
        if self.enable_dynamic_top_k and self._has_pending_required_poi():
            return self.dynamic_top_k_pending
        return self.top_k_candidates

    def _bundle_weights(self, comps=None, days=None):
        weights = {
            "intercity": float(getattr(self, "bundle_w_intercity", 0.5)),
            "hotel": float(getattr(self, "bundle_w_hotel", 0.4)),
            "route": float(getattr(self, "bundle_w_route", 1.0)),
            "time": float(getattr(self, "bundle_w_time", 0.6)),
            "bonus": float(getattr(self, "bundle_w_bonus", 1.2)),
        }
        if not comps:
            return weights
        # Budget tightness at the *opening* move. The mid-DFS ``_budget_pressure``
        # is "spent-so-far / budget", which is ~0 here (nothing spent yet), so it
        # never lifts weights when the hotel is being picked. Instead measure how
        # much of each budget the cheapest feasible option already consumes
        # (floor / budget): a tight budget must pull the scorer toward the cheap /
        # central choices rather than the geo-only optimum.
        def _tight(floor, budget):
            try:
                budget = float(budget)
            except (TypeError, ValueError):
                return 0.0
            if budget <= 0:
                return 1.0
            return max(0.0, min(1.0, float(floor) / budget))

        ic_floor = min(c["intercity"] for c in comps)
        hotel_floor = min(c["hotel_cost"] for c in comps)
        route_floor = min(c["route_lb"] for c in comps)
        rest_floor = self._must_restaurant_price_floor()
        night_factor = max(1, (int(days) - 1)) if days else 1
        ic_t = _tight(ic_floor, self.intercity_budget)
        hotel_t = _tight(hotel_floor, self.hotel_budget)
        # ``route_lb`` is a per-hotel intra-city cost proxy; scale by nights as a
        # coarse daily-transport driver for the intra-city budget.
        inner_t = _tight(route_floor * night_factor, self.innercity_budget)
        overall_t = _tight(ic_floor + hotel_floor + rest_floor, self.overall_budget)
        gain = float(getattr(self, "bundle_budget_weight_gain", 1.5))
        weights["intercity"] *= 1.0 + gain * max(ic_t, overall_t)
        weights["hotel"] *= 1.0 + gain * max(hotel_t, overall_t)
        weights["route"] *= 1.0 + gain * max(inner_t, overall_t)
        return weights

    def _bundle_location_anchors(self, query):
        """Geographic anchors for scoring hotel location in a bundle.

        Must-visit POIs are the ideal anchors. When a query has none, the
        hotel's intra-city transport cost is still driven by wherever the plan
        actually goes, so proxy that cluster with a small sample of candidate
        attractions/restaurants. Without this, ``route_lb`` collapses to 0 for
        every hotel and location becomes invisible to the scorer -- which lets a
        cheap far-flung hotel win and blow a tight intra-city transport budget.
        """
        cached = getattr(self, "_bundle_loc_anchor_cache", None)
        if cached is not None:
            return cached
        names = list(self._must_poi_anchor_names())
        if not names:
            for key, k in (("attractions", 4), ("restaurants", 3)):
                df = self.memory.get(key)
                if df is None or getattr(df, "empty", True):
                    continue
                sub = df.sort_values("price") if "price" in df.columns else df
                names.extend(str(n) for n in sub["name"].head(k).tolist())
        self._bundle_loc_anchor_cache = names
        return names

    def _must_restaurant_price_floor(self):
        """Lower bound on required-restaurant spend: sum of each must's min price."""
        res = self.memory.get("restaurants")
        if res is None or res.empty or not self.must_visit_restaurant:
            return 0.0
        floor = 0.0
        for name in self.must_visit_restaurant:
            rows = res[res["name"] == name]
            if not rows.empty:
                try:
                    floor += float(rows["price"].min() or 0)
                except (TypeError, ValueError):
                    pass
        return floor

    def _bundle_hotel_rooms(self, query, hotel):
        people = query["people_number"]
        room_type = hotel["numbed"]
        rooms = int((people - 1) / room_type) + 1
        if self.room_number is not None:
            rooms = self.room_number
        return rooms

    def _bundle_feasible(self, query, go, back, hotel, relax_poi=False):
        """Hard pruning only. Returns a cost dict (with a soft budget flag) or
        None when the bundle is definitively infeasible.

        The overall-budget lower bound is admissible but is exposed as a soft
        ``budget_ok`` flag rather than eliminating the bundle here: if it would
        remove every candidate, the caller still needs bundles to run DFS and
        produce a best-effort plan (matching the legacy path behaviour).

        ``relax_poi`` drops the required-POI reachability lower bounds. They are
        admissible but occasionally over-tight; when they would prune *every*
        bundle the caller retries with them relaxed so DFS still gets candidates
        rather than returning an empty plan.
        """
        people = query["people_number"]
        days = query["days"]
        try:
            intercity = (float(go["Cost"]) + float(back["Cost"])) * people
        except (TypeError, ValueError):
            return None
        if self.intercity_budget is not None and intercity > self.intercity_budget:
            return None
        hotel_cost = 0.0
        if hotel is not None:
            if not self._hotel_satisfies_hard_constraints(hotel):
                return None
            rooms = self._bundle_hotel_rooms(query, hotel)
            hotel_cost = float(hotel["price"]) * rooms * (days - 1)
            if self.hotel_budget is not None and hotel_cost > self.hotel_budget:
                return None
        elif time_compare_if_earlier_equal(back["BeginTime"], go["EndTime"]):
            # one-day trip: return leg must depart after arrival
            return None
        if relax_poi:
            rest_floor = self._must_restaurant_price_floor()
            budget_ok = (
                self.overall_budget is None
                or intercity + hotel_cost + rest_floor <= self.overall_budget
            )
            return {"intercity": intercity, "hotel_cost": hotel_cost, "budget_ok": budget_ok}
        if not self._can_satisfy_required_poi_after_go_arrival(query, go):
            return None
        if not self._can_satisfy_required_poi_between_intercity(query, go, back):
            return None
        # Metro-only + late arrival is often infeasible for short trips (Section 6d).
        if self.enable_metro_only_prune and self._bundle_metro_only_infeasible(query, go):
            return None
        rest_floor = self._must_restaurant_price_floor()
        budget_ok = (
            self.overall_budget is None
            or intercity + hotel_cost + rest_floor <= self.overall_budget
        )
        return {"intercity": intercity, "hotel_cost": hotel_cost, "budget_ok": budget_ok}

    def _bundle_metro_only_infeasible(self, query, go):
        """Heuristic prune: metro-only city travel with a late arrival on a
        short trip leaves too little feasible daytime to satisfy required POIs.
        """
        metro_only = (
            self.must_innercity_transport is not None
            and set(self.must_innercity_transport) == {"metro"}
        ) or (
            self.must_not_innercity_transport is not None
            and {"walk", "taxi"}.issubset(set(self.must_not_innercity_transport))
        )
        if not metro_only:
            return False
        if query.get("days", 1) > 2:
            return False
        try:
            arr = time_to_minutes(str(go["EndTime"]))
        except (TypeError, ValueError):
            return False
        return arr >= 15 * 60 and self._required_poi_constraints_present()

    def _bundle_score_components(self, query, go, back, hotel, feas):
        city = query["target_city"]
        anchors = self._bundle_location_anchors(query)
        origin = hotel["name"] if hotel is not None else go.get("To")
        route_lb = self._geo_route_lb(city, origin, anchors)
        try:
            arr = time_to_minutes(str(go["EndTime"]))
            dep = time_to_minutes(str(back["BeginTime"]))
            begin = time_to_minutes(str(go["BeginTime"]))
        except (TypeError, ValueError):
            arr, dep, begin = 0, 24 * 60, 0
        late_arrival = max(0, arr - self.BUNDLE_LATE_ARRIVAL_MIN)
        red_eye = 1.0 if begin >= 22 * 60 else 0.0
        time_pen = late_arrival / 60.0 + red_eye
        if query["days"] == 1:
            usable = max(1, dep - arr)
            time_pen += 60.0 / usable
        hard_bonus = self._hotel_hard_bonus(hotel) if hotel is not None else 0.0
        return {
            "intercity": feas["intercity"],
            "hotel_cost": feas["hotel_cost"],
            "route_lb": route_lb,
            "time_pen": time_pen,
            "hard_bonus": hard_bonus,
        }

    def _cost_augmented_indices(self, transport_info, ranking, cap):
        """Preference-ranked top-``cap`` unioned with the cheapest-by-cost
        top-``cap``.

        The preference ranking sorts intercity options by time/segment fit, so
        under a tight intercity/overall budget the budget-feasible cheap trains
        can sit far outside the top-``cap`` beam and get truncated away -- which
        prunes every bundle and returns an empty plan. When a budget is active,
        make sure the cheapest options are always in the candidate pool.
        """
        idxs = list(ranking[:cap])
        if self.intercity_budget is None and self.overall_budget is None:
            return idxs
        seen = set(idxs)
        try:
            cost_sorted = sorted(
                range(len(transport_info)),
                key=lambda i: (
                    float(transport_info.iloc[i]["Cost"])
                    if pd.notna(transport_info.iloc[i]["Cost"])
                    else float("inf")
                ),
            )
        except (KeyError, TypeError, ValueError):
            return idxs
        for i in cost_sorted[:cap]:
            if i not in seen:
                idxs.append(i)
                seen.add(i)
        return idxs

    def _enumerate_bundles(self, query, go_info, back_info, ranking_go, ranking_hotel):
        """Bounded (go, back, hotel) triples from per-axis rankings."""
        go_cap = self.bundle_go_top
        back_cap = self.bundle_back_top
        hotel_cap = self.bundle_hotel_top
        bundles = []
        go_indices = self._cost_augmented_indices(go_info, ranking_go, go_cap)
        for gi in go_indices:
            go = go_info.iloc[gi]
            if pd.isna(go["Cost"]):
                continue
            ranking_back = self.ranking_intercity_transport_back(back_info, query, go)
            back_indices = self._cost_augmented_indices(back_info, ranking_back, back_cap)
            for bi in back_indices:
                back = back_info.iloc[bi]
                if pd.isna(back["Cost"]):
                    continue
                if query["days"] > 1:
                    for hi in ranking_hotel[:hotel_cap]:
                        bundles.append((gi, bi, hi))
                else:
                    bundles.append((gi, bi, None))
        return bundles

    def _score_bundles_pass(self, query, go_info, back_info, bundles, relax_poi):
        accommodations = self.memory["accommodations"]
        scored_rows = []
        enumerated = 0
        for gi, bi, hi in bundles:
            enumerated += 1
            go = go_info.iloc[gi]
            back = back_info.iloc[bi]
            hotel = accommodations.iloc[hi] if hi is not None else None
            feas = self._bundle_feasible(query, go, back, hotel, relax_poi=relax_poi)
            if feas is None:
                continue
            components = self._bundle_score_components(query, go, back, hotel, feas)
            scored_rows.append(
                {
                    "gi": gi,
                    "bi": bi,
                    "hi": hi,
                    "components": components,
                    "budget_ok": feas.get("budget_ok", True),
                }
            )
            if self._search_time_exceeded():
                break
        return scored_rows, enumerated

    def _ranked_bundles(self, query, go_info, back_info, ranking_go, ranking_hotel):
        """Feasible-filtered, geo/budget-scored top-N (go, back, hotel) bundles."""
        bundles = self._enumerate_bundles(
            query, go_info, back_info, ranking_go, ranking_hotel
        )
        scored_rows, enumerated = self._score_bundles_pass(
            query, go_info, back_info, bundles, relax_poi=False
        )
        relaxed = False
        if not scored_rows and not self._search_time_exceeded():
            # Every bundle was pruned (commonly by the required-POI reachability
            # lower bounds). Retry with those relaxed so DFS still gets a beam to
            # work with instead of falling straight through to an empty plan.
            scored_rows, enumerated = self._score_bundles_pass(
                query, go_info, back_info, bundles, relax_poi=True
            )
            relaxed = bool(scored_rows)
        print(
            f"bundle scorer: enumerated={enumerated} feasible={len(scored_rows)}"
            f"{' (poi-relaxed)' if relaxed else ''}"
        )
        if not scored_rows:
            return []
        # Prefer bundles that respect the overall-budget lower bound, but only if
        # at least one exists; otherwise keep all so DFS/best-effort still runs.
        budget_ok_rows = [row for row in scored_rows if row.get("budget_ok", True)]
        if budget_ok_rows:
            scored_rows = budget_ok_rows
        comps = [row["components"] for row in scored_rows]
        n_ic = self._normalize_lower_better([c["intercity"] for c in comps])
        n_hotel = self._normalize_lower_better([c["hotel_cost"] for c in comps])
        n_route = self._normalize_lower_better([c["route_lb"] for c in comps])
        n_time = self._normalize_lower_better([c["time_pen"] for c in comps])
        n_bonus = self._normalize_lower_better([-c["hard_bonus"] for c in comps])
        weights = self._bundle_weights(comps, days=query.get("days"))
        ranked = []
        for i, row in enumerate(scored_rows):
            score = (
                weights["intercity"] * n_ic[i]
                + weights["hotel"] * n_hotel[i]
                + weights["route"] * n_route[i]
                + weights["time"] * n_time[i]
                + weights["bonus"] * n_bonus[i]
            )
            tie = comps[i]["intercity"] + comps[i]["hotel_cost"] + comps[i]["route_lb"]
            ranked.append((score, tie, row))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return [row for _, _, row in ranked[: self.bundle_top_n]]

    def _reset_visiting_state(self):
        """Clear per-attempt DFS visiting state before trying a new bundle."""
        self.restaurants_visiting = []
        self.attractions_visiting = []
        self.food_type_visiting = []
        self.spot_type_visiting = []
        self.attraction_names_visiting = []
        self.restaurant_names_visiting = []
        self.all_satisfy_flag = False
        if self.enable_dfs_memoization:
            self._dfs_state_seen = set()

    def _bundle_summary_for_llm(self, query, go_info, back_info, ranked):
        accommodations = self.memory["accommodations"]
        summary = []
        for pos, row in enumerate(ranked):
            go = go_info.iloc[row["gi"]]
            back = back_info.iloc[row["bi"]]
            comps = row["components"]
            summary.append(
                {
                    "id": pos,
                    "go_id": go.get("FlightID") or go.get("TrainID"),
                    "go_depart": str(go.get("BeginTime")),
                    "go_arrive": str(go.get("EndTime")),
                    "back_id": back.get("FlightID") or back.get("TrainID"),
                    "back_depart": str(back.get("BeginTime")),
                    "hotel": (
                        accommodations.iloc[row["hi"]]["name"]
                        if row["hi"] is not None
                        else None
                    ),
                    "intercity_cost": round(float(comps["intercity"]), 1),
                    "hotel_cost": round(float(comps["hotel_cost"]), 1),
                    "must_route_lb": round(float(comps["route_lb"]), 1),
                }
            )
        return summary

    def _llm_rerank_bundles(self, query, go_info, back_info, ranked):
        """Reorder the deterministic top-N with the LLM for hard queries only.

        Rerank-only: the LLM never invents candidates nor asserts feasibility.
        On any failure or when disabled/untriggered it returns the deterministic
        order unchanged, so evaluation stays reproducible.
        """
        if not self.use_llm_bundle_rerank or not ranked:
            return ranked
        llm_name = getattr(getattr(self, "backbone_llm", None), "name", "")
        if not self.backbone_llm or llm_name == "EmptyLLM":
            return ranked

        # Narrow trigger: only hard queries (>=2 must POIs or tight budget).
        n_must = len(self.must_see_attraction or []) + len(self.must_visit_restaurant or [])
        min_cost = 0.0
        if ranked:
            first = ranked[0]["components"]
            min_cost = float(first["intercity"]) + float(first["hotel_cost"]) + self._must_restaurant_price_floor()
        budget_tight = (
            self.overall_budget is not None
            and self.overall_budget > 0
            and min_cost / self.overall_budget > 0.8
        )
        if not (n_must >= 2 or budget_tight):
            return ranked

        candidates = self._bundle_summary_for_llm(query, go_info, back_info, ranked)
        cache_key = json.dumps(
            {"uid": query.get("uid"), "cand": candidates}, ensure_ascii=False, sort_keys=True
        )
        if cache_key in self._bundle_rerank_cache:
            order = self._bundle_rerank_cache[cache_key]
            return self._apply_bundle_order(ranked, order)

        payload = {
            "task": "Rank travel-start bundles by how likely a full itinerary is feasible. Do NOT invent data; only reorder by id.",
            "query": query.get("nature_language", ""),
            "must_attractions": self.must_see_attraction or [],
            "must_restaurants": self.must_visit_restaurant or [],
            "overall_budget": self.overall_budget,
            "days": query.get("days"),
            "candidates": candidates,
            "instructions": "Prefer bundles whose hotel is close to the must POIs (small must_route_lb) and whose total cost leaves budget for POIs. Return JSON {\"order\": [ids best-first]}.",
        }
        messages = [
            {"role": "system", "content": "You rerank deterministic travel-start bundles. Output JSON only."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        try:
            start = time.time()
            response = self.backbone_llm(messages, one_line=False, json_mode=True)
            self.llm_inference_time_count += time.time() - start
            parsed = json.loads(response)
            order = parsed.get("order", parsed if isinstance(parsed, list) else [])
            order = [int(i) for i in order if isinstance(i, (int, float))]
            self._bundle_rerank_cache[cache_key] = order
            self.llm_rec_count += 1
            return self._apply_bundle_order(ranked, order)
        except Exception:
            self.llm_rec_format_error += 1
            self._bundle_rerank_cache[cache_key] = []
            return ranked

    def _apply_bundle_order(self, ranked, order):
        """Reorder ranked by LLM-provided id list; append any omitted, dedup."""
        if not order:
            return ranked
        seen = set()
        reordered = []
        for i in order:
            if 0 <= i < len(ranked) and i not in seen:
                reordered.append(ranked[i])
                seen.add(i)
        for i in range(len(ranked)):
            if i not in seen:
                reordered.append(ranked[i])
        return reordered

    def _bundle_search(self, query, go_info, back_info, ranking_go, ranking_hotel, poi_plan):
        accommodations = self.memory["accommodations"]
        ranked = self._ranked_bundles(query, go_info, back_info, ranking_go, ranking_hotel)
        if not ranked:
            # No bundle survived hard pruning (e.g. every intercity option over
            # its budget). Fall back to best-effort rather than an empty plan.
            self.default_plan["backtrack_count"] = self.backtrack_count
            return self._best_effort_search_result(query, None, poi_plan)

        ranked = self._llm_rerank_bundles(query, go_info, back_info, ranked)

        # Run DFS on the top bundle_dfs_top first, then the remaining bundles as
        # fallback while time remains (avoids a too-narrow beam missing a solution).
        for pos in range(len(ranked)):
            if self._search_time_exceeded():
                self.default_plan["backtrack_count"] = self.backtrack_count
                return self._best_effort_search_result(query, None, poi_plan)

            row = ranked[pos]
            go = go_info.iloc[row["gi"]]
            back = back_info.iloc[row["bi"]]
            poi_plan["go_transport"] = go
            poi_plan["back_transport"] = back
            people = query["people_number"]
            self.intercity_cost = (float(go["Cost"]) + float(back["Cost"])) * people
            if row["hi"] is not None:
                hotel = accommodations.iloc[row["hi"]]
                poi_plan["accommodation"] = hotel
                self.required_rooms = self._bundle_hotel_rooms(query, hotel)
                self.hotel_cost = float(hotel["price"]) * self.required_rooms * (query["days"] - 1)
            else:
                poi_plan.pop("accommodation", None)
                self.hotel_cost = 0

            self._current_poi_plan = poi_plan
            self._reset_visiting_state()
            print(
                "bundle search: go={} back={} hotel={} (rank {}/{})".format(
                    go.get("FlightID") or go.get("TrainID"),
                    back.get("FlightID") or back.get("TrainID"),
                    poi_plan.get("accommodation", {}).get("name") if row["hi"] is not None else "-",
                    pos + 1,
                    len(ranked),
                )
            )
            try:
                success, plan = self.dfs_poi(
                    query, poi_plan, plan=[], current_time="", current_position=""
                )
            except TimeOutError:
                print("TimeOutError")
                return False, {"error_info": "TimeOutError"}

            if success:
                return True, plan
            if self._is_terminal_plan_failure(plan):
                return False, plan
            self.backtrack_count += 1

        self.default_plan["backtrack_count"] = self.backtrack_count
        return self._best_effort_search_result(query, None, poi_plan)

    def _rank_hotels_for_innercity_budget(self, ranking_idx, hotel_info, query, poi_plan):
        if not ranking_idx:
            return ranking_idx

        if self.segment_index is not None:
            ranking_idx = self.segment_index.rank_hotels(
                self._segment_query(query),
                hotel_info,
                poi_plan.get("go_transport"),
                poi_plan.get("back_transport"),
                ranking_idx,
            )

        anchors = []
        go_transport = poi_plan.get("go_transport")
        back_transport = poi_plan.get("back_transport")
        if go_transport is not None:
            anchors.append(go_transport["To"])
        if back_transport is not None:
            anchors.append(back_transport["From"])
        # An early named POI is effectively another fixed terminal: choosing a
        # cheap airport hotel tens of kilometres away can make ``start <= T``
        # impossible or consume the whole inner-city budget the next morning.
        for poi_name, arrive_info in (self.activities_arrive_time_dict or {}).items():
            if (
                arrive_info
                and arrive_info[0] == "early"
                and time_compare_if_earlier_equal(arrive_info[1], "09:00")
                and poi_name not in anchors
            ):
                anchors.append(poi_name)
        # Geo-aware: every must-visit POI is a fixed destination the plan must
        # reach. Anchoring the hotel on them pushes route_cost up for hotels far
        # from the required POIs (e.g. airport hotels vs downtown must sites).
        if self.geo_anchor_must:
            for poi_name in self._must_poi_anchor_names():
                if poi_name not in anchors:
                    anchors.append(poi_name)

        ranking_idx = [
            idx
            for idx in ranking_idx
            if self._hotel_satisfies_hard_constraints(hotel_info.iloc[idx])
        ]
        if not ranking_idx:
            return []

        previous_poi_plan = getattr(self, "_current_poi_plan", None)
        self._current_poi_plan = poi_plan
        policy = self._build_hotel_ranking_weights(query)
        self._current_poi_plan = previous_poi_plan
        weights = policy.get("weights", {})
        scored = []
        for idx in ranking_idx:
            row = hotel_info.iloc[idx]
            route_cost = 0.0
            route_distance = 0.0
            route_hits = 0
            for anchor in anchors:
                best = (
                    self.segment_index._best_intracity_segment(
                        query["target_city"], anchor, row["name"]
                    )
                    if self.segment_index is not None
                    else None
                )
                if best is not None:
                    route_hits += 1
                    route_cost += float(best.get("cost", 0) or 0)
                    route_distance += float(best.get("distance", 0) or 0)
                else:
                    dist = self.calculate_distance(query, anchor, row["name"])
                    if dist is not None:
                        route_hits += 1
                        route_distance += float(dist)
            if route_hits == 0:
                route_cost = 10**6
                route_distance = 10**6
            scored.append(
                {
                    "idx": idx,
                    "route_cost": route_cost,
                    "route_distance": route_distance,
                    "price": float(row.get("price", 0) or 0),
                    "hard_bonus": self._hotel_hard_bonus(row),
                }
            )

        route_cost_norm = self._normalize_lower_better([item["route_cost"] for item in scored])
        route_distance_norm = self._normalize_lower_better([item["route_distance"] for item in scored])
        price_norm = self._normalize_lower_better([item["price"] for item in scored])
        weighted = []
        for pos, item in enumerate(scored):
            score = (
                float(weights.get("route_cost", 0.45)) * route_cost_norm[pos]
                + float(weights.get("route_distance", 0.20)) * route_distance_norm[pos]
                + float(weights.get("poi_price", 0.35)) * price_norm[pos]
                - item["hard_bonus"]
            )
            weighted.append(
                (
                    (
                        score,
                        item["route_cost"],
                        item["route_distance"],
                        item["price"],
                        pos,
                    ),
                    item["idx"],
                )
            )

        return [idx for _, idx in sorted(weighted, key=lambda item: item[0])]

    def _estimate_intracity_travel_minutes(self, city, start, end):
        if self.segment_index is not None:
            seg = self.segment_index._best_intracity_segment(city, start, end)
            if seg is not None:
                duration = seg.get("duration")
                if duration is not None:
                    return max(int(round(float(duration))), 0)
        return 45

    def _required_poi_constraints_present(self):
        return any(
            getattr(self, attr, None)
            for attr in (
                "must_see_attraction",
                "must_visit_restaurant",
                "must_see_attraction_type",
                "must_visit_restaurant_type",
            )
        )

    def _poi_has_visit_window(
        self,
        query,
        start_position,
        earliest_time,
        poi,
        latest_position=None,
        latest_time=None,
        min_visit_buffer=30,
    ):
        if poi is None or not start_position or not earliest_time:
            return True
        name = poi.get("name")
        if not name:
            return True
        travel_min = self._estimate_intracity_travel_minutes(
            query["target_city"], start_position, name
        )
        arrived = add_time_delta(earliest_time, travel_min)
        open_min = time_to_minutes(str(poi.get("opentime", "00:00")))
        close_min = time_to_minutes(str(poi.get("endtime", "23:59")))
        arrived_min = time_to_minutes(arrived)
        if close_min <= open_min:
            close_min += 24 * 60
        visit_start = max(arrived_min, open_min)
        start_constraint = (getattr(self, "activities_arrive_time_dict", None) or {}).get(name)
        if (
            start_constraint
            and start_constraint[0] == "early"
            and visit_start > time_to_minutes(start_constraint[1])
        ):
            return False
        visit_end_limit = close_min
        if latest_position and latest_time:
            leave_travel_min = self._estimate_intracity_travel_minutes(
                query["target_city"], name, latest_position
            )
            visit_end_limit = min(
                visit_end_limit,
                time_to_minutes(str(latest_time)) - leave_travel_min,
            )
        min_visit_end = visit_start + min_visit_buffer
        leave_constraint = (getattr(self, "activities_leave_time_dict", None) or {}).get(name)
        if leave_constraint and leave_constraint[0] == "late":
            min_visit_end = max(min_visit_end, time_to_minutes(leave_constraint[1]))
        return min_visit_end <= visit_end_limit

    def _can_reach_poi_after_arrival(self, query, go_row, poi_names, poi_df):
        if not poi_names or poi_df is None or poi_df.empty:
            return True
        arrival_station = go_row.get("To")
        arrival_time = go_row.get("EndTime")
        if not arrival_station or not arrival_time:
            return True
        city = query["target_city"]
        min_visit_buffer = 30
        for name in poi_names:
            matches = poi_df[poi_df["name"] == name]
            if matches.empty:
                continue
            if not self._poi_has_visit_window(
                query, arrival_station, arrival_time, matches.iloc[0], min_visit_buffer=min_visit_buffer
            ):
                return False
        return True

    def _can_reach_poi_type_after_arrival(self, query, go_row, required_types, poi_df, type_col):
        if not required_types or poi_df is None or poi_df.empty:
            return True
        arrival_station = go_row.get("To")
        arrival_time = go_row.get("EndTime")
        if not arrival_station or not arrival_time:
            return True
        for required_type in required_types:
            matches = poi_df[poi_df[type_col] == required_type]
            if matches.empty:
                continue
            if not any(
                self._poi_has_visit_window(
                    query, arrival_station, arrival_time, row
                )
                for _, row in matches.iterrows()
            ):
                return False
        return True

    def _can_reach_poi_between_intercity(
        self, query, go_row, back_row, poi_names, poi_df
    ):
        if not poi_names or poi_df is None or poi_df.empty:
            return True
        arrival_station = go_row.get("To")
        arrival_time = go_row.get("EndTime")
        depart_station = back_row.get("From")
        depart_time = back_row.get("BeginTime")
        if not arrival_station or not arrival_time or not depart_station or not depart_time:
            return True
        for name in poi_names:
            matches = poi_df[poi_df["name"] == name]
            if matches.empty:
                continue
            if not self._poi_has_visit_window(
                query,
                arrival_station,
                arrival_time,
                matches.iloc[0],
                latest_position=depart_station,
                latest_time=depart_time,
            ):
                return False
        return True

    def _can_reach_poi_type_between_intercity(
        self, query, go_row, back_row, required_types, poi_df, type_col
    ):
        if not required_types or poi_df is None or poi_df.empty:
            return True
        arrival_station = go_row.get("To")
        arrival_time = go_row.get("EndTime")
        depart_station = back_row.get("From")
        depart_time = back_row.get("BeginTime")
        if not arrival_station or not arrival_time or not depart_station or not depart_time:
            return True
        for required_type in required_types:
            matches = poi_df[poi_df[type_col] == required_type]
            if matches.empty:
                continue
            if not any(
                self._poi_has_visit_window(
                    query,
                    arrival_station,
                    arrival_time,
                    row,
                    latest_position=depart_station,
                    latest_time=depart_time,
                )
                for _, row in matches.iterrows()
            ):
                return False
        return True

    def _can_visit_must_see_after_go_arrival(self, query, go_row):
        if not self.must_see_attraction:
            return True
        return self._can_reach_poi_after_arrival(
            query,
            go_row,
            self.must_see_attraction,
            self.memory.get("attractions"),
        )

    def _can_visit_must_restaurant_after_go_arrival(self, query, go_row):
        if not self.must_visit_restaurant:
            return True
        return self._can_reach_poi_after_arrival(
            query,
            go_row,
            self.must_visit_restaurant,
            self.memory.get("restaurants"),
        )

    def _can_satisfy_required_poi_after_go_arrival(self, query, go_row):
        return self._can_visit_must_see_after_go_arrival(
            query, go_row
        ) and self._can_visit_must_restaurant_after_go_arrival(
            query, go_row
        ) and self._can_reach_poi_type_after_arrival(
            query,
            go_row,
            getattr(self, "must_see_attraction_type", None),
            self.memory.get("attractions"),
            "type",
        ) and self._can_reach_poi_type_after_arrival(
            query,
            go_row,
            getattr(self, "must_visit_restaurant_type", None),
            self.memory.get("restaurants"),
            "cuisine",
        )

    def _can_satisfy_required_poi_between_intercity(self, query, go_row, back_row):
        if query.get("days", 0) > 2:
            return True
        return self._can_reach_poi_between_intercity(
            query,
            go_row,
            back_row,
            getattr(self, "must_see_attraction", None),
            self.memory.get("attractions"),
        ) and self._can_reach_poi_between_intercity(
            query,
            go_row,
            back_row,
            getattr(self, "must_visit_restaurant", None),
            self.memory.get("restaurants"),
        ) and self._can_reach_poi_type_between_intercity(
            query,
            go_row,
            back_row,
            getattr(self, "must_see_attraction_type", None),
            self.memory.get("attractions"),
            "type",
        ) and self._can_reach_poi_type_between_intercity(
            query,
            go_row,
            back_row,
            getattr(self, "must_visit_restaurant_type", None),
            self.memory.get("restaurants"),
            "cuisine",
        )

    def _prioritize_required_poi_feasible_go_trains(
        self, transport_info, query, ordered_indices
    ):
        if not self._required_poi_constraints_present():
            return ordered_indices
        feasible = []
        infeasible = []
        for idx in ordered_indices:
            if self._can_satisfy_required_poi_after_go_arrival(
                query, transport_info.iloc[idx]
            ):
                feasible.append(idx)
            else:
                infeasible.append(idx)
        if not feasible:
            return ordered_indices
        feasible.sort(key=lambda i: transport_info.iloc[i].get("EndTime", "99:99"))
        return feasible + infeasible

    def _prioritize_required_poi_feasible_back_trains(
        self, transport_info, query, selected_go, ordered_indices
    ):
        if not self._required_poi_constraints_present():
            return ordered_indices
        feasible = []
        infeasible = []
        for idx in ordered_indices:
            if self._can_satisfy_required_poi_between_intercity(
                query, selected_go, transport_info.iloc[idx]
            ):
                feasible.append(idx)
            else:
                infeasible.append(idx)
        if not feasible:
            return ordered_indices
        feasible.sort(key=lambda i: transport_info.iloc[i].get("BeginTime", "00:00"), reverse=True)
        return feasible + infeasible

    def ranking_intercity_transport_go(self, transport_info, query):
        time_list = transport_info["BeginTime"].tolist()
        price_list = transport_info["Cost"].tolist()

        # 按时间排序
        sorted_indices = np.argsort(time_list)

        # 如果有预算，过滤掉超预算60%的
        if self.intercity_budget is not None:
            budget_threshold = self.intercity_budget * 0.6
            filtered_indices = [idx for idx in sorted_indices if price_list[idx] <= budget_threshold]
            if filtered_indices:
                sorted_indices = filtered_indices
            # 如果全部超预算，则保持 sorted_indices 不变（按时间排序）

        # 如果有 overall_budget，则使用 time_ranking + price_ranking 排序
        if self.overall_budget is not None:
            # 计算时间排名
            time_ranking = np.zeros(len(time_list), dtype=int)
            for i, idx in enumerate(np.argsort(time_list)):
                time_ranking[idx] = i + 1
            # 计算价格排名
            price_ranking = np.zeros(len(price_list), dtype=int)
            for i, idx in enumerate(np.argsort(price_list)):
                price_ranking[idx] = i + 1
            # 最终综合排序
            combined_ranking = time_ranking + price_ranking
            sorted_indices = list(np.argsort(combined_ranking))

        sorted_indices = list(sorted_indices)
        if self.segment_index is not None:
            segment_order = self.segment_index.rank_intercity(
                self._segment_query(query), "go", transport_info
            )
            allowed = set(sorted_indices)
            reranked = [idx for idx in segment_order if idx in allowed]
            reranked.extend(idx for idx in sorted_indices if idx not in set(reranked))
            sorted_indices = reranked

        return self._prioritize_required_poi_feasible_go_trains(
            transport_info, query, sorted_indices
        )

    def ranking_intercity_transport_back(self, transport_info, query, selected_go):
        time_list = transport_info["BeginTime"].tolist()
        sorted_lst = sorted(enumerate(time_list), key=lambda x: x[1], reverse=True)
        sorted_indices = [index for index, value in sorted_lst]
        time_ranking = np.zeros_like(sorted_indices)
        for i, idx in enumerate(sorted_indices):
            time_ranking[idx] = i + 1

        ranking_idx = list(np.argsort(time_ranking))
        if self.segment_index is not None:
            segment_order = self.segment_index.rank_intercity(
                self._segment_query(query), "back", transport_info
            )
            allowed = set(ranking_idx)
            reranked = [idx for idx in segment_order if idx in allowed]
            reranked.extend(idx for idx in ranking_idx if idx not in set(reranked))
            ranking_idx = reranked

        return self._prioritize_required_poi_feasible_back_trains(
            transport_info, query, selected_go, ranking_idx
        )

    def ranking_hotel(self, hotel_info, query):
        candidate_idx = set(range(len(hotel_info)))

        # ---------- must_live_hotel ----------
        if self.must_live_hotel:
            must_idx = set(
                hotel_info[hotel_info["name"].isin(self.must_live_hotel)].index.tolist()
            )
            if must_idx:
                candidate_idx &= must_idx
            else:
                return []

        # ---------- must_not_live_hotel ----------
        if self.must_not_live_hotel:
            not_idx = set(
                hotel_info[hotel_info["name"].isin(self.must_not_live_hotel)].index.tolist()
            )
            candidate_idx -= not_idx

        # ---------- must_live_hotel_service ----------
        if self.must_live_hotel_feature:
            service_idx = set()
            for idx, row in hotel_info.iterrows():
                if self._hotel_matches_required_feature(row):
                    service_idx.add(idx)
            if service_idx:
                candidate_idx &= service_idx
            else:
                return []

        # ---------- must_not_live_hotel_feature ----------
        if self.must_not_live_hotel_feature:
            candidate_idx = {
                idx
                for idx in candidate_idx
                if not self._hotel_matches_forbidden_feature(hotel_info.loc[idx])
            }
            if not candidate_idx:
                return []

        # ---------- must_live_hotel_location_limit ----------
        if self.must_live_hotel_location_limit:
            location_limit = self.must_live_hotel_location_limit
            location_idx = set()
            for idx, row in hotel_info.iterrows():
                if idx not in candidate_idx:
                    continue
                hotel_name = row["name"]
                for item in location_limit:
                    ok = True
                    for poi_name, max_dist in item.items():
                        dist = self.calculate_distance(query, poi_name, hotel_name)
                        print(f"poi name:{poi_name}, hotel name:{hotel_name}, distance:{dist}")
                        if dist > max_dist:
                            ok = False
                            break
                    if ok:
                        location_idx.add(idx)
            if location_idx:
                candidate_idx &= location_idx
            else:
                return []

        # ---------- 最终排序：按价格升序 ----------
        ranking_idx = []
        cost_list = hotel_info.loc[list(candidate_idx), "price"].tolist()
        sorted_lst = sorted(
            zip(list(candidate_idx), cost_list),
            key=lambda x: x[1]
        )
        for r_i, _ in sorted_lst:
            ranking_idx.append(r_i)

        return ranking_idx

    def select_and_add_breakfast(
        self,
        plan,
        poi_plan,
        current_day,
        current_time,
        current_position,
        transports_sel,
        *,
        start_time="06:00",
        end_time="06:30",
    ):
        hotel_name = poi_plan["accommodation"]["name"]
        innercity = transports_sel or []
        if current_position == hotel_name:
            innercity = []
        plan[current_day]["activities"] = self.add_poi(
            plan[current_day]["activities"],
            hotel_name,
            "breakfast",
            0,
            0,
            start_time,
            end_time,
            innercity_transports=innercity,
        )
        if getattr(self, "_enable_plan_sync", False) and getattr(self, "query", None) is not None:
            if not self._after_append_activity(self.query, plan, current_day):
                self._pop_last_activity_if_matches(
                    plan, current_day, "breakfast", hotel_name
                )
        return plan

    def select_next_poi_type(self, candidates_type, plan, poi_plan, current_day, current_time, current_position):

        if current_day == self.query["days"] - 1:
            if time_compare_if_earlier_equal(poi_plan["back_transport"]["BeginTime"],
                                             add_time_delta(current_time, 180)):
                return "back-intercity-transport", ["back-intercity-transport"]

        if self._innercity_budget_saver_enabled() and not self._has_pending_required_poi():
            if current_day == self.query["days"] - 1 and "back-intercity-transport" in candidates_type:
                return "back-intercity-transport", candidates_type
            if "hotel" in candidates_type:
                return "hotel", candidates_type

        if len(candidates_type) == 1:
            return candidates_type[0], candidates_type

        if self.query["days"] > 1 and time_compare_if_earlier_equal(current_time, "6:00"):
            if "hotel" in candidates_type:
                return "hotel", candidates_type
            else:
                return candidates_type[0], candidates_type

        # Arrive-time restaurant: a named restaurant must be visited before T
        # (e.g. <=11:00). Take lunch BEFORE morning attractions push the clock past
        # the deadline, so the required restaurant can be served at the early-lunch
        # slot. Tightly gated: only when such a restaurant is still pending.
        if "lunch" in candidates_type and self._pending_early_arrive_restaurant(current_time):
            return "lunch", candidates_type

        if time_compare_if_earlier_equal("08:30", current_time) and time_compare_if_earlier_equal(current_time, "10:30"):
            if "attraction" in candidates_type:
                return "attraction", candidates_type
            else:
                return candidates_type[0], candidates_type

        # V2: protect DDR. Once lunch/dinner windows are close, take the meal
        # before another attraction can push the plan out of the valid meal time.
        if time_compare_if_earlier_equal("10:30", current_time) and time_compare_if_earlier_equal(current_time, "13:00"):
            if "lunch" in candidates_type:
                return "lunch", candidates_type
            else:
                if "attraction" in candidates_type:
                    return "attraction", candidates_type
                else:
                    return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal("15:30", current_time) and time_compare_if_earlier_equal(current_time, "20:00"):
            if "dinner" in candidates_type:
                return "dinner", candidates_type
            else:
                if "attraction" in candidates_type:
                    return "attraction", candidates_type
                else:
                    return candidates_type[0], candidates_type

        if time_compare_if_earlier_equal(current_time, "20:30"):
            if "attraction" in candidates_type:
                return "attraction", candidates_type

        if "dinner" in candidates_type:
            return "dinner", candidates_type

        return "hotel", candidates_type

    def _default_attraction_minutes(self, legacy_minutes):
        """Default attraction dwell for the main DFS search / add_attraction path.

        ``legacy_minutes`` is the call site's original hardcoded value (kept as
        the fallback when ``enable_compact_dwell`` is off, to preserve exact
        prior behavior). Callers with a per-POI hard minimum still apply it
        downstream via ``select_poi_time`` / ``_scheduled_poi_times``.
        """
        if self.enable_compact_dwell:
            return self.compact_attraction_minutes
        return legacy_minutes

    def _default_meal_minutes(self, legacy_minutes):
        """Default meal dwell for the main DFS search path. See
        ``_default_attraction_minutes`` for the compact-dwell rationale."""
        if self.enable_compact_dwell:
            return self.compact_meal_minutes
        return legacy_minutes

    def select_poi_time(self, poi_name, default_minutes):
        """
        返回某 POI 的停留时长（分钟）。
        若用户约束 activities_stay_time_dict 指定了该 POI 的最短停留，
        则取 max(默认, 约束要求)，确保满足停留时间硬约束；否则用默认。
        """
        if self.activities_stay_time_dict is not None:
            required = self.activities_stay_time_dict.get(poi_name)
            if required is not None:
                try:
                    return max(int(default_minutes), int(required))
                except (TypeError, ValueError):
                    return default_minutes
        return default_minutes

    def _scheduled_poi_times(
        self, poi_name, arrival_time, opentime, endtime, default_minutes, poi_type
    ):
        """Return a visit interval that satisfies this POI's extracted time window.

        The verifier represents a visit window as ``start <= latest_start`` and
        ``end >= earliest_end``.  Merely filtering an already-created 60/90
        minute visit loses feasible cases (in particular restaurant windows):
        the selected activity needs to be deliberately extended to the required
        end time before it is appended to the plan.
        """
        start_time = opentime if time_compare_if_earlier_equal(arrival_time, opentime) else arrival_time
        if poi_type == "lunch" and time_compare_if_earlier_equal(start_time, "11:00"):
            start_time = "11:00"
        elif poi_type == "dinner" and time_compare_if_earlier_equal(start_time, "17:00"):
            start_time = "17:00"

        arrive_info = (self.activities_arrive_time_dict or {}).get(poi_name)
        if arrive_info and arrive_info[0] == "early" and not time_compare_if_earlier_equal(
            start_time, arrive_info[1]
        ):
            return None

        duration = self.select_poi_time(poi_name, default_minutes)
        leave_info = (self.activities_leave_time_dict or {}).get(poi_name)
        if leave_info and leave_info[0] == "late":
            required_end = leave_info[1]
            # The evaluator's windows are same-day.  If the planned start is
            # already after the requested end, the ordinary visit duration is
            # enough; otherwise extend exactly to the lower-bound end time.
            if time_compare_if_earlier_equal(start_time, required_end):
                duration = max(duration, get_time_delta(start_time, required_end))

        end_time = add_time_delta(start_time, duration)
        # A closing time earlier than opening time denotes overnight service and
        # must not be used as a same-day upper bound.
        if (
            time_compare_if_earlier_equal(endtime, end_time)
            and not time_compare_if_earlier_equal(endtime, opentime)
        ):
            end_time = endtime

        if arrive_leave_violated(self, poi_name, start_time, end_time):
            return None
        return start_time, end_time

    def extract_user_constraints_by_DSL(self, query):
        import re

        # 统一转成字符串，防止 None 或非字符串类型
        dsl = query.get("hard_logic_py", "")
        if isinstance(dsl, (list, tuple)):
            dsl = "\n".join(str(item) for item in dsl)
        elif not isinstance(dsl, str):
            dsl = str(dsl) if dsl is not None else ""

        def _unescape_literal_text(text):
            return text.replace("\\'", "'").replace('\\"', '"')

        def extract_list(s):
            items = []
            idx = 0
            while idx < len(s):
                while idx < len(s) and s[idx] in " \t\r\n,":
                    idx += 1
                if idx >= len(s):
                    break
                quote = s[idx] if s[idx] in "'\"" else None
                if quote is None:
                    end = s.find(",", idx)
                    if end == -1:
                        end = len(s)
                    value = s[idx:end].strip()
                    if value:
                        items.append(value)
                    idx = end + 1
                    continue
                content_start = idx + 1
                closing = None
                scan = content_start
                while scan < len(s):
                    if s[scan] == quote and s[scan - 1] != "\\":
                        rest = s[scan + 1 :].lstrip()
                        if not rest or rest[0] in ",}])":
                            closing = scan
                            break
                    scan += 1
                if closing is None:
                    break
                items.append(_unescape_literal_text(s[content_start:closing]))
                idx = closing + 1
            if items:
                return items
            stripped = s.strip()
            return [stripped] if stripped else []

        def _literal_value(text):
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return text.strip("'\"")

        activity_literal = r"(?P<name>'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\")"
        any_activity_literal = r"(?P<name>'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\")"

        def extract_activity_time_pairs(dsl_str, value_pattern, converter):
            normalized = normalize_hard_logic_constraint(dsl_str)
            pattern = re.compile(
                r"if\s+activity_position\(activity\)\s*==\s*"
                + activity_literal
                + r".*?"
                + value_pattern,
                flags=re.S,
            )
            pairs = []
            for match in pattern.finditer(normalized):
                pairs.append((_literal_value(match.group("name")), converter(match)))
            return pairs

        def parse_single_dsl(dsl_str, query):
            """对单条 DSL 进行匹配"""
            # Use the same canonical concept labels and legacy quote repair as
            # the official evaluator before extracting planner constraints.
            # Otherwise names such as ``Chef's ...`` are truncated and English
            # type aliases (for example ``red tourism sites``) never match the
            # database values used by search.
            dsl_str = normalize_concept_constraint_source(dsl_str)
            dsl_str = normalize_hard_logic_constraint(dsl_str)
            res = {}

            def _append_unique(key, values):
                if not values:
                    return
                current = res.get(key)
                if current is None:
                    current = []
                if not isinstance(current, list):
                    current = [current]
                for value in values:
                    if value is not None and value not in current:
                        current.append(value)
                res[key] = current

            def _extract_set_constraints(var_name, negative=False):
                if negative:
                    patterns = [
                        rf"result\s*=\s*not\s*\(\s*\{{([^}}]*)\}}\s*(?:&|<=)\s*{var_name}",
                        rf"result\s*=\s*not\s*\(\s*{var_name}\s*(?:&|<=)\s*\{{([^}}]*)\}}",
                    ]
                else:
                    patterns = [
                        rf"result\s*=\s*\(\s*\{{([^}}]*)\}}\s*(?:&|<=)\s*{var_name}",
                        rf"result\s*=\s*\(\s*{var_name}\s*(?:&|<=)\s*\{{([^}}]*)\}}",
                    ]
                values = []
                for pat in patterns:
                    for match in re.finditer(pat, dsl_str):
                        values.extend(extract_list(match.group(1)))
                return values or None

            # all_satisfy
            res["all_satisfy"] = False if re.search(
                r"result_list\s*=\s*\[\]\s*.*\s*result\s*=\s*False\s*.*\s*result\s*=\s*result\s*or\s*r",
                dsl_str,
                flags=re.S
            ) else True

            # attractions
            res["must_see_attraction"] = _extract_set_constraints("attraction_name_set")
            res["must_see_attraction_type"] = _extract_set_constraints("attraction_type_set")
            res["must_not_see_attraction"] = _extract_set_constraints("attraction_name_set", negative=True)
            res["must_not_see_attraction_type"] = _extract_set_constraints("attraction_type_set", negative=True)

            # only_free_attractions
            m = re.search(
                r"attraction_cost\s*\+=\s*activity_cost\(activity\).*?attraction_cost\s*<=\s*0",
                dsl_str, flags=re.S
            )
            if m:
                res["only_free_attractions"] = True
            else:
                res.pop("only_free_attractions", None)  # 如果不存在不会报错

            # activities time
            matches = extract_activity_time_pairs(
                dsl_str,
                r"activity_time\(activity\)\s*>=\s*(?P<value>[0-9]+)",
                lambda match: int(match.group("value")),
            )
            res["activities_stay_time_dict"] = {name: int(time) for name, time in matches} if matches else None

            matches = extract_activity_time_pairs(
                dsl_str,
                r"activity_start_time\(activity\)\s*<=\s*(?P<value>'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\")",
                lambda match: _literal_value(match.group("value")),
            )
            res["activities_arrive_time_dict"] = {name: ["early", t] for name, t in matches} if matches else None

            matches = extract_activity_time_pairs(
                dsl_str,
                r"activity_end_time\(activity\)\s*>=\s*(?P<value>'(?:\\.|[^\\'])*'|\"(?:\\.|[^\\\"])*\")",
                lambda match: _literal_value(match.group("value")),
            )
            res["activities_leave_time_dict"] = {name: ["late", t] for name, t in matches} if matches else None

            # restaurant
            res["must_visit_restaurant"] = _extract_set_constraints("restaurant_name_set")
            res["must_visit_restaurant_type"] = _extract_set_constraints("restaurant_type_set")
            res["must_not_visit_restaurant"] = _extract_set_constraints("restaurant_name_set", negative=True)
            res["must_not_visit_restaurant_type"] = _extract_set_constraints("restaurant_type_set", negative=True)

            def _match_any(var_name):
                # Positive `result = ({...} & var)` / `(var & {...})` is intersection
                # semantics -> "any one of". `<=` (subset) is "all of". NEG uses
                # `result = not(... & ...)` and is excluded by requiring `(` right after `=`.
                return bool(
                    re.search(rf"result\s*=\s*\(\s*\{{[^}}]*\}}\s*&\s*{var_name}", dsl_str)
                    or re.search(rf"result\s*=\s*\(\s*{var_name}\s*&\s*\{{[^}}]*\}}", dsl_str)
                )

            res["attraction_type_match_any"] = _match_any("attraction_type_set")
            res["restaurant_type_match_any"] = _match_any("restaurant_type_set")

            # hotel
            res["must_live_hotel"] = _extract_set_constraints("accommodation_name_set")
            res["must_not_live_hotel"] = _extract_set_constraints("accommodation_name_set", negative=True)
            res["must_live_hotel_feature"] = _extract_set_constraints("accommodation_type_set")
            res["must_not_live_hotel_feature"] = _extract_set_constraints(
                "accommodation_type_set", negative=True
            )

            m = re.search(
                r"poi_distance\(target_city\(plan\)\s*,\s*(['\"])(.+)\1\s*,\s*accommodation_position\)\s*<=\s*([0-9\.]+)",
                dsl_str)
            res["must_live_hotel_location_limit"] = [{m.group(2): float(m.group(3))}] if m else None

            m = re.search(r"room_type\(activity\)\s*!=\s*([0-9]+)", dsl_str)
            res["bed_number"] = int(m.group(1)) if m else None

            m = re.search(r"room_count\(activity\)\s*!=\s*([0-9]+)", dsl_str)
            res["room_number"] = int(m.group(1)) if m else None

            # innercity transport
            res["must_innercity_transport"] = None
            if res["must_innercity_transport"] is None:
                m = re.search(r'result=\(\s*\{(.*?)\}\s*&\s*inner_city_transportation_set', dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None
            if res["must_innercity_transport"] is None:
                m = re.search(r"result=\(innercity_transport_set<=\s*\{([^}]*)\}\s*\)", dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None
            if res["must_innercity_transport"] is None:
                m = re.search(r"result=\(\s*\{([^}]*)\}\s*<=innercity_transport_set\)", dsl_str)
                res["must_innercity_transport"] = extract_list(m.group(1)) if m else None

            m = re.search(r'result\s*=\s*not\s*\(\s*\{(.*?)\}\s*&\s*inner_city_transportation_set', dsl_str)
            res["must_not_innercity_transport"] = extract_list(m.group(1)) if m else None

            # transport rules by distance
            m = re.search(
                r"innercity_transport_type\(activity_transports\(activity\)\)\s*!=\s*'([^']+)'.*?innercity_transport_distance\(activity_transports\(activity\)\)\s*>\s*([0-9\.]+)",
                dsl_str, flags=re.S)
            res["transport_rules_by_distance"] = [{"min_distance": float(m.group(2)), "max_distance": None,
                                                   "transport_type": [m.group(1)]}] if m else None

            # intercity transport
            m = re.search(r'allactivities\(plan\)\[0\]\[\'type\'\]\s*==\s*["\']([^"\']+)["\']', dsl_str)
            res["must_depart_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[0\]\[\'type\'\]\s*!=\s*["\']([^"\']+)["\']', dsl_str)
            res["must_not_depart_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[-1\]\[\'type\'\]\s*==\s*["\']([^"\']+)["\']', dsl_str)
            res["must_return_transport"] = extract_list(m.group(1)) if m else None
            m = re.search(r'allactivities\(plan\)\[-1\]\[\'type\'\]\s*!=\s*["\']([^"\']+)["\']', dsl_str)
            res["must_not_return_transport"] = extract_list(m.group(1)) if m else None

            if res["must_depart_transport"] is None and res["must_not_depart_transport"] is None and res[
                "must_return_transport"] is None and res["must_not_return_transport"] is None:
                m = re.search(r'result=\(\{([^}]*)\}==intercity_transport_set\)', dsl_str)
                res["intercity transport"] = extract_list(m.group(1)) if m else None
                if res.get("intercity transport"):
                    res["must_depart_transport"] = res["intercity transport"]
                    res["must_return_transport"] = res["intercity transport"]

            # budget
            budget_patterns = {
                "attraction_budget": r"attraction_cost\s*<=\s*([0-9]+)",
                "restaurant_budget": r"restaurant_cost\s*<=\s*([0-9]+)",
                "hotel_budget": r"accommodation_cost\s*<=\s*([0-9]+)",
                "innercity_budget": r"inner_city_transportation_cost\s*<=\s*([0-9]+)",
                "intercity_budget": r"inter_city_transportation_cost\s*<=\s*([0-9]+)",
                "overall_budget": r"total_cost\s*<=\s*([0-9]+)"
            }
            for key, pat in budget_patterns.items():
                m = re.search(pat, dsl_str)
                res[key] = float(m.group(1)) if m else None

            m = re.search(r"result=\(hotel_cost/people_count\(plan\)/\(day_count\(plan\)-1\)<=([0-9\.]+)\)", dsl_str)
            hb = float(m.group(1)) if m else None
            if hb != None:
                res["hotel_budget"] = query["days"] * query["people_number"] * hb

            m = re.search(r"result=\(food_cost/food_count/people_count\(plan\)<=([0-9\.]+)\)", dsl_str)
            res["restaurant_budget_per_meal"] = float(m.group(1)) if m else None

            attr_info = self.memory["attractions"]
            res_info = self.memory["restaurants"]
            hotel_info = self.memory.get("accommodations", pd.DataFrame(columns=["name"]))

            # 确保 must_see_attraction / must_visit_restaurant 存在且是列表
            res.setdefault("must_see_attraction", [])
            res.setdefault("must_visit_restaurant", [])

            # 需要处理的三个 key
            dict_keys = [
                "activities_stay_time_dict",
                "activities_arrive_time_dict",
                "activities_leave_time_dict"
            ]

            # 初始化 must_see_attraction / must_visit_restaurant
            if not isinstance(res.get("must_see_attraction"), list):
                res["must_see_attraction"] = []
            if not isinstance(res.get("must_visit_restaurant"), list):
                res["must_visit_restaurant"] = []
            if not isinstance(res.get("must_live_hotel"), list):
                res["must_live_hotel"] = []

            for key in dict_keys:
                if res.get(key) is not None:
                    for name in res[key].keys():  # 直接遍历字典的 key
                        if name in attr_info["name"].values:
                            if name not in res["must_see_attraction"]:
                                res["must_see_attraction"].append(name)
                        elif name in res_info["name"].values:
                            if name not in res["must_visit_restaurant"]:
                                res["must_visit_restaurant"].append(name)

            # Any hard_logic predicate that explicitly mentions an activity position is
            # a search target, even when it is not written as a *_name_set constraint.
            mentioned_names = []
            for match in re.finditer(
                r"activity_position\(activity\)\s*==\s*" + any_activity_literal,
                dsl_str,
            ):
                mentioned_names.append(_literal_value(match.group("name")))
            for name in mentioned_names:
                if name in attr_info["name"].values:
                    _append_unique("must_see_attraction", [name])
                elif name in res_info["name"].values:
                    _append_unique("must_visit_restaurant", [name])
                elif name in hotel_info["name"].values:
                    _append_unique("must_live_hotel", [name])

            order_name_by_idx = {}
            for match in re.finditer(
                r"if\s+activity_position\(activity\)\s*==\s*"
                + any_activity_literal
                + r"\s*:\s*idx_activity(?P<idx>[0-9]+)\s*=\s*i",
                dsl_str,
                flags=re.S,
            ):
                order_name_by_idx[match.group("idx")] = _literal_value(match.group("name"))
            order_pairs = []
            for match in re.finditer(
                r"if\s+idx_activity(?P<before>[0-9]+)\s*<\s*idx_activity(?P<after>[0-9]+)",
                dsl_str,
            ):
                before = order_name_by_idx.get(match.group("before"))
                after = order_name_by_idx.get(match.group("after"))
                if before and after:
                    order_pairs.append((before, after))
                    for name in (before, after):
                        if name in attr_info["name"].values:
                            _append_unique("must_see_attraction", [name])
                        elif name in res_info["name"].values:
                            _append_unique("must_visit_restaurant", [name])
            if order_pairs:
                res["must_visit_order"] = order_pairs

            res_filtered = {k: v for k, v in res.items() if v is not None}

            return res_filtered

        # 先解析完整 DSL
        results_main = parse_single_dsl(dsl, query)

        # 如果 all_satisfy=False，则分割 DSL
        results_list = []
        if not results_main["all_satisfy"]:
            sub_dsls = [s.strip() for s in dsl.split("result_list.append(result)") if s.strip()]
            for sub_dsl in sub_dsls:
                results_list.append(parse_single_dsl(sub_dsl, query))
        else:
            results_list = [results_main]

        return results_main, results_list

    def constraints_validation(self, query, plan, poi_plan):

        self.constraints_validation_count += 1
        repair_full_itinerary(self, query, plan)

        res_plan = {
            "people_number": query["people_number"],
            "start_city": query["start_city"],
            "target_city": query["target_city"],
            "itinerary": plan,
        }
        print("validate the plan [for query {}]: ".format(query["uid"]))
        print(res_plan)

        bool_result = func_commonsense_constraints(query, res_plan, verbose=True)

        # if not bool_result:
        #     exit(0)

        if bool_result:
            self.commonsense_pass_count += 1

        try:
            extracted_vars = get_symbolic_concepts(query, res_plan, need_ood=False)

        except:
            extracted_vars = None

        print(extracted_vars)

        logical_result = evaluate_constraints_py(query["hard_logic_py"], res_plan, verbose=True)

        print(logical_result)

        meal_report = self._daily_meal_report(res_plan)
        res_plan["daily_meal_check"] = meal_report
        res_plan["meal_completeness_pass"] = meal_report["pass"]
        if meal_report["pass"]:
            print("Daily meal completeness passed.")
        else:
            print("Daily meal completeness failed:", meal_report["missing"])


        logical_pass = True
        for idx, item in enumerate(logical_result):
            logical_pass = logical_pass and item

            if item:
                print(query["hard_logic_py"][idx], "passed!")
            else:
                print(query["hard_logic_py"][idx], "failed...")

        self._update_best_plan(query, res_plan, bool_result, logical_result)

        if logical_pass:
            self.logical_pass_count += 1

        bool_result = bool_result and logical_pass
        if self.require_complete_daily_meals:
            bool_result = bool_result and meal_report["pass"]

        if bool_result:
            print("\n Pass! \n")
            self.all_constraints_pass += 1
        else:
            print("\n Failed \n")

        # plan = res_plan

        # print(result)
        # exit(0)

        if bool_result:
            res_plan["search_time_sec"] = time.time() - self.time_before_search
            res_plan["llm_inference_time_sec"] = self.llm_inference_time_count
            return True, res_plan
        else:
            return False, res_plan

    def add_intercity_transport(
            self, activities, intercity_info, innercity_transports=None, tickets=1
    ):
        # cost_per_ticket = intercity_info["Cost"]
        if innercity_transports is None:
            innercity_transports = []

        activity_i = {
            "start_time": intercity_info["BeginTime"],
            "end_time": intercity_info["EndTime"],
            "start": intercity_info["From"],
            "end": intercity_info["To"],
            "price": intercity_info["Cost"], #None if pd.isna(cost_per_ticket) else cost_per_ticket
            "cost": intercity_info["Cost"] * tickets, # None if pd.isna(cost_per_ticket) else cost_per_ticket * tickets,
            "tickets": tickets,
            "transports": innercity_transports,
        }
        # if not pd.isna(intercity_info["TrainID"]):
        #     activity_i["TrainID"] = intercity_info["TrainID"]
        #     activity_i["type"] = "train"
        # elif not pd.isna(intercity_info["FlightID"]):
        #     activity_i["FlightID"] = intercity_info["FlightID"]
        #     activity_i["type"] = "airplane"

        if "TrainID" in intercity_info and not pd.isna(intercity_info["TrainID"]):
            activity_i["TrainID"] = intercity_info["TrainID"]
            activity_i["type"] = "train"
        elif "FlightID" in intercity_info and not pd.isna(intercity_info["FlightID"]):
            activity_i["FlightID"] = intercity_info["FlightID"]
            activity_i["type"] = "airplane"

        activities.append(activity_i)
        return activities

    def add_poi(
            self,
            activities,
            position,
            poi_type,
            price,
            cost,
            start_time,
            end_time,
            innercity_transports,
    ):
        activity_i = {
            "position": position,
            "type": poi_type,
            "price": price,
            "cost": cost,
            "start_time": start_time,
            "end_time": end_time,
            "transports": innercity_transports,
        }

        activities.append(activity_i)
        return activities

    def add_accommodation(
            self,
            current_plan,
            hotel_sel,
            current_day,
            arrived_time,
            required_rooms,
            transports_sel,
    ):
        if not isinstance(current_plan, list):
            return current_plan
        transports_sel = list(transports_sel) if transports_sel else []
        if transports_sel and not self._innercity_transports_valid(transports_sel):
            return current_plan
        arrived_time = clamp_time_to_day_end(arrived_time)
        if self._arrived_time_too_late_for_hotel(arrived_time):
            return current_plan

        ensure_day_plan(current_plan, current_day)
        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=hotel_sel["name"],
            poi_type="accommodation",
            price=int(hotel_sel["price"]),
            cost=int(hotel_sel["price"]) * required_rooms,
            start_time=arrived_time,
            end_time="24:00",
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["room_type"] = hotel_sel["numbed"]
        current_plan[current_day]["activities"][-1]["rooms"] = required_rooms

        if getattr(self, "_enable_plan_sync", False) and getattr(self, "query", None) is not None:
            if not self._after_append_activity(self.query, current_plan, current_day):
                self._pop_last_activity_if_matches(
                    current_plan, current_day, "accommodation", hotel_sel["name"]
                )

        return current_plan

    def add_restaurant(
            self,
            current_plan,
            poi_type,
            poi_sel,
            current_day,
            arrived_time,
            transports_sel,
            *,
            start_time=None,
            end_time=None,
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...
        # if time_compare_if_earlier_equal(endtime, arrived_time):
        #     raise Exception("Add POI error")
        if start_time is None:
            if time_compare_if_earlier_equal(arrived_time, opentime):
                act_start_time = opentime
            else:
                act_start_time = arrived_time
        else:
            act_start_time = start_time

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                act_start_time, "11:00"
        ):
            act_start_time = "11:00"
        if poi_type == "lunch" and time_compare_if_earlier_equal(endtime, "11:00"):
            raise Exception("ERROR: restaurant closed before 11:00")

        if poi_type == "dinner" and time_compare_if_earlier_equal(
                act_start_time, "17:00"
        ):
            act_start_time = "17:00"

        if poi_type == "dinner" and time_compare_if_earlier_equal(endtime, "17:00"):
            if not time_compare_if_earlier_equal(endtime, opentime):
                raise Exception("ERROR: restaurant closed before 17:00")

        if poi_type == "lunch" and time_compare_if_earlier_equal(
                "13:00", act_start_time
        ):
            raise Exception("ERROR: lunch begins after 13:00")
        if poi_type == "dinner" and time_compare_if_earlier_equal(
                "20:00", act_start_time
        ):
            raise Exception("ERROR: dinner begins after 20:00")

        if end_time is None:
            poi_time = self.select_poi_time(poi_sel["name"], self._default_meal_minutes(60))
            act_end_time = add_time_delta(act_start_time, poi_time)
            aet = act_end_time
            # 如果结束时间超过景点关闭时间，则截断为关闭时间
            if time_compare_if_earlier_equal(endtime, act_end_time):
                act_end_time = endtime
                if time_compare_if_earlier_equal(endtime, opentime):  # 营业到第二天
                    act_end_time = aet
        else:
            act_end_time = end_time

        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        if getattr(self, "_enable_plan_sync", False) and getattr(self, "query", None) is not None:
            if not self._after_append_activity(self.query, current_plan, current_day):
                self._pop_last_activity_if_matches(
                    current_plan, current_day, poi_type, poi_sel["name"]
                )
        return current_plan

    def add_attraction(
            self, current_plan, poi_type, poi_sel, current_day, arrived_time, transports_sel
    ):

        # 开放时间
        opentime, endtime = (
            poi_sel["opentime"],
            poi_sel["endtime"],
        )

        # it is closed ...

        opentime, endtime = poi_sel["opentime"], poi_sel["endtime"]
        # it is closed ...
        if time_compare_if_earlier_equal(endtime, arrived_time):
            raise Exception("Add POI error")

        if time_compare_if_earlier_equal(arrived_time, opentime):
            act_start_time = opentime
        else:
            act_start_time = arrived_time

        poi_time = self.select_poi_time(poi_sel["name"], self._default_attraction_minutes(90))
        act_end_time = add_time_delta(act_start_time, poi_time)
        if time_compare_if_earlier_equal(endtime, act_end_time):
            act_end_time = endtime

        current_plan[current_day]["activities"] = self.add_poi(
            activities=current_plan[current_day]["activities"],
            position=poi_sel["name"],
            poi_type=poi_type,
            price=int(poi_sel["price"]),
            cost=int(poi_sel["price"]) * self.query["people_number"],
            start_time=act_start_time,
            end_time=act_end_time,
            innercity_transports=transports_sel,
        )
        current_plan[current_day]["activities"][-1]["tickets"] = self.query["people_number"]

        if getattr(self, "_enable_plan_sync", False) and getattr(self, "query", None) is not None:
            if not self._after_append_activity(self.query, current_plan, current_day):
                self._pop_last_activity_if_matches(
                    current_plan, current_day, poi_type, poi_sel["name"]
                )

        return current_plan

    def check_if_too_late(
            self, query, current_day, current_time, current_position, poi_plan
    ):

        arrived_time = current_time  # 兜底默认值，防止后续分支未赋值时引用导致 NameError

        # The blanket "after 23:00 = give up" rule wrongly prunes a late
        # intercity arrival on a non-final day, where the only remaining move is
        # to go to the hotel and sleep (a perfectly valid plan). Only keep the
        # hard cutoff on the final day (where a late state risks missing the
        # return transport); on other days fall through to the hotel-reachability
        # branch below.
        if (
            current_time != ""
            and time_compare_if_earlier_equal("23:00", current_time)
            and current_day == query["days"] - 1
        ):
            print("too late, after 23:00")
            return True

        if current_time != "" and current_day == query["days"] - 1:
            # We should go back in time ...
            if "back_transport" not in poi_plan:
                return False

            transports_ranking = self.innercity_transports_ranking
            if self.transport_rules_by_distance is not None:
                temp_distance = self.calculate_distance(query, current_position, poi_plan["back_transport"]["From"])
                transports_ranking = self.get_transport_by_distance(temp_distance)

            for transport_type_sel in transports_ranking:

                self.search_nodes += 1

                transports_sel = self.collect_innercity_transport(
                    query["target_city"],
                    current_position,
                    poi_plan["back_transport"]["From"],
                    current_time,
                    transport_type_sel,
                )
                if not isinstance(transports_sel, list):
                    self.backtrack_count += 1
                    print("inner-city transport error, backtrack...")
                    continue

                if len(transports_sel) > 0:
                    arrived_time = transports_sel[-1]["end_time"]
                else:
                    arrived_time = current_time

                if time_compare_if_earlier_equal(arrived_time, poi_plan["back_transport"]["BeginTime"]):
                    return False

            print(
                "Can not go back source-city in time, current POI {}, station arrived time: {}".format(
                    current_position, arrived_time
                )
            )
            return True


        elif current_time != "":
            if "accommodation" in poi_plan:
                hotel_sel = poi_plan["accommodation"]
                transports_ranking = self.innercity_transports_ranking
                if self.transport_rules_by_distance is not None:
                    temp_distance = self.calculate_distance(query, current_position, hotel_sel["name"])
                    transports_ranking = self.get_transport_by_distance(temp_distance)

                for transport_type_sel in transports_ranking:
                    self.search_nodes += 1
                    print("collecting innercity transport to see if possible back to hotel")
                    transports_sel = self.collect_innercity_transport(
                        query["target_city"],
                        current_position,
                        hotel_sel["name"],
                        current_time,
                        transport_type_sel,
                    )
                    if not isinstance(transports_sel, list):
                        self.backtrack_count += 1
                        print("inner-city transport error, backtrack...")
                        continue

                    if len(transports_sel) > 0:
                        arrived_time = transports_sel[-1]["end_time"]
                    else:
                        arrived_time = current_time
                    if time_compare_if_earlier_equal(arrived_time, "24:00"):
                        return False

                print(
                    "Can not go back to hotel, current POI {}, hotel arrived time: {}".format(
                        current_position, arrived_time
                    )
                )
                return True

        return False

    def collect_poi_info_all(self, city, poi_type):
        if poi_type == "accommodation":
            func_name = "accommodations_select"
        elif poi_type == "attraction":
            func_name = "attractions_select"
        elif poi_type == "restaurant":
            func_name = "restaurants_select"
        else:
            raise NotImplementedError

        poi_info = self.env(
            "{func}('{city}', 'name', lambda x: True)".format(func=func_name, city=city)
        )["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                poi_info = pd.concat([poi_info, info_i], axis=0, ignore_index=True)

        # print(poi_info)
        return poi_info

    def collect_innercity_transport(self, city, start, end, start_time, trans_type):

        call_str = (
            'goto("{city}", "{start}", "{end}", "{start_time}", "{trans_type}")'.format(
                city=city,
                start=start,
                end=end,
                start_time=start_time,
                trans_type=trans_type,
            )
        )

        # print(call_str)
        if start == end:
            return []
        info = self.env(call_str)["data"]

        # print(f"transport: {info}")

        if not isinstance(info, list):
            return "No solution"

        if len(info) == 0:
            return []

        if len(info) == 3:
            info[1]["price"] = info[1]["cost"]
            info[1]["tickets"] = self.query["people_number"]
            info[1]["cost"] = info[1]["price"] * info[1]["tickets"]

            info[0]["price"] = info[0]["cost"]
            info[2]["price"] = info[2]["cost"]
        elif info[0]["mode"] == "taxi":
            info[0]["price"] = info[0]["cost"]
            info[0]["cars"] = int((self.query["people_number"] - 1) / 4) + 1
            info[0]["cost"] = info[0]["price"] * info[0]["cars"]
        elif info[0]["mode"] == "walk":
            info[0]["price"] = info[0]["cost"]

        return info

    def collect_intercity_transport(self, source_city, target_city, trans_type):

        info_return = self.env(
            "intercity_transport_select('{source_city}', '{target_city}', '{trans_type}')".format(
                source_city=source_city, target_city=target_city, trans_type=trans_type
            )
        )
        if not info_return["success"]:
            return pd.DataFrame([])
        trans_info = info_return["data"]
        # print(poi_info)
        while True:
            info_i = self.env("next_page()")["data"]
            if len(info_i) == 0:
                break
            else:
                trans_info = pd.concat([trans_info, info_i], axis=0, ignore_index=True)
        # print(poi_info)
        return trans_info
