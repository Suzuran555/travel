"""Team "Antarctic penguins" Phase-2 agent.

run(query) pipeline, all in-process and within the per-query time budget:

  1. NL -> DSL translation with the backbone LLM (vendored hardened
     bilingual prompt stacks -- nl2sl_hybrid_en / nl2sl_hybrid_zh, routed
     per query by lang_router.py -- + mechanical verifiers), then DSL
     canonicalization. The oracle constraint fields of the query are
     STRIPPED before anything else runs -- the agent only ever acts on its
     own generated constraints.
  2. UrbanTripOptimizedV6 symbolic search (vendored, with the production
     flag set that produced our phase-1 result).
  3. In-run enrichment battery (vendored library ports of the phase-1
     enrichment chain), gated exclusively on the GENERATED constraints via
     the stock evaluation modules.

Environment knobs:
  TPC_TIME_BUDGET        total seconds per query (default 290; keep it a few
                         seconds under the harness --timeout)
  TPC_ENRICH_RESERVE     seconds reserved for enrichment after planning (45)
  TPC_EMIT_MARGIN        seconds before budget end by which SOME schema-valid
                         plan is always returned (15); on planner overrun a
                         deterministic env-DB fallback plan is emitted
  TPC_URBANTRIP_KWARGS   JSON dict merged over the built-in planner flags
  (backbone selection: see tpc_llm.py)
"""
import json
import os
import sys
import threading
import time

from chinatravel.agent.base import BaseAgent

from . import env_fix
from .v6 import UrbanTripOptimizedV6
from .enrich.runner import run_battery
from .fallback_plan import build_fallback_plan

# The production planner configuration (URBANTRIP_KWARGS of the phase-1
# submission chain + the honest-Qwen e2e sim).
PROD_PLANNER_FLAGS = {
    "use_bundle_search": True,
    "geo_anchor_must": True,
    "enable_fallback_hard_repair": True,
    "enable_metro_only_prune": True,
    "enable_dynamic_top_k": True,
    "enable_dfs_memoization": True,
    "enable_budget_drop_repair": True,
    "enable_travelday_time_bias": True,
    "travelday_arr_weight": 0.30,
    "travelday_dep_weight": 0.30,
    "enable_transit_time_score": True,
    "transit_signal_min_duration": True,
    "transit_geo_fallback": True,
    "transit_geo_feasibility": True,
    "transit_weight_floor": 3.0,
    "debug": False,
}

ORACLE_FIELDS = ("hard_logic", "hard_logic_py", "hard_logic_nl", "hard_logic_py_nl")

# Wall-clock-dependent debug counters the planner writes into the plan dict.
# They vary run-to-run even when the itinerary is byte-identical (postprocess
# loops are deadline-bounded), so they are stripped from the emitted plan to
# keep the output content-deterministic.
VOLATILE_PLAN_KEYS = (
    "search_time_sec",
    "llm_inference_time_sec",
    "meal_postprocess_attempts",
    "dav_postprocess_attempts",
)


def _np_scalar(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o)}")


class TPCAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="TPC", **kwargs)
        env_fix.apply_class_patches()
        if self.env is not None:
            env_fix.fix_env(self.env)   # run_tpc builds WorldEnv before we load

        self.time_budget = float(os.environ.get("TPC_TIME_BUDGET", "290"))
        # 60 (was 45): the mustpoi repair stage needs headroom on plans where
        # the planner exhausts its slice -- exactly the plans that fail
        self.enrich_reserve = float(os.environ.get("TPC_ENRICH_RESERVE", "60"))
        # a schema-valid plan is ALWAYS emitted this many seconds before the
        # time budget runs out (fallback plan on planner overrun)
        self.emit_margin = float(os.environ.get("TPC_EMIT_MARGIN", "15"))
        self._planner_thread = None

        planner_kwargs = dict(kwargs)
        planner_kwargs.update(PROD_PLANNER_FLAGS)
        extra = os.environ.get("TPC_URBANTRIP_KWARGS")
        if extra:
            try:
                planner_kwargs.update(json.loads(extra))
            except (TypeError, ValueError) as exc:
                print(f"Ignoring invalid TPC_URBANTRIP_KWARGS ({exc})")
        planner_kwargs.setdefault("method", kwargs.get("method", "TPCAgent"))
        planner_kwargs["env"] = self.env
        planner_kwargs["backbone_llm"] = self.backbone_llm
        planner_kwargs.setdefault(
            "external_timeout", max(60.0, self.time_budget - self.enrich_reserve)
        )
        self.planner = UrbanTripOptimizedV6(**planner_kwargs)

    # ------------------------------------------------------------------
    def run(self, query, prob_idx, oralce_translation=False):
        start = time.time()
        deadline = start + self.time_budget
        # hard internal deadline: SOME schema-valid plan is returned by here
        emit_deadline = start + max(1.0, self.time_budget - self.emit_margin)
        self.reset_clock()

        query = dict(query)
        if not oralce_translation:
            # never let the oracle constraint annotations reach the pipeline
            for key in ORACLE_FIELDS:
                query.pop(key, None)

        # a planner thread abandoned by a previous over-budget query unwinds
        # within seconds once its capped LLM request dies; give it a moment so
        # it cannot race this query's search state
        prev = self._planner_thread
        if prev is not None and prev.is_alive():
            prev.join(timeout=20.0)
            if prev.is_alive():
                print("[tpc_agent] WARNING: abandoned planner thread still alive")

        # cap every backbone request to this query's emission deadline: the
        # harness watchdog cannot interrupt a blocking socket read
        try:
            self.backbone_llm.request_deadline = emit_deadline - 2.0
        except Exception:
            pass

        # deterministic budget-exhaustion fallback, built up-front from the
        # env DB (~1s) so emitting it later needs no env access at all
        fallback = None
        try:
            fallback = build_fallback_plan(query, self.env)
        except Exception as exc:
            print(f"[tpc_agent] fallback build failed: {exc}")

        result = {}

        def _planner_job():
            try:
                result["value"] = self.planner.run(
                    query, prob_idx, oralce_translation=oralce_translation
                )
            except BaseException as exc:  # never lose the query to an exception
                result["error"] = exc

        worker = threading.Thread(
            target=_planner_job, name=f"tpc-planner-{prob_idx}", daemon=True
        )
        self._planner_thread = worker
        worker.start()
        worker.join(timeout=max(0.0, emit_deadline - time.time()))
        timed_out = worker.is_alive()
        # the planner redirects stdout/stderr into its per-query logger
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

        succ, plan = False, None
        if not timed_out and "value" in result:
            succ, plan = result["value"]
        elif "error" in result:
            print(f"[tpc_agent] planner raised: {result['error']!r}")
        else:
            print(f"[tpc_agent] planner over emission deadline "
                  f"({time.time() - start:.1f}s), emitting fallback plan")

        if not (isinstance(plan, dict) and plan.get("itinerary")):
            if fallback is not None:
                plan = fallback
                succ = False
            elif plan is None:
                plan = {"error_info": "planner over budget and no fallback available"}

        if isinstance(plan, dict):
            # the planner emits numpy scalars; schema/commonsense evaluation
            # (and the final save) require a pure-JSON plan
            plan = json.loads(json.dumps(plan, ensure_ascii=False, default=_np_scalar))

        translated = getattr(self.planner, "query", None)
        if not isinstance(translated, dict) or translated.get("uid") != query.get("uid"):
            translated = None

        # Timed-out plans come from the best-effort tail where the search has
        # already dropped constraint pruning -- they need the repair battery
        # MOST, so timeout no longer skips it (plan repair beats replanning:
        # the A800 align run shipped 17 plans violating their own generated
        # constraints, all unrepaired). The deadline guard inside run_battery
        # still bounds the extra time.
        if (isinstance(plan, dict) and plan.get("itinerary")
                and translated is not None
                and translated.get("hard_logic_py") is not None
                and time.time() < deadline - 10):
            gate_query = {
                k: v for k, v in translated.items()
                if not str(k).startswith("_urbantrip_")
            }
            try:
                plan = run_battery(
                    query.get("uid", prob_idx), plan, gate_query,
                    self.planner, self.env, deadline - 5,
                    lang=self.lang,
                )
            except Exception as exc:
                print(f"[enrich] battery failed, keeping planner output: {exc}")

        if isinstance(plan, dict):
            # run-to-run volatile debug counters must not reach the output
            for key in VOLATILE_PLAN_KEYS:
                plan.pop(key, None)

        print(f"[tpc_agent] total {time.time() - start:.1f}s (budget {self.time_budget:.0f}s)")
        return succ, plan

    def reset(self):
        pass
