"""Team "Antarctic penguins" Phase-2 agent.

run(query) pipeline, all in-process and within the per-query time budget:

  1. NL -> DSL translation with the backbone LLM (vendored hardened
     nl2sl_hybrid_en prompts + mechanical verifiers), then DSL
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
  TPC_URBANTRIP_KWARGS   JSON dict merged over the built-in planner flags
  (backbone selection: see tpc_llm.py)
"""
import json
import os
import sys
import time

from chinatravel.agent.base import BaseAgent

from . import env_fix
from .v6 import UrbanTripOptimizedV6
from .enrich.runner import run_battery

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
        self.enrich_reserve = float(os.environ.get("TPC_ENRICH_RESERVE", "45"))

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
        self.reset_clock()

        query = dict(query)
        if not oralce_translation:
            # never let the oracle constraint annotations reach the pipeline
            for key in ORACLE_FIELDS:
                query.pop(key, None)

        try:
            succ, plan = self.planner.run(
                query, prob_idx, oralce_translation=oralce_translation
            )
        finally:
            # the planner redirects stdout/stderr into its per-query logger
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

        if isinstance(plan, dict):
            # the planner emits numpy scalars; schema/commonsense evaluation
            # (and the final save) require a pure-JSON plan
            plan = json.loads(json.dumps(plan, ensure_ascii=False, default=_np_scalar))

        translated = getattr(self.planner, "query", None)
        if not isinstance(translated, dict) or translated.get("uid") != query.get("uid"):
            translated = None

        if (isinstance(plan, dict) and plan.get("itinerary")
                and translated is not None
                and translated.get("hard_logic_py") is not None):
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

        print(f"[tpc_agent] total {time.time() - start:.1f}s (budget {self.time_budget:.0f}s)")
        return succ, plan

    def reset(self):
        pass
