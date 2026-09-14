"""Shared runtime context for the in-run enrichment battery.

The root-repo enrichment scripts were batch tools: they load_query()'d the
whole (oracle-bearing) query set at import and iterated a results directory.
In the Phase-2 harness every pass runs IN-PROCESS on the single query being
answered, gated exclusively on the GENERATED constraints the translator just
produced. This module is the single seam where that swap happens:

  * ``qd`` maps uid -> the TRANSLATED query dict (generated hard_logic_py,
    never the oracle files);
  * ``passes(uid, plan)`` = schema + commonsense + generated-hard-logic, via
    the stock evaluation modules;
  * ``agent_for(city)`` returns the live planner agent (its memory/transport
    helpers back the candidate recall of every pass).

No module here reads query files from disk.
"""
import os

from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
from chinatravel.evaluation.commonsense_constraint import evaluate_commonsense_constraints
from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
from chinatravel.symbol_verification.concept_func import innercity_transport_time
from chinatravel.symbol_verification.hard_constraint import _set_tool_lang
from chinatravel.data.load_datasets import load_json_file

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# package sits at <root>/chinatravel/agent/<pkg>/
_CT_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))

qd = {}          # uid -> translated query (generated constraints installed)
_agents = {}     # city -> live planner agent
_env = None      # shared WorldEnv
_lang = "en"
_cur = None      # current uid (mirrors the old ER._cur diagnostic)
RES = None       # results dir concept does not exist in-run

sch = load_json_file(os.path.join(_CT_ROOT, "evaluation", "output_schema.json"))


def db_path(rel):
    """Absolute path into the stock checkout's EN database."""
    return os.path.join(_CT_ROOT, "environment", "database_en", rel)


def setup(uid, query, agent, world_env, lang="en"):
    global _env, _lang, _cur
    qd.clear()
    qd[uid] = query
    _agents.clear()
    city = query.get("target_city")
    if city:
        _agents[city] = agent
    _agents["__default__"] = agent
    _env = world_env
    _lang = lang
    _cur = uid
    _set_tool_lang(lang)


def get_env():
    return _env


def agent_for(city):
    return _agents.get(city) or _agents["__default__"]


def passes(uid, plan):
    s = set(evaluate_schema_constraints([uid], {uid: plan}, schema=sch)[2])
    c = set(evaluate_commonsense_constraints([uid], qd, {uid: plan}, verbose=False, lang=_lang)[3])
    if uid not in s or uid not in c:
        return False
    l = set(evaluate_hard_constraints_v2([uid], qd, {uid: plan}, env_pass_id=list(c), verbose=False, lang=_lang)[-1])
    return uid in l


# ---- soft-metric helpers (verbatim semantics of the root scripts) -----------

MEALS = ("breakfast", "lunch", "dinner")


def hm(t):
    h, m = str(t).split(":")[:2]
    return int(h) * 60 + int(m)


def mh(x):
    return f"{int(x) // 60:02d}:{int(x) % 60:02d}"


def actpos(a):
    return a.get("position") or a.get("end") or a.get("start")


def soft(plan):
    days = max(1, len(plan.get("itinerary", [])))
    a = sum(1 for d in plan["itinerary"] for x in d["activities"] if x.get("type") == "attraction")
    m = sum(1 for d in plan["itinerary"] for x in d["activities"] if x.get("type") in MEALS)
    return a / (4 * days), (m / days) / 3


def att_of(plan):
    """ATT exactly as eval computes it (leg-average transit minutes)."""
    time_cost = legs = 0
    for d in plan.get("itinerary", []):
        for a in d.get("activities", []):
            tr = a.get("transports", [])
            if tr:
                legs += 1
                time_cost += sum(hm(l["end_time"]) - hm(l["start_time"]) for l in tr)
    avg = time_cost / legs if legs > 0 else -1.0
    return max(0.0, min(1.0, (-1.0 / 105.0) * avg + 8.0 / 7.0))


def soft_of(plan):
    """(dav, ddr, att) or None if the plan has no itinerary (certain fail)."""
    it = plan.get("itinerary") or []
    if not it:
        return None
    days = len(it)
    a = sum(1 for d in it for x in d.get("activities", []) if x.get("type") == "attraction")
    m = sum(1 for d in it for x in d.get("activities", []) if x.get("type") in MEALS)
    dav = max(0.0, min(1.0, a / (4.0 * days)))
    ddr = max(0.0, min(1.0, (m / days) / 3.0))
    return dav, ddr, att_of(plan)
