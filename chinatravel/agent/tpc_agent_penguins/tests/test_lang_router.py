"""Unit tests for the bilingual query-language router and its prompt stacks.

Verifies (all offline, no LLM calls):
  1. CJK-ratio detection: zh/en/mixed/degenerate inputs, runner-lang fallback.
  2. The route selects the right hardened prompt stack (zh module carries the
     Chinese rules/few-shots, en module the English ones), both with the full
     rounds-1-4 hardening markers.
  3. The v6 routing decision is reproducible from the same inputs the planner
     uses (nature_language + runner lang).

Run:  .venv/bin/python chinatravel/agent/tpc_agent_penguins/tests/test_lang_router.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", ".."))

from chinatravel.agent.tpc_agent_penguins.lang_router import (  # noqa: E402
    detect_query_lang,
)
from chinatravel.agent.tpc_agent_penguins import nl2sl_hybrid_en as en_mod  # noqa: E402
from chinatravel.agent.tpc_agent_penguins import nl2sl_hybrid_zh as zh_mod  # noqa: E402


def _has_cjk(text):
    return any("一" <= ch <= "鿿" for ch in text)


ZH_QUERY = ("当前位置南京。我和儿子想去杭州玩2天，想住亲子房，"
            "逛逛大学校园，请帮我规划行程。")
EN_QUERY = ("[Current location: Nanjing, Destination: Hangzhou, Number of "
            "travelers: 2, Duration of travel: 2 days] I want to go to "
            "Hangzhou with my son for 2 days. I would like to stay in a "
            "family-friendly room and visit some university campuses.")
# an English query citing a Chinese term must stay on the en route
EN_WITH_CJK_TERM = ("Please book 机票 to Beijing; we need a hotel with a "
                    "gym and three days of sightseeing around the city "
                    "including the Palace Museum and a hot pot dinner.")
# a Chinese query citing latin tokens must stay on the zh route
ZH_WITH_LATIN = "当前位置上海。我想坐高铁G字头去北京玩3天，预算3000 RMB。"


def test_detection_basic():
    assert detect_query_lang(ZH_QUERY) == "zh"
    assert detect_query_lang(EN_QUERY) == "en"
    assert detect_query_lang(EN_WITH_CJK_TERM) == "en"
    assert detect_query_lang(ZH_WITH_LATIN) == "zh"


def test_detection_degenerate_falls_back_to_runner_lang():
    assert detect_query_lang("", runner_lang="zh") == "zh"
    assert detect_query_lang(None, runner_lang="zh") == "zh"
    assert detect_query_lang("12345 !!!", runner_lang="zh") == "zh"
    assert detect_query_lang("", runner_lang="en") == "en"
    assert detect_query_lang("") == "en"          # hard default
    assert detect_query_lang(None, runner_lang="fr") == "en"
    assert detect_query_lang(12345) == "en"       # non-string tolerated


def test_runner_lang_never_overrides_clear_text():
    # prompts must follow the query text, not the environment language
    assert detect_query_lang(ZH_QUERY, runner_lang="en") == "zh"
    assert detect_query_lang(EN_QUERY, runner_lang="zh") == "en"


def test_zh_route_selects_zh_prompt_stack():
    # step-1 instruction: Chinese rules (a)-(e) hardening
    p1 = zh_mod.NL2SL_INSTRUCTION.format(ZH_QUERY)
    assert "永远输出基础约束" in p1          # rule (a)
    assert "任选其一" in p1                  # rule (e), disjunction
    # step-2 DSL prompt: Chinese hard rules + canonical patterns
    assert "硬性规则" in zh_mod.sl_trans_prompt
    assert "析取" in zh_mod.sl_trans_prompt
    # zh disjunction reflect turns
    assert "编号分支" in zh_mod.disjunction_reflect_header_zh
    assert _has_cjk(zh_mod.disjunction_reflect_tail_zh)
    # zh example plans for the executability check
    assert sorted(zh_mod.EXAMPLE_PLANS.keys()) == [1, 2, 3, 4, 5, 6]


def test_en_route_selects_en_prompt_stack():
    p1 = en_mod.NL2SL_INSTRUCTION.format(EN_QUERY)
    assert p1  # instruction renders
    if os.environ.get("PENGUINS_PROMPT_LANG", "en").strip().lower() == "zh":
        # A/B switch active: the en stack deliberately carries zh-rendered
        # INSTRUCTION text (prompts_zh_instr) around the same English DSL
        assert _has_cjk(en_mod.sl_trans_prompt)
        assert _has_cjk(en_mod.disjunction_reflect_tail)
        return
    assert not _has_cjk(en_mod.sl_trans_prompt.split("###")[0])
    assert "HARD RULES" in en_mod.sl_trans_prompt
    assert "Colloquial idioms are HARD requirements" in en_mod.sl_trans_prompt
    assert "Re-emit ONLY the corrected disjunction" in en_mod.disjunction_reflect_tail


def test_both_routes_share_the_hardened_verifiers():
    # the zh path reuses the en module's language-agnostic normalizers and
    # disjunction verifier (single implementation, zh reflect prompt text)
    assert zh_mod.normalize_generated_constraints \
        is en_mod.normalize_generated_constraints
    assert zh_mod.enforce_disjunction is en_mod.enforce_disjunction
    # and both call the round-4 coverage verifier from the vendored module
    import inspect
    assert "enforce_coverage" in inspect.getsource(zh_mod.nl2sl_step3)
    assert "enforce_coverage" in inspect.getsource(en_mod.nl2sl_step3)


def test_v6_routing_decision_matches_router():
    # the planner routes on (nature_language, runner lang); reproduce the
    # decision for representative queries under both runner langs
    for runner in ("en", "zh"):
        assert detect_query_lang(ZH_QUERY, runner_lang=runner) == "zh"
        assert detect_query_lang(EN_QUERY, runner_lang=runner) == "en"
    # v6 wires the router into the live-translation branch
    import inspect
    from chinatravel.agent.tpc_agent_penguins import v6
    src = inspect.getsource(v6.UrbanTripOptimizedV6.run)
    assert "detect_query_lang" in src
    assert "nl2sl_hybrid_zh" in src and "nl2sl_hybrid_en" in src


def main():
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
