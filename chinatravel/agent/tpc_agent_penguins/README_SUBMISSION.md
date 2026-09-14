# TPCAgent — final Phase-2 implementation

This directory is the canonical implementation described in
[the technical report](../../../tech_report/ijcai26_official/Antarctic%20penguins.tex).
The repository's normal entry point and crash-isolated harness both import it
directly. The planner, translation, and refinement code is preserved from
release commit `99f59f6`.

`tpc_agent.py` wraps the vendored `v6.py` planner. Qwen3.6-27B translates
requests into DSL; constraint stabilization, planning, and verified refinement
are deterministic. The refinement battery contains four feasibility repairs
and twelve subsequent soft-metric stages. The package also includes deadline
handling and a database-derived fallback.

Both translation languages, the prompts, and the intracity/intercity segment
tables under `data/segments/en/` remain together in this package.
The stock AST checkers, example plans, environment tools, databases, and
evaluators remain external dependencies in `chinatravel/`.

Use the commands in the [root README](../../../README.md).
The formal reproduction command is `bash scripts/validate_report.sh` from
the repository root, with Python 3.12 and a reachable Qwen3.6-27B service.
It runs 100 oracle-stripped queries and invokes the official evaluator only
after inference. The report's 97.70 familiarization score is not a guarantee
for a new run.

Runtime defaults are defined in `tpc_agent.py`,
`agent_env/scripts/tpc_agent_runner.py`, and `agent_env/config.toml.tpcagent`.
The wrapper budget is 290 seconds, with a 60-second refinement reserve and
an emission deadline near 275 seconds. The supervisor uses up to eight workers
and raises their outer per-query cap to 1,200 seconds; the shared soft budget
is 17,400 seconds. Budget configuration remains in source, without overrides
introduced by this cleanup.
