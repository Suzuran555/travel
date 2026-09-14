# Phase-2 runtime

The technical report uses `scripts/solve_script_with_harness.py` with
`config.toml.tpcagent`. It starts crash-isolated workers and calls the final
`chinatravel.agent.tpc_agent_penguins` package via `scripts/tpc_agent_runner.py`.

For fresh 100-query inference and official scoring, run from the repository root:

```bash
bash scripts/validate_report.sh
```

See the [root README](../README.md) for the Python environment, local datasets,
and model-service requirements. The wrapper selects the tracked report
configuration explicitly; a local ignored `config.toml` is optional.

`adapter.py`, `cli.py`, `http_server.py`, and `mcp_stdio.py` expose the
benchmark environment tools for debugging and external clients. The shared
harness retains those integration paths; the report run selects only
`--harness tpcagent`. Runtime outputs under `runs/` are ignored by Git.
