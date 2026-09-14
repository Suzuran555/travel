# TPCAgent development scripts

Run these scripts from the repository root after `conda activate chinatravel`.
The report implementation is `chinatravel/agent/tpc_agent_penguins/`.

## Offline segment indexes

`build_urbantrip_segments.py` builds ranking tables from the database.
`audit_segment_must_coverage.py` reads a query split and reports which required
POIs occur as endpoints in an intracity table. Their shared parsing helpers
live in `segment_must_pois.py`; they are not runtime dependencies.

Both scripts default to the same table location as the production
`SegmentIndex`: `chinatravel/agent/tpc_agent_penguins/data/segments/<lang>/`.
The report package ships English tables. A build overwrites the two JSONL files
in its output directory, so use a separate directory for development:

```bash
python scripts/build_urbantrip_segments.py --lang en --out /tmp/tpc-segment-rebuild
python scripts/audit_segment_must_coverage.py --lang en --only-uncovered
```

Use `--segment-path` to audit a different intracity table. The audit defaults to
`phase2_familiar.txt`; `--split` accepts another available split. The optional
builder flag `--must-from-split` reads oracle DSL annotations for an offline
coverage experiment. It is not part of inference, and rebuilding tables with a
different split does not reproduce the shipped report artifact.
When several local datasets contain the same query ID, the offline loader
prefers a copy with `hard_logic_py` annotations over an oracle-stripped copy.

## Held-out input simulation

`make_heldout_sim.py` copies familiarization queries while removing oracle
annotations, allowing the harness to exercise missing-oracle handling. See its
`--help` for source, split, and output paths. This simulates the input format;
the queries remain familiarization queries and are not the organizers' held-out
test set. Keep the original annotated split for official evaluation.

## Development serving proxies

`mock_sglang.py` forwards an SGLang-compatible HTTP interface to local Ollama.
`mock_sglang_openai.py` forwards it to an OpenAI-compatible upstream configured
through environment variables. They support local integration work; neither is
the formal SGLang serving stack used for the report's on-stack score. Results
obtained with these proxies must be identified as development results.

Unit scripts under `tests/` and `chinatravel/agent/tpc_agent_penguins/tests/`
provide regression sanity checks. Final validation requires a full relevant
testcase run followed by the unchanged official `eval_tpc.py`; keep each run's
result snapshot separately.
