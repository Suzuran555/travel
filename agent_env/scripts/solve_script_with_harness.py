#!/usr/bin/env python3
"""Solve ChinaTravel queries with a non-interactive agent harness and evaluate them."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "agent_env" / "config.toml"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HIDDEN_QUERY_KEYS = {"hard_logic", "hard_logic_py", "hard_logic_nl"}
OUTPUT_OPEN_TAG = "<output>"
OUTPUT_CLOSE_TAG = "</output>"


def default_tool_python() -> str:
    return "uv run python"


def normalize_lang(lang: Any) -> str:
    value = str(lang).lower()
    if value not in {"zh", "en"}:
        raise ValueError("ChinaTravel language must be 'zh' or 'en'.")
    return value


def load_queries(split: str, lang: str) -> tuple[list[str], dict[str, Any]]:
    from chinatravel.data.load_datasets import load_query

    args = argparse.Namespace(splits=split, oracle_translation=True, lang=lang)
    query_ids, query_data = load_query(args)
    if not query_ids:
        raise RuntimeError(f"No queries found for split: {split}")
    return query_ids, query_data


def load_one_query(
    split: str, uid: str | None, lang: str = "en"
) -> tuple[str, dict[str, Any]]:
    query_ids, query_data = load_queries(split, lang)
    query_uid = uid or query_ids[0]
    if query_uid not in query_data:
        raise RuntimeError(f"Query uid {query_uid!r} not found in split {split!r}")
    return query_uid, query_data[query_uid]


def visible_query(query: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in query.items() if key not in HIDDEN_QUERY_KEYS}


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        raise ValueError("Cannot pass null as a Codex -c value.")
    return str(value)


def output_source_name(harness: str) -> str:
    return "Codex" if harness == "codex" else "OpenCode"


def extract_tagged_output(text: str) -> str:
    match = re.search(
        r"<output>\s*(.*?)\s*</output>", text, flags=re.DOTALL | re.IGNORECASE
    )
    if not match:
        preview = text.strip()[:500].replace("\n", "\\n")
        raise ValueError(
            f"Agent output did not contain <output>...</output>. Preview: {preview!r}"
        )
    return match.group(1).strip()


def repair_json_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        stripped = stripped[start : end + 1]
    return re.sub(r",(\s*[}\]])", r"\1", stripped)


def extract_json_object(
    text: str, *, require_tags: bool = False, source_name: str = "Agent"
) -> dict[str, Any]:
    stripped = extract_tagged_output(text) if require_tags else text.strip()
    try:
        value = json.loads(repair_json_text(stripped))
    except json.JSONDecodeError:
        preview = stripped[:500].replace("\n", "\\n")
        raise ValueError(
            f"{source_name} output did not contain parseable JSON. Preview: {preview!r}"
        ) from None
    if not isinstance(value, dict):
        raise ValueError(f"{source_name} output was valid JSON but not a JSON object.")
    return value


def load_plan_from_output_txt(
    output_txt_path: Path, require_tags: bool, source_name: str
) -> dict[str, Any]:
    return extract_json_object(
        output_txt_path.read_text(encoding="utf-8"),
        require_tags=require_tags,
        source_name=source_name,
    )


def build_prompt(
    split: str,
    uid: str,
    query: dict[str, Any],
    tool_python: str,
    lang: str,
) -> str:
    query_text = json.dumps(query, ensure_ascii=False, indent=2)
    cli_python = f"{tool_python} -m agent_env.cli --lang {lang}"
    return f"""You are solving one ChinaTravel benchmark query.

Work quietly. Do not narrate your plan, progress, or calculations. Use CLI calls only
to gather exact facts. Keep the itinerary relaxed and simple. After gathering enough
facts, immediately return the final JSON object between {OUTPUT_OPEN_TAG} and {OUTPUT_CLOSE_TAG}.

Use the local CLI API to inspect the environment. The repository root is:
{PROJECT_ROOT}

Run CLI commands from that repository root using this Python command:
{tool_python}

Use this command for every `agent_env.cli` call. Do not use bare `python`.

Important commands:
- {cli_python} tools
- {cli_python} call attractions_keys '{{"city":"<target_city>"}}'
- {cli_python} call restaurants_keys '{{"city":"<target_city>"}}'
- {cli_python} call accommodations_keys '{{"city":"<target_city>"}}'
- {cli_python} call intercity_transport_select '{{"start_city":"<start_city>","end_city":"<target_city>","intercity_type":"train","earliest_leave_time":"07:00"}}'
- {cli_python} call goto '{{"city":"<target_city>","start":"<exact start>","end":"<exact end>","start_time":"HH:MM","transport_type":"metro"}}'

Use exact names, prices, costs, times, distances, TrainID/FlightID values, and transport
segments returned by the CLI. Do not invent environment facts.

Output the JSON result between these tags:
{OUTPUT_OPEN_TAG}
{{...valid JSON matching chinatravel/evaluation/output_schema.json...}}
{OUTPUT_CLOSE_TAG}

Split: {split}
UID: {uid}
Visible query:
{query_text}
"""


def provider_model_name(model: str, provider_id: str) -> str:
    prefix = f"{provider_id}/"
    if model.startswith(prefix):
        return model[len(prefix) :]
    if "/" in model:
        return model.rsplit("/", 1)[1]
    return model


def build_opencode_config(opencode_config: dict[str, Any], model: str) -> dict[str, Any]:
    provider = opencode_config.get("provider", {})
    if not isinstance(provider, dict):
        raise ValueError("Config section [opencode.provider] must be a table.")

    provider_id = str(provider.get("id") or "dashscope")
    model_id = provider_model_name(model, provider_id)
    api_key_env = str(opencode_config.get("api_key_env") or "DASHSCOPE_API_KEY")

    provider_options: dict[str, Any] = {
        "baseURL": str(provider.get("base_url") or provider.get("baseURL") or ""),
        "apiKey": f"{{env:{api_key_env}}}",
    }
    extra_options = provider.get("options", {})
    if isinstance(extra_options, dict):
        provider_options.update(extra_options)
    if not provider_options["baseURL"]:
        raise ValueError("[opencode.provider].base_url is required.")

    model_config: dict[str, Any] = {
        "name": str(provider.get("model_name") or model_id),
        "tool_call": bool(provider.get("tool_call", True)),
    }
    model_options = provider.get("model_options", {})
    if isinstance(model_options, dict):
        model_config["options"] = model_options

    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            provider_id: {
                "npm": str(provider.get("npm") or "@ai-sdk/openai-compatible"),
                "name": str(provider.get("name") or provider_id),
                "options": provider_options,
                "models": {model_id: model_config},
            }
        },
        "model": model if "/" in model else f"{provider_id}/{model}",
    }


def apply_opencode_env(env: dict[str, str], opencode_config: dict[str, Any]) -> None:
    api_key = opencode_config.get("api_key")
    api_key_env = str(opencode_config.get("api_key_env") or "DASHSCOPE_API_KEY")
    if api_key:
        env[api_key_env] = str(api_key)


def extract_opencode_text(stdout: str) -> str:
    text_parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "text":
            continue
        part = event.get("part", {})
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    if not text_parts and stdout.strip():
        return stdout.strip()
    return "".join(text_parts).strip()


def extract_opencode_error(stdout: str) -> str | None:
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "error":
            continue
        error = event.get("error")
        if isinstance(error, dict):
            data = error.get("data")
            if isinstance(data, dict) and data.get("message"):
                return str(data["message"])
            if error.get("message"):
                return str(error["message"])
            return json.dumps(error, ensure_ascii=False, default=json_default)
        if error:
            return str(error)
        return "OpenCode emitted an error event."
    return None


def run_opencode(
    prompt: str,
    run_dir: Path,
    output_txt_path: Path,
    model: str | None,
    timeout: int,
    opencode_config: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not model:
        raise ValueError("[opencode].model is required.")

    opencode_json_path = run_dir / "opencode.json"
    write_json(opencode_json_path, build_opencode_config(opencode_config, str(model)))

    cmd = [
        "opencode",
        "run",
        "--dir",
        str(PROJECT_ROOT),
        "--model",
        str(model),
        "--format",
        "json",
    ]
    if bool(opencode_config.get("dangerously_skip_permissions", True)):
        cmd.append("--dangerously-skip-permissions")
    agent = opencode_config.get("agent")
    if agent:
        cmd.extend(["--agent", str(agent)])
    variant = opencode_config.get("variant")
    if variant:
        cmd.extend(["--variant", str(variant)])
    cmd.append(prompt)

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["OPENCODE_CONFIG"] = str(opencode_json_path)
    apply_opencode_env(env, opencode_config)
    completed = subprocess.run(
        cmd,
        cwd=run_dir,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    opencode_error = extract_opencode_error(completed.stdout)
    if opencode_error:
        stderr = completed.stderr
        if stderr:
            stderr = f"{stderr.rstrip()}\n{opencode_error}\n"
        else:
            stderr = f"{opencode_error}\n"
        return subprocess.CompletedProcess(
            completed.args, completed.returncode or 1, completed.stdout, stderr
        )
    if completed.returncode == 0:
        output_txt_path.write_text(
            extract_opencode_text(completed.stdout), encoding="utf-8"
        )
    return completed


def apply_codex_env(env: dict[str, str], codex_config: dict[str, Any]) -> None:
    api_key = codex_config.get("api_key")
    api_key_env = codex_config.get("api_key_env")
    provider = codex_config.get("provider", {})
    if not isinstance(provider, dict):
        provider = {}
    env_key = str(provider.get("env_key") or api_key_env or "OPENAI_API_KEY")
    if api_key:
        env[env_key] = str(api_key)


def build_codex_config_overrides(codex_config: dict[str, Any]) -> list[tuple[str, Any]]:
    overrides: list[tuple[str, Any]] = []

    provider = codex_config.get("provider", {})
    if isinstance(provider, dict) and provider:
        provider_id = provider.get("id")
        if provider_id:
            provider_id = str(provider_id)
            overrides.append(("model_provider", provider_id))
            provider_keys = {
                "name": provider.get("name"),
                "base_url": provider.get("base_url"),
                "env_key": provider.get("env_key") or codex_config.get("api_key_env"),
                "wire_api": provider.get("wire_api"),
            }
            for key, value in provider_keys.items():
                if value is not None:
                    overrides.append((f"model_providers.{provider_id}.{key}", value))

    extra_config = codex_config.get("extra_config", {})
    if isinstance(extra_config, dict):
        for key, value in extra_config.items():
            overrides.append((str(key), value))

    return overrides


def run_codex(
    prompt: str,
    run_dir: Path,
    output_txt_path: Path,
    model: str | None,
    timeout: int,
    codex_config: dict[str, Any],
) -> subprocess.CompletedProcess[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "codex",
        "exec",
        "--cd",
        str(run_dir),
        "--add-dir",
        str(PROJECT_ROOT),
        "--sandbox",
        str(codex_config.get("sandbox") or "workspace-write"),
        "--output-last-message",
        str(output_txt_path),
    ]
    output_schema = bool(codex_config.get("output_schema", False))
    if output_schema:
        cmd.extend(
            [
                "--output-schema",
                str(PROJECT_ROOT / "chinatravel" / "evaluation" / "output_schema.json"),
            ]
        )
    if model:
        cmd.extend(["--model", model])
    for key, value in build_codex_config_overrides(codex_config):
        cmd.extend(["--config", f"{key}={toml_value(value)}"])
    cmd.append("-")

    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    apply_codex_env(env, codex_config)
    return subprocess.run(
        cmd,
        cwd=run_dir,
        env=env,
        text=True,
        input=prompt,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def evaluate_one(
    split: str,
    uid: str,
    query: dict[str, Any],
    plan: dict[str, Any],
    lang: str,
) -> dict[str, Any]:
    from chinatravel.evaluation.commonsense_constraint import (
        evaluate_commonsense_constraints,
    )
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
    from chinatravel.evaluation.utils import load_json_file

    query_index = [uid]
    query_data = {uid: query}
    result_data = {uid: plan}
    schema = load_json_file(
        str(PROJECT_ROOT / "chinatravel" / "evaluation" / "output_schema.json")
    )

    schema_rate, schema_result_agg, schema_pass_id = evaluate_schema_constraints(
        query_index, result_data, schema=schema
    )
    macro_comm, micro_comm, common_result_agg, commonsense_pass_id = (
        evaluate_commonsense_constraints(
            query_index, query_data, result_data, verbose=False, lang=lang
        )
    )
    (
        macro_logi,
        micro_logi,
        conditional_macro_logi,
        conditional_micro_logi,
        logi_result_agg,
        logi_pass_id,
    ) = evaluate_hard_constraints_v2(
        query_index,
        query_data,
        result_data,
        env_pass_id=commonsense_pass_id,
        verbose=False,
        lang=lang,
    )

    return {
        "split": split,
        "uid": uid,
        "schema_pass": uid in schema_pass_id,
        "commonsense_pass": uid in commonsense_pass_id,
        "logical_pass": uid in logi_pass_id,
        "all_pass": uid
        in set(schema_pass_id) & set(commonsense_pass_id) & set(logi_pass_id),
        "schema_rate": schema_rate,
        "commonsense_micro": micro_comm,
        "commonsense_macro": macro_comm,
        "logical_micro": micro_logi,
        "logical_macro": macro_logi,
        "conditional_logical_micro": conditional_micro_logi,
        "conditional_logical_macro": conditional_macro_logi,
        "schema_details": schema_result_agg.to_dict(orient="records"),
        "commonsense_details": common_result_agg.to_dict(orient="records"),
        "logical_details": logi_result_agg.to_dict(orient="records"),
    }


def evaluate_split(
    split: str,
    method: str,
    query_ids: list[str],
    query_data: dict[str, Any],
    lang: str,
    parse_failure_ids: list[str] | None = None,
) -> dict[str, Any]:
    from chinatravel.evaluation.commonsense_constraint import (
        evaluate_commonsense_constraints,
    )
    from chinatravel.evaluation.hard_constraint import evaluate_hard_constraints_v2
    from chinatravel.evaluation.schema_constraint import evaluate_schema_constraints
    from chinatravel.evaluation.utils import load_json_file

    result_dir = PROJECT_ROOT / "results" / method
    parse_failure_ids = parse_failure_ids or []
    parse_failure_set = set(parse_failure_ids)
    result_data = {}
    evaluated_ids = []
    for uid in query_ids:
        if uid in parse_failure_set:
            continue
        result_path = result_dir / f"{uid}.json"
        if result_path.exists():
            result_data[uid] = read_json(result_path)
            evaluated_ids.append(uid)

    if evaluated_ids:
        schema = load_json_file(
            str(PROJECT_ROOT / "chinatravel" / "evaluation" / "output_schema.json")
        )
        schema_rate, schema_result_agg, schema_pass_id = evaluate_schema_constraints(
            evaluated_ids, result_data, schema=schema
        )
        macro_comm, micro_comm, common_result_agg, commonsense_pass_id = (
            evaluate_commonsense_constraints(
                evaluated_ids, query_data, result_data, verbose=False, lang=lang
            )
        )
        (
            macro_logi,
            micro_logi,
            conditional_macro_logi,
            conditional_micro_logi,
            logi_result_agg,
            logi_pass_id,
        ) = evaluate_hard_constraints_v2(
            evaluated_ids,
            query_data,
            result_data,
            env_pass_id=commonsense_pass_id,
            verbose=False,
            lang=lang,
        )
        schema_details = schema_result_agg.to_dict(orient="records")
        commonsense_details = common_result_agg.to_dict(orient="records")
        logical_details = logi_result_agg.to_dict(orient="records")
    else:
        schema_rate = macro_comm = micro_comm = 0
        macro_logi = micro_logi = conditional_macro_logi = conditional_micro_logi = 0
        schema_pass_id = []
        commonsense_pass_id = []
        logi_pass_id = []
        schema_details = []
        commonsense_details = []
        logical_details = []

    for failed_uid in parse_failure_ids:
        schema_details.append(
            {"data_id": failed_uid, "schema": 0, "parse_failure": True}
        )
        commonsense_details.append({"data_id": failed_uid, "parse_failure": True})
        logical_details.append({"data_id": failed_uid, "parse_failure": True})

    all_pass = set(schema_pass_id) & set(commonsense_pass_id) & set(logi_pass_id)
    parsed_count = len(evaluated_ids)
    parse_failure_count = len(parse_failure_ids)
    evaluated_count = parsed_count + parse_failure_count
    pass_rate_scale = parsed_count / evaluated_count if evaluated_count else 0
    return {
        "split": split,
        "method": method,
        "evaluated": evaluated_count,
        "parsed_evaluated": parsed_count,
        "parse_failures": parse_failure_count,
        "schema_pass_count": len(schema_pass_id),
        "commonsense_pass_count": len(commonsense_pass_id),
        "logical_pass_count": len(logi_pass_id),
        "all_pass_count": len(all_pass),
        "all_pass_ids": sorted(all_pass),
        "schema_rate": schema_rate * pass_rate_scale,
        "commonsense_micro": micro_comm * pass_rate_scale,
        "commonsense_macro": macro_comm * pass_rate_scale,
        "logical_micro": micro_logi * pass_rate_scale,
        "logical_macro": macro_logi * pass_rate_scale,
        "conditional_logical_micro": conditional_micro_logi * pass_rate_scale,
        "conditional_logical_macro": conditional_macro_logi * pass_rate_scale,
        "schema_details": schema_details,
        "commonsense_details": commonsense_details,
        "logical_details": logical_details,
    }


def failed_evaluation(split: str, uid: str, reason: str) -> dict[str, Any]:
    return {
        "split": split,
        "uid": uid,
        "schema_pass": False,
        "commonsense_pass": False,
        "logical_pass": False,
        "all_pass": False,
        "schema_rate": 0,
        "commonsense_micro": 0,
        "commonsense_macro": 0,
        "logical_micro": 0,
        "logical_macro": 0,
        "conditional_logical_micro": 0,
        "conditional_logical_macro": 0,
        "schema_details": [{"data_id": uid, "schema": 0}],
        "commonsense_details": [{"data_id": uid}],
        "logical_details": [{"data_id": uid}],
        "parse_failure": True,
        "failure_reason": reason,
    }


def config_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return section


def choose(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def choose_nonempty(cli_value: Any, config_value: Any, default: Any) -> Any:
    value = choose(cli_value, config_value, default)
    if isinstance(value, str) and not value.strip():
        return default
    return value


def method_model_name(model: str | None) -> str:
    if not model:
        return "model"
    return provider_model_name(str(model), str(model).split("/", 1)[0])


def safe_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "unnamed"


def default_method(model: str | None, split: str, harness: str) -> str:
    return "-".join(
        [
            safe_path_component(method_model_name(model)),
            safe_path_component(split),
            safe_path_component(harness),
        ]
    )


def solve_query(
    *,
    harness: str,
    split: str,
    lang: str,
    uid: str,
    query: dict[str, Any],
    method: str,
    model: str | None,
    timeout: int,
    work_dir: str,
    harness_config: dict[str, Any],
    no_run_harness: bool,
    plan_file: str | None,
    tool_python: str,
    require_output_tags: bool,
) -> dict[str, Any] | None:
    public_query = visible_query(query)

    run_dir = PROJECT_ROOT / work_dir / f"{split}_{uid}"
    prompt_path = run_dir / "prompt.txt"
    output_txt_path = run_dir / "output.txt"
    output_json_path = run_dir / "output.json"
    eval_path = run_dir / "evaluation.json"
    result_path = PROJECT_ROOT / "results" / method / f"{uid}.json"

    prompt = build_prompt(split, uid, public_query, tool_python, lang)
    run_dir.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(prompt, encoding="utf-8")

    print(f"Loaded query: split={split} uid={uid}")
    print(f"Prompt: {prompt_path}")

    if plan_file:
        plan = read_json(Path(plan_file))
    elif no_run_harness:
        print(f"Skipping {output_source_name(harness)} call because --no-run-harness was set.")
        return None
    elif harness == "tpcagent":
        # Custom in-process agent: run the team's deterministic NeSy planner.
        from agent_env.scripts.tpc_agent_runner import run_tpc_agent

        plan = run_tpc_agent(
            uid=uid,
            query=public_query,
            lang=lang,
            tpcagent_config=harness_config,
            timeout=timeout,
            cache_dir=str(PROJECT_ROOT / "cache" / method),
            log_dir=str(run_dir),
        )
    else:
        if harness == "opencode":
            completed = run_opencode(
                prompt=prompt,
                run_dir=run_dir,
                output_txt_path=output_txt_path,
                model=model,
                timeout=timeout,
                opencode_config=harness_config,
            )
            stdout_path = run_dir / "opencode_stdout.jsonl"
            stderr_path = run_dir / "opencode_stderr.txt"
            failure_label = "opencode run"
        elif harness == "codex":
            completed = run_codex(
                prompt=prompt,
                run_dir=run_dir,
                output_txt_path=output_txt_path,
                model=model,
                timeout=timeout,
                codex_config=harness_config,
            )
            stdout_path = run_dir / "codex_stdout.txt"
            stderr_path = run_dir / "codex_stderr.txt"
            failure_label = "codex exec"
        else:
            raise ValueError(f"Unsupported harness: {harness}")

        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{failure_label} failed with exit code {completed.returncode}. "
                f"See {stdout_path} and {stderr_path}."
            )
        try:
            plan = load_plan_from_output_txt(
                output_txt_path,
                require_tags=require_output_tags,
                source_name=output_source_name(harness),
            )
        except ValueError as exc:
            (run_dir / "parse_error.txt").write_text(str(exc), encoding="utf-8")
            evaluation = failed_evaluation(split, uid, str(exc))
            write_json(eval_path, evaluation)
            print(
                f"{output_source_name(harness)} finished but did not produce a JSON plan. "
                f"See {output_txt_path}, {stdout_path}, {stderr_path}, and "
                f"{run_dir / 'parse_error.txt'}."
            )
            print(f"Marked parse failure as completely wrong: {eval_path}")
            print(
                json.dumps(
                    evaluation, ensure_ascii=False, indent=2, default=json_default
                )
            )
            return evaluation

    write_json(output_json_path, plan)
    write_json(result_path, plan)
    print(f"Saved plan: {result_path}")

    # Formal held-out queries carry NO oracle fields; the internal evaluator
    # would KeyError('hard_logic_py'). The result file above is the deliverable
    # (organizers score separately), so never let internal scoring abort the run.
    try:
        evaluation = evaluate_one(split, uid, query, plan, lang)
    except Exception as exc:
        print(f"[eval] internal evaluate_one skipped ({type(exc).__name__}: {exc})")
        evaluation = {"uid": uid, "split": split, "method": method, "eval_skipped": True}
    write_json(eval_path, evaluation)
    print(f"Saved evaluation: {eval_path}")
    print(json.dumps(evaluation, ensure_ascii=False, indent=2, default=json_default))
    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a ChinaTravel split, solve it with a harness, and evaluate the output."
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG_PATH), help="TOML config path."
    )
    parser.add_argument(
        "--split", default=None, help="ChinaTravel split name. Overrides [run].split."
    )
    parser.add_argument(
        "--lang",
        choices=["zh", "en"],
        default=None,
        help="Query and sandbox language. Overrides [run].lang.",
    )
    parser.add_argument(
        "--uid", default=None, help="Optional query UID. Overrides [run].uid."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit split queries. Overrides [run].limit.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Optional result/eval method override. Default: <model>-<split>-<harness>.",
    )
    parser.add_argument(
        "--harness",
        choices=["opencode", "codex", "tpcagent"],
        default=None,
        help="Harness to run. Overrides [run].harness.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the selected harness.",
    )
    parser.add_argument(
        "--opencode-model",
        default=None,
        help="Optional model passed to opencode run. Overrides [opencode].model.",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        help="Optional model passed to codex exec. Overrides [codex].model.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="OpenCode subprocess timeout in seconds.",
    )
    parser.add_argument(
        "--no-run-harness",
        action="store_true",
        default=None,
        help="Load the query and write the prompt, but do not call the harness.",
    )
    parser.add_argument(
        "--run-harness",
        action="store_false",
        dest="no_run_harness",
        help="Force the harness on when [run].no_run_harness is true.",
    )
    parser.add_argument(
        "--no-run-opencode",
        action="store_true",
        default=None,
        help="Deprecated alias for --no-run-harness.",
    )
    parser.add_argument(
        "--run-opencode",
        action="store_false",
        dest="no_run_opencode",
        help="Deprecated alias for --run-harness.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Skip queries that already have result JSON files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not skip existing result JSON files.",
    )
    parser.add_argument(
        "--plan-file",
        default=None,
        help="Evaluate an existing plan JSON. Overrides [run].plan_file.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for prompts, raw OpenCode output, and evaluation JSON.",
    )
    parser.add_argument(
        "--tool-python",
        default=None,
        help="Python command the harness should use for agent_env.cli. Overrides [run].tool_python.",
    )
    parser.add_argument(
        "--no-require-output-tags",
        action="store_true",
        default=None,
        help="Parse raw JSON without requiring <output> tags.",
    )
    parser.add_argument(
        "--_worker",
        dest="worker_mode",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,  # internal: run one crash-isolated worker
    )
    parser.add_argument(
        "--shard",
        nargs=2,
        type=int,
        metavar=("K", "N"),
        default=None,
        help=argparse.SUPPRESS,  # internal: process queries with index %% N == K
    )
    return parser.parse_args()


# --- crash-isolation supervisor ------------------------------------------
# The organizer automation runs this script once; a single interpreter-level
# crash (segfault in a C extension, OOM kill, ...) partway through the split
# would otherwise lose the whole run. The default entry point therefore acts
# as a SUPERVISOR: it spawns crash-isolated worker copies of itself (one per
# shard, sharing the work via --resume idempotency), restarts any worker that
# dies, and force-fallbacks a query that repeatedly kills its worker.

def _crashstate_path(shard_idx):
    return PROJECT_ROOT / "agent_env" / "runs" / f"_crashstate_{shard_idx}.json"


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


def _write_json(path, obj):
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj))
    except Exception:
        pass


def _supervise() -> int:
    import time as _time

    argv = list(sys.argv[1:])
    start = _time.time()
    env = dict(os.environ)
    # workers must measure the GLOBAL budget from the ORIGINAL start, so a
    # restarted worker never gets a fresh clock
    env.setdefault("PENGUINS_RUN_START_EPOCH", str(int(start)))
    workers = int(env.get("PENGUINS_WORKERS", "0") or 0)
    if workers <= 0:
        cpus = os.cpu_count() or 4
        workers = max(1, min(8, cpus // 2))
    single_uid = "--uid" in argv
    if single_uid:
        workers = 1
    if workers > 1:
        # sharded workers get a wider per-query cap: each shard holds ~1/N of
        # the split, so the per-query share of the global budget grows ~N-fold
        env.setdefault("PENGUINS_PER_QUERY_CAP", "900")
    script = str(Path(__file__).resolve())
    wall_guard = start + 18600  # never restart past ~5h10m
    hard_deadline = start + 21000  # absolute give-up on stragglers (~5h50m)

    def spawn(shard_idx):
        cmd = [sys.executable, "-u", script] + argv + ["--_worker", "--resume"]
        if workers > 1:
            cmd += ["--shard", str(shard_idx), str(workers)]
        wenv = dict(env)
        wenv["PENGUINS_SHARD_INDEX"] = str(shard_idx)
        # keep N workers from oversubscribing the CPU via BLAS thread pools
        wenv.setdefault("OMP_NUM_THREADS", "2")
        wenv.setdefault("OPENBLAS_NUM_THREADS", "2")
        return subprocess.Popen(cmd, env=wenv)

    for i in range(workers):
        _write_json(_crashstate_path(i), {"last_uid": None, "counts": {}})
    procs = {i: spawn(i) for i in range(workers)}
    restarts = {i: 0 for i in range(workers)}
    noprog = {i: 0 for i in range(workers)}
    last_marker = {i: None for i in range(workers)}
    done_rc = {}
    print(f"[supervisor] {workers} worker(s), crash-isolated, resume-sharing")

    while procs:
        _time.sleep(3)
        if _time.time() > hard_deadline:
            print("[supervisor] hard deadline — terminating remaining workers")
            for p in procs.values():
                p.terminate()
            for i in list(procs):
                done_rc[i] = 1
            break
        for i, p in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            if rc == 0:
                done_rc[i] = 0
                del procs[i]
                continue
            cs = _read_json(_crashstate_path(i), {"last_uid": None, "counts": {}})
            marker = cs.get("last_uid")
            if marker:
                cs.setdefault("counts", {})[marker] = cs["counts"].get(marker, 0) + 1
                _write_json(_crashstate_path(i), cs)
            progressed = marker is not None and marker != last_marker[i]
            noprog[i] = 0 if progressed else noprog[i] + 1
            last_marker[i] = marker
            restarts[i] += 1
            if marker is None and restarts[i] >= 2:
                print(f"[supervisor] worker {i} failed twice before any query (rc={rc}) — giving up on it")
                done_rc[i] = rc
                del procs[i]
                continue
            if noprog[i] >= 4 or restarts[i] > 40 or _time.time() > wall_guard:
                print(f"[supervisor] worker {i}: restarts={restarts[i]} no-progress={noprog[i]} rc={rc} — giving up on it")
                done_rc[i] = rc
                del procs[i]
                continue
            print(f"[supervisor] worker {i} died rc={rc} at uid={marker}; restart #{restarts[i]}")
            procs[i] = spawn(i)

    if any(rc != 0 for rc in done_rc.values()) and _time.time() < hard_deadline:
        # mop-up pass: one serial worker sweeps every remaining query
        # (resume skips finished ones; exhausted budget emits fast fallbacks)
        print("[supervisor] mop-up pass for queries lost to failed workers")
        _write_json(_crashstate_path("mop"), {"last_uid": None, "counts": {}})
        wenv = dict(env)
        wenv["PENGUINS_SHARD_INDEX"] = "mop"
        cmd = [sys.executable, "-u", script] + argv + ["--_worker", "--resume"]
        rc = subprocess.call(cmd, env=wenv)
        print(f"[supervisor] mop-up exit {rc}")
        return rc
    return 0 if all(rc == 0 for rc in done_rc.values()) else max(done_rc.values())


def main() -> None:
    args = parse_args()
    config = read_config(Path(args.config))
    run_config = config_section(config, "run")
    harness = str(choose(args.harness, run_config.get("harness"), "opencode"))
    if harness not in {"opencode", "codex", "tpcagent"}:
        raise ValueError("Harness must be one of: opencode, codex, tpcagent.")
    harness_config = config_section(config, harness)
    opencode_config = config_section(config, "opencode")
    codex_config = config_section(config, "codex")

    split = str(choose(args.split, run_config.get("split"), "easy"))
    lang = normalize_lang(choose(args.lang, run_config.get("lang"), "en"))
    uid = choose(args.uid, run_config.get("uid"), None)
    limit = choose(args.limit, run_config.get("limit"), None)
    if harness == "opencode":
        harness_model_arg = choose(args.model, args.opencode_model, None)
        selected_config = opencode_config
    elif harness == "tpcagent":
        harness_model_arg = choose(args.model, None, None)
        selected_config = harness_config
    else:
        harness_model_arg = choose(args.model, args.codex_model, None)
        selected_config = codex_config
    model = choose(harness_model_arg, selected_config.get("model"), None)
    method = str(
        choose_nonempty(
            args.method,
            run_config.get("method"),
            default_method(model, split, harness),
        )
    )
    timeout = int(choose(args.timeout, selected_config.get("timeout"), 900))
    work_dir = str(
        choose_nonempty(
            args.work_dir, run_config.get("work_dir"), f"agent_env/runs/{method}"
        )
    )
    tool_python = str(
        choose(args.tool_python, run_config.get("tool_python"), default_tool_python())
    )
    no_run_alias = choose(args.no_run_harness, args.no_run_opencode, None)
    no_run_harness = bool(
        choose(
            no_run_alias,
            choose(run_config.get("no_run_harness"), run_config.get("no_run_opencode"), None),
            False,
        )
    )
    resume = bool(choose(args.resume, run_config.get("resume"), False))
    plan_file = choose(args.plan_file, run_config.get("plan_file"), None)
    require_output_tags = bool(
        choose(
            (
                None
                if args.no_require_output_tags is None
                else not args.no_require_output_tags
            ),
            selected_config.get("require_output_tags"),
            True,
        )
    )

    query_ids, query_data = load_queries(split, lang)
    if uid:
        uid = str(uid)
        if uid not in query_data:
            raise RuntimeError(f"Query uid {uid!r} not found in split {split!r}")
        selected_ids = [uid]
    else:
        selected_ids = query_ids
        if limit is not None:
            limit = int(limit)
            if limit >= 1:
                selected_ids = selected_ids[:limit]

    if args.shard is not None:
        shard_k, shard_n = int(args.shard[0]), int(args.shard[1])
        selected_ids = [u for i, u in enumerate(selected_ids) if i % shard_n == shard_k]
        print(f"Shard {shard_k}/{shard_n}: {len(selected_ids)} queries")

    print(f"Config: {Path(args.config)}")
    print(
        f"Run: split={split} lang={lang} queries={len(selected_ids)} harness={harness} method={method} "
        f"model={model or '<config default>'} resume={resume}"
    )

    shard_idx = os.environ.get("PENGUINS_SHARD_INDEX")
    crash_counts = {}
    if shard_idx is not None:
        crash_counts = _read_json(
            _crashstate_path(shard_idx), {"last_uid": None, "counts": {}}
        ).get("counts", {})

    evaluations = []
    skipped_completed = 0
    for index, uid in enumerate(selected_ids, start=1):
        print(f"\n[{index}/{len(selected_ids)}] Solving {uid}")
        result_path = PROJECT_ROOT / "results" / method / f"{uid}.json"
        if resume and result_path.exists():
            skipped_completed += 1
            print(f"Skipping completed query because result exists: {result_path}")
            continue
        if shard_idx is not None:
            # crash accounting: mark the query in flight so the supervisor can
            # attribute a worker death, and force a deterministic fallback for
            # any query that has already killed this worker twice
            _write_json(
                _crashstate_path(shard_idx),
                {"last_uid": uid, "counts": crash_counts},
            )
            if int(crash_counts.get(uid, 0)) >= 2:
                os.environ["PENGUINS_FORCE_FALLBACK_UID"] = str(uid)
                print(f"crash-guard: {uid} killed the worker twice — deterministic fallback")
            else:
                os.environ.pop("PENGUINS_FORCE_FALLBACK_UID", None)
        evaluation = solve_query(
            harness=harness,
            split=split,
            lang=lang,
            uid=uid,
            query=query_data[uid],
            method=method,
            model=model,
            timeout=timeout,
            work_dir=work_dir,
            harness_config=harness_config,
            no_run_harness=no_run_harness,
            plan_file=plan_file,
            tool_python=tool_python,
            require_output_tags=require_output_tags,
        )
        if evaluation is not None:
            evaluations.append(evaluation)

    parse_failures = sum(
        1 for evaluation in evaluations if evaluation.get("parse_failure")
    )
    parse_failure_ids = [
        evaluation["uid"]
        for evaluation in evaluations
        if evaluation.get("parse_failure")
    ]
    if (evaluations or skipped_completed) and not args.uid:
        summary_path = PROJECT_ROOT / work_dir / f"{split}_summary.json"
        # Formal held-out data carries NO oracle fields (hard_logic_py etc.),
        # so the aggregate evaluator raises KeyError. Every result file is
        # already written above and is the actual deliverable -- internal
        # scoring must never fail the run (a non-zero exit could be read as
        # a failed submission by the organizers' automation).
        try:
            summary = evaluate_split(
                split,
                method,
                selected_ids,
                query_data,
                lang,
                parse_failure_ids=parse_failure_ids,
            )
            summary["skipped_completed"] = skipped_completed
            write_json(summary_path, summary)
            print(f"\nSaved split summary: {summary_path}")
            print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
        except Exception as exc:
            print(f"\nSplit summary skipped (oracle-less data?): {exc!r}")

    print(f"\nParse failures: {parse_failures}")
    if resume:
        print(f"Skipped completed queries: {skipped_completed}")


if __name__ == "__main__":
    if "--_worker" in sys.argv:
        try:
            main()
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(1)
    else:
        raise SystemExit(_supervise())
