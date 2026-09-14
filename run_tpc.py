import argparse

import sys
import os
from func_timeout import func_timeout, FunctionTimedOut

project_root_path = os.path.dirname(os.path.abspath(__file__))
if project_root_path not in sys.path:
    sys.path.insert(0, project_root_path)

from chinatravel.data.load_datasets import load_query, save_json_file
from chinatravel.agent.load_model import init_agent, init_llm
from chinatravel.environment.world_env import WorldEnv


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run the report's Phase-2 TPCAgent.")
    parser.add_argument(
        "--splits",
        "-s",
        type=str,
        default="phase2_heldout_sim",
        help="query subset",
    )
    parser.add_argument("--index", "-id", type=str, default=None, help="query index")
    parser.add_argument(
        "--skip", "-sk", type=int, default=0, help="skip if the plan exists"
    )
    parser.add_argument(
        "--agent",
        "-a",
        type=str,
        default="TPCAgent",
        choices=["TPCAgent"],
    )
    parser.add_argument(
        "--llm",
        "-l",
        type=str,
        default="TPCLLM",
        choices=["TPCLLM"],
        help="Translation model client (TPCLLM uses the configured HTTP endpoint).",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="Result/cache directory name; defaults to agent and model display name.",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Outer timeout in seconds; the planner keeps the report package's budgets.",
    )
    parser.add_argument("--lang", "--locale", choices=["zh", "en"], default="en", help="Language environment to load.")

    # The submitted agent always generates its own constraints. The loader
    # removes reference fields too; TPCAgent strips all four again at entry.
    parser.set_defaults(oracle_translation=False)

    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.method is not None and (
        args.method in ("", ".", "..")
        or os.path.basename(args.method) != args.method
    ):
        parser.error("--method must be a directory name, not a path")

    print(args)

    query_index, query_data = load_query(args)
    print(len(query_index), "samples")

    if args.index is not None:
        query_index = [args.index]

    backbone_llm = init_llm(args.llm)
    method = args.method or args.agent + "_" + backbone_llm.name
    if args.method is None and args.lang == "en":
        method += "_en"

    cache_dir = os.path.join(project_root_path, "cache", method)

    res_dir = os.path.join(
        project_root_path, "results", method
    )
    log_dir = os.path.join(
        project_root_path, "cache", method
    )
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("res_dir: ", res_dir)
    print("log_dir:", log_dir)

    kwargs = {
        "method": args.agent,
        "env": WorldEnv(lang=args.lang),
        "backbone_llm": backbone_llm,
        "cache_dir": cache_dir,
        "log_dir": log_dir,
        "debug": False,
        "lang": args.lang,
    }
    agent = init_agent(kwargs)

    succ_count, eval_count = 0, 0

    for i, data_idx in enumerate(query_index):

        sys.stdout = sys.__stdout__
        print("------------------------------")
        print(
            "Process [{}/{}], Success [{}/{}]:".format(
                i, len(query_index), succ_count, eval_count
            )
        )
        print("data uid: ", data_idx)

        if args.skip and os.path.exists(os.path.join(res_dir, f"{data_idx}.json")):
            continue
        eval_count += 1
        query_i = query_data[data_idx]
        print(query_i)
        try:
            succ, plan = func_timeout(
                args.timeout,
                agent.run,
                args=(query_i,),
                kwargs=dict(
                    prob_idx=data_idx, oralce_translation=False
                ),
            )
        except FunctionTimedOut:
            succ, plan = 0, {"error": f"timeout after {args.timeout}s"}

        except Exception as e:
            succ, plan = 0, {"error": str(e)}

        if succ:
            succ_count += 1

        save_json_file(
            json_data=plan, file_path=os.path.join(res_dir, f"{data_idx}.json")
        )
