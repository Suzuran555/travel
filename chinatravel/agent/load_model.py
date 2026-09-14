"""Construct the report's Phase-2 agent and its translation model."""


def init_agent(kwargs):
    method = kwargs.get("method", "TPCAgent")
    if method != "TPCAgent":
        raise ValueError(f"Unsupported agent {method!r}; use TPCAgent.")

    from .tpc_agent_penguins.tpc_agent import TPCAgent

    return TPCAgent(**kwargs)


def init_llm(llm_name, max_model_len=None):
    if llm_name != "TPCLLM":
        raise ValueError(f"Unsupported translation model {llm_name!r}; use TPCLLM.")

    # Import before constructing a WorldEnv so the report's environment
    # patches are applied. Model inference runs in the configured HTTP server.
    from .tpc_agent_penguins.tpc_llm import TPCLLM

    return TPCLLM()
