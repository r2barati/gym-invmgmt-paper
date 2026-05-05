"""
llm_utils.py — Shared utilities for LLM-based inventory agents.

Provides:
  - Model loading (local GGUF with auto-download, OpenAI API)
  - Unified query dispatch (chat completion with JSON mode)
  - Robust JSON order parsing with failure tracking
  - Prompt formatting helpers
"""

import re
import json
import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ── Model Loading ──────────────────────────────────────────────────

_cached_llm = None  # Module-level cache to avoid reloading per agent


def get_models_dir() -> str:
    """Return the project's data/models directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    models_dir = os.path.join(project_root, 'data', 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def load_local_model(
    repo_id: str = "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename: str = "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    n_ctx: int = 4096,
):
    """
    Lazy-load a local GGUF model. Downloads from HuggingFace Hub if not found.

    Returns a llama_cpp.Llama instance.
    """
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm

    from llama_cpp import Llama

    models_dir = get_models_dir()
    local_path = os.path.join(models_dir, filename)

    if not os.path.exists(local_path):
        logger.info("GGUF model not found locally. Downloading from HuggingFace Hub...")
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=models_dir,
            )
        except ImportError:
            raise ImportError(
                f"GGUF model not found at {local_path} and `huggingface_hub` "
                f"is not installed. Either:\n"
                f"  1. Install: pip install huggingface_hub\n"
                f"  2. Manually download {filename} from {repo_id} to {models_dir}"
            )

    _cached_llm = Llama(model_path=local_path, n_ctx=n_ctx, verbose=False)
    logger.info("Local GGUF model loaded: %s", local_path)
    return _cached_llm


# ── Query Dispatch ─────────────────────────────────────────────────

from typing import Optional

def query_chat(
    messages: list[dict],
    backend: str = 'local',
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """
    Send a chat-completion request to the configured backend.

    Parameters
    ----------
    messages : list of {"role": str, "content": str}
    backend : 'local' | 'openai' | 'mock'
    api_key : API key for OpenAI backend
    model_name : Model name for OpenAI backend
    max_tokens : Max generation tokens
    temperature : Sampling temperature (0 = deterministic)

    Returns
    -------
    str : The assistant's response text.
    """
    if backend == 'local':
        llm = load_local_model()
        # Use chat completion for better structured output
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "<|im_end|>"],
            )
            return response['choices'][0]['message']['content']
        except Exception:
            # Fallback to raw completion if chat format not supported
            prompt = "\n".join(
                f"{'### ' + m['role'].title() + ':'}\n{m['content']}"
                for m in messages
            )
            output = llm(prompt, max_tokens=max_tokens,
                         stop=["</s>", "<|im_end|>"], echo=False)
            return output['choices'][0]['text']

    elif backend == 'openai':
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name or 'gpt-4',
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    elif backend == 'mock':
        # Deterministic mock backend for tests; returns a fixed valid response
        n_links = 8  # Will be overridden by caller if needed
        # Try to extract expected link count from the prompt
        for m in messages:
            match = re.search(r'(\d+) reorder links', m['content'])
            if match:
                n_links = int(match.group(1))
                break
        orders = [10.0] * n_links
        return json.dumps({"orders": orders})

    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


# ── JSON Order Parsing ─────────────────────────────────────────────

def parse_orders_json(
    response: str,
    n_links: int,
) -> tuple[np.ndarray, bool]:
    """
    Parse order quantities from an LLM response.

    Returns
    -------
    (orders, success) : tuple
        orders : np.ndarray of shape (n_links,)
        success : bool — True if valid orders were extracted, False if
                  falling back to default.
    """
    default = np.full(n_links, 10.0)

    try:
        # Strategy 1: JSON code block
        match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        if match:
            data = json.loads(match.group(1))
            orders = data.get('orders', None)
            if orders and len(orders) == n_links:
                return np.maximum(np.array(orders, dtype=np.float64), 0.0), True
            orders_by_link = data.get('orders_by_link', None)
            parsed = _parse_orders_by_link(orders_by_link, n_links)
            if parsed is not None:
                return parsed, True

        # Strategy 1b: Full raw JSON object. This handles nested objects such
        # as {"orders_by_link": {"link_0": 1, ...}}, which regex extraction
        # cannot safely parse.
        try:
            data = json.loads(response.strip())
            orders = data.get('orders', None)
            if orders and len(orders) == n_links:
                return np.maximum(np.array(orders, dtype=np.float64), 0.0), True
            orders_by_link = data.get('orders_by_link', None)
            parsed = _parse_orders_by_link(orders_by_link, n_links)
            if parsed is not None:
                return parsed, True
        except json.JSONDecodeError:
            pass

        # Strategy 2: Inline JSON with "orders" key
        match = re.search(r'\{[^}]*"orders"\s*:\s*\[([^\]]+)\]', response)
        if match:
            vals = [float(x.strip()) for x in match.group(1).split(',')]
            if len(vals) == n_links:
                return np.maximum(np.array(vals, dtype=np.float64), 0.0), True

        # Strategy 3: Raw JSON object anywhere
        for match in re.finditer(r'\{[^{}]+\}', response):
            try:
                data = json.loads(match.group())
                orders = data.get('orders', None)
                if orders and len(orders) == n_links:
                    return np.maximum(np.array(orders, dtype=np.float64), 0.0), True
                orders_by_link = data.get('orders_by_link', None)
                parsed = _parse_orders_by_link(orders_by_link, n_links)
                if parsed is not None:
                    return parsed, True
            except json.JSONDecodeError:
                continue

    except Exception as e:
        logger.warning("LLM parse error: %s", e)

    logger.warning(
        "Failed to parse LLM response (len=%d). Using default [10.0, ...]. "
        "First 200 chars: %s",
        len(response), response[:200]
    )
    return default, False


def _parse_orders_by_link(orders_by_link, n_links: int):
    """Parse {"link_0": q0, ..., "link_N": qN} into action order."""
    if not isinstance(orders_by_link, dict):
        return None

    values = []
    for i in range(n_links):
        keys = (f"link_{i}", str(i), i)
        found = None
        for key in keys:
            if key in orders_by_link:
                found = orders_by_link[key]
                break
        if found is None:
            return None
        values.append(float(found))

    return np.maximum(np.array(values, dtype=np.float64), 0.0)


# ── Prompt Helpers ─────────────────────────────────────────────────

def format_link_desc(reorder_links: list) -> str:
    """Format reorder links as 'src→dst' comma-separated string."""
    return ", ".join(f"{src}→{dst}" for src, dst in reorder_links)


def get_node_type(node, network) -> str:
    """Return human-readable node type string."""
    if node in network.factory:
        return "Factory"
    elif node in network.retail:
        return "Retailer"
    elif node in network.distrib:
        return "Distributor"
    return "Node"


def get_node_costs(node, env) -> dict:
    """Extract cost parameters for a given node from the environment graph."""
    graph = env.network.graph
    costs = {}

    # Holding cost
    if node not in env.network.rawmat:
        costs['holding_cost_h'] = graph.nodes[node].get('h', 0.0)

    # Factory costs
    if node in env.network.factory:
        costs['capacity_C'] = graph.nodes[node].get('C', float('inf'))
        costs['conversion_v'] = graph.nodes[node].get('v', 1.0)
        costs['operating_cost_o'] = graph.nodes[node].get('o', 0.0)

    # Backlog penalty (from retail edges)
    if node in env.network.retail:
        for k in graph.successors(node):
            if (node, k) in env.network.retail_map:
                costs['backlog_penalty_b'] = graph.edges[(node, k)].get('b', 0.0)
                costs['sale_price_p'] = graph.edges[(node, k)].get('p', 0.0)
                break

    return costs


def get_lead_times_for_node(node, env) -> dict:
    """Get lead times of all incoming reorder links to a node."""
    lead_times = {}
    for edge, lt in env.network.lead_times.items():
        if edge[1] == node:
            lead_times[f"{edge[0]}→{edge[1]}"] = lt
    return lead_times


def get_arriving_deliveries(node, env, period: int) -> list[dict]:
    """
    Compute arriving deliveries for a node, matching CoreEnv._update_state logic.

    Returns a list of dicts: [{"link": "src→dst", "delay": k, "quantity": float}, ...]
    """
    arrivals = []
    for edge_tuple in env.network.reorder_links:
        if edge_tuple[1] != node:
            continue
        idx = env.network.reorder_map[edge_tuple]
        L = env.network.lead_times[edge_tuple]
        for k in range(L):
            order_time = period - L + k
            qty = 0.0
            if 0 <= order_time < period:
                qty = float(env.R[order_time, idx])
            if qty > 0:
                arrivals.append({
                    "link": f"{edge_tuple[0]}→{edge_tuple[1]}",
                    "arriving_in_steps": k,
                    "quantity": round(qty, 1),
                })
    return arrivals
