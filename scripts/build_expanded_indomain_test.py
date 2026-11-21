#!/usr/bin/env python3
"""
Build a 500-sample Expanded In-Domain Test Set (`data/rlla_4k/test_indomain_500.parquet`, N=500)
to complement the 80-sample original `rlla_4k/test.parquet`.

Strict In-Domain Criteria:
  1. 100% of the tools in each test sample's system prompt MUST appear in `rlla_4k/train.parquet`
     (so the tools and domain schemas are strictly In-Domain / seen during training).
  2. 0% overlap in user queries with `rlla_4k/train.parquet` (3,920 rows) or `test.parquet` (80 rows).
  3. Balanced composition matching RLLA-4k:
     - 250 ToolACE In-Domain samples + 250 xLAM In-Domain samples
     - 250 Single-Tool calls + 250 Multi-Tool calls
  4. 100% AST-validated Ground Truth JSON (eliminating the `.split(",")` truncation bug).
"""

from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RLLA_DIR = PROJECT_ROOT / "data" / "rlla_4k"

RLLA_SYSTEM_TEMPLATE = """You are a helpful multi-turn dialogue assistant capable of leveraging tool calls to solve user tasks and provide structured chat responses.

**Available Tools**
In your response, you can use the following tools:
{tools_block}

**Steps for Each Turn**
1. **Think:** Recall relevant context and analyze the current user goal.
2. **Decide on Tool Usage:** If a tool is needed, specify the tool and its parameters.
3. **Respond Appropriately:** If a response is needed, generate one while maintaining consistency across user queries.

**Output Format**
```plaintext
<think> Your thoughts and reasoning </think>
<tool_call>
{{"name": "Tool name", "parameters": {{"Parameter name": "Parameter content", "... ...": "... ..."}}}}
{{"name": "... ...", "parameters": {{"... ...": "... ...", "... ...": "... ..."}}}}
...
</tool_call>
<response> AI's final response </response>
```

**Important Notes**
1. You must always include the `<think>` field to outline your reasoning. Provide at least one of `<tool_call>` or `<response>`. Decide whether to use `<tool_call>` (possibly multiple times), `<response>`, or both.
2. You can invoke multiple tool calls simultaneously in the `<tool_call>` fields. Each tool call should be a JSON object with a "name" field and an "parameters" field containing a dictionary of parameters. If no parameters are needed, leave the "parameters" field an empty dictionary.
3. Refer to the previous dialogue records in the history, including the user's queries, previous `<tool_call>`, `<response>`, and any tool feedback noted as `<obs>` (if exists)."""


def format_rlla_tools_block(tools: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, t in enumerate(tools, start=1):
        name = t["name"]
        desc = t.get("description", "")
        params = t.get("parameters", {})
        if isinstance(params, dict) and "properties" in params and isinstance(params.get("properties"), dict):
            props = params["properties"]
            req_raw = params.get("required") or []
            req = set(req_raw) if isinstance(req_raw, list) else set()
            flat_params = {}
            for pk, pv in props.items():
                if not isinstance(pv, dict):
                    continue
                p_entry: Dict[str, Any] = {
                    "description": pv.get("description", ""),
                    "type": pv.get("type", "string"),
                }
                if "default" in pv:
                    p_entry["default"] = pv["default"]
                elif pk not in req:
                    p_entry["default"] = ""
                flat_params[pk] = p_entry
            params_str = json.dumps(flat_params, ensure_ascii=False)
        else:
            params_str = json.dumps(params, ensure_ascii=False)
        lines.append(f"{idx}. Name: {name}\nDescription: {desc}\nParameters: {params_str}")
    return "\n".join(lines)


def parse_toolace_call_list(call_str: str, valid_tool_names: set[str]) -> Optional[List[Dict[str, Any]]]:
    """
    Safely parse ToolACE `[Func Name(k1=v1, k2=v2), FuncB(...)]` where function names
    may contain spaces/hyphens, using balanced parenthesis scanning + `ast.parse("f(" + args + ")")`.
    """
    s = call_str.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return None
    inner = s[1:-1].strip()
    if not inner:
        return None

    # Split top-level calls separated by `, ` outside parentheses/quotes/brackets
    top_calls = []
    depth = 0
    in_str = None
    escape = False
    start = 0
    for i, ch in enumerate(inner):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"'):
            in_str = ch
        elif ch in ("(", "[", "{"):
            depth += 1
        elif ch in (")", "]", "}"):
            depth -= 1
        elif ch == "," and depth == 0:
            top_calls.append(inner[start:i].strip())
            start = i + 1
    top_calls.append(inner[start:].strip())

    parsed_calls = []
    for c_str in top_calls:
        if "(" not in c_str or not c_str.endswith(")"):
            return None
        fn_name, args_part = c_str.split("(", 1)
        fn_name = fn_name.strip()
        if fn_name not in valid_tool_names:
            return None
        args_inner = args_part[:-1].strip()
        try:
            tree = ast.parse(f"f({args_inner})", mode="eval")
        except SyntaxError:
            return None
        if not isinstance(tree.body, ast.Call) or tree.body.args:
            return None
        params = {}
        for kw in tree.body.keywords:
            if kw.arg is None:
                return None
            try:
                val = ast.literal_eval(kw.value)
            except Exception:
                return None
            params[kw.arg] = val
        parsed_calls.append({"name": fn_name, "parameters": params})
    return parsed_calls if parsed_calls else None


def main() -> None:
    random.seed(42)
    tr = pd.read_parquet(RLLA_DIR / "train.parquet")
    te = pd.read_parquet(RLLA_DIR / "test.parquet")

    seen_queries = set()
    train_tool_names = set()
    for df in (tr, te):
        for row in df.itertuples():
            u = row.prompt[1]["content"]
            m_u = re.search(r"<user>\s*(.*?)\s*</user>", u, re.DOTALL)
            if m_u:
                seen_queries.add(m_u.group(1).strip())
            s = row.prompt[0]["content"]
            for m in re.finditer(r"\d+\.\s*Name:\s*(.+)", s):
                train_tool_names.add(m.group(1).strip())

    print(f"Loaded {len(seen_queries)} existing RLLA-4k queries and {len(train_tool_names)} In-Domain tool names.")

    xlam_single, xlam_multi = [], []
    tace_single, tace_multi = [], []

    # 1. Extract strictly In-Domain held-out samples from xLAM-60k
    xlam = json.load(open("/tmp/xlam60k.json", "r", encoding="utf-8"))
    for item in xlam:
        q = item["query"].strip()
        if q in seen_queries or len(q) < 15:
            continue
        try:
            tools = json.loads(item["tools"])
            answers = json.loads(item["answers"])
        except Exception:
            continue
        if not tools or not answers:
            continue
        t_names = {t.get("name", "") for t in tools}
        if not t_names.issubset(train_tool_names):
            continue

        gt_calls = []
        valid = True
        for a in answers:
            if "name" not in a or "arguments" not in a or not isinstance(a["arguments"], dict):
                valid = False
                break
            gt_calls.append({"name": a["name"], "parameters": a["arguments"]})
        if not valid:
            continue

        sys_prompt = RLLA_SYSTEM_TEMPLATE.format(tools_block=format_rlla_tools_block(tools))
        usr_prompt = f"**Dialogue Records History**\n<user> {q} </user>"
        gt_str = (
            "<think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>\n"
            "<tool_call>\n"
            + "\n".join(json.dumps(c, ensure_ascii=False) for c in gt_calls)
            + "\n</tool_call>"
        )
        rec = {
            "source": "xlam_indomain",
            "is_multi": len(gt_calls) >= 2,
            "prompt": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            "ground_truth": gt_str,
            "query": q,
        }
        if len(gt_calls) == 1:
            xlam_single.append(rec)
        else:
            xlam_multi.append(rec)

    # 2. Extract strictly In-Domain held-out samples from ToolACE
    toolace = json.load(open("/tmp/toolace.json", "r", encoding="utf-8"))
    for item in toolace:
        convs = item.get("conversations", [])
        if len(convs) != 2 or convs[0].get("from") != "user" or convs[1].get("from") != "assistant":
            continue
        q = convs[0]["value"].strip()
        if q in seen_queries or len(q) < 15:
            continue
        sys_raw = item.get("system", "")
        if "Here is a list of functions in JSON format that you can invoke:\n" not in sys_raw:
            continue
        json_part = sys_raw.split("Here is a list of functions in JSON format that you can invoke:\n", 1)[1]
        json_part = json_part.split("Should you decide to return the function call(s)", 1)[0].strip()
        if json_part.endswith("."):
            json_part = json_part[:-1].strip()
        try:
            tools = json.loads(json_part)
        except Exception:
            continue
        if not isinstance(tools, list) or not tools:
            continue
        t_names = {t.get("name", "") for t in tools if isinstance(t, dict)}
        if not t_names or not t_names.issubset(train_tool_names):
            continue

        gt_calls = parse_toolace_call_list(convs[1]["value"], t_names)
        if not gt_calls:
            continue

        sys_prompt = RLLA_SYSTEM_TEMPLATE.format(tools_block=format_rlla_tools_block(tools))
        usr_prompt = f"**Dialogue Records History**\n<user> {q} </user>"
        gt_str = (
            "<think> I should use the appropriate tool with proper parameters to respond to the user's need. </think>\n"
            "<tool_call>\n"
            + "\n".join(json.dumps(c, ensure_ascii=False) for c in gt_calls)
            + "\n</tool_call>"
        )
        rec = {
            "source": "toolace_indomain",
            "is_multi": len(gt_calls) >= 2,
            "prompt": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": usr_prompt}],
            "ground_truth": gt_str,
            "query": q,
        }
        if len(gt_calls) == 1:
            tace_single.append(rec)
        else:
            tace_multi.append(rec)

    def dedup(pool: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for r in pool:
            if r["query"] not in seen:
                seen.add(r["query"])
                out.append(r)
        return out

    xlam_single, xlam_multi = dedup(xlam_single), dedup(xlam_multi)
    tace_single, tace_multi = dedup(tace_single), dedup(tace_multi)

    print(
        f"Candidate strictly In-Domain pool:\n"
        f"  ToolACE: Single={len(tace_single)}, Multi={len(tace_multi)}\n"
        f"  xLAM   : Single={len(xlam_single)}, Multi={len(xlam_multi)}"
    )

    random.shuffle(tace_single)
    random.shuffle(tace_multi)
    random.shuffle(xlam_single)
    random.shuffle(xlam_multi)

    # Sample 125 from each of the 4 buckets = 500 samples (250 ToolACE + 250 xLAM; 250 Single + 250 Multi)
    chosen = tace_single[:125] + tace_multi[:125] + xlam_single[:125] + xlam_multi[:125]
    random.shuffle(chosen)

    rows = []
    for idx, item in enumerate(chosen):
        rows.append(
            {
                "data_source": "rlla",
                "prompt": item["prompt"],
                "ability": "tool_call",
                "reward_model": {"ground_truth": item["ground_truth"], "style": "rule"},
                "extra_info": {
                    "index": idx,
                    "source": item["source"],
                    "sub_category": "rlla_multi_tool" if item["is_multi"] else "rlla_single_tool",
                    "split": "test_indomain_500",
                },
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = RLLA_DIR / "test_indomain_500.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Saved {len(out_df)} strictly In-Domain test samples (250 ToolACE + 250 xLAM) to {out_path}")


if __name__ == "__main__":
    main()
