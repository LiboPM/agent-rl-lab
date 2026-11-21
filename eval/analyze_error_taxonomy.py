#!/usr/bin/env python3
"""
Fine-Grained Error Taxonomy & Failure Mode Analysis across all 8 models and 4 benchmark suites:
  - RLLA-4k Test (N=80)
  - RLLA-4k Expanded In-Domain Test (N=500)
  - BFCL-v3 Standard OOD Tool-Calling (N=1,000)
  - BFCL-v3 Live Real-World OOD Tool-Calling (N=1,351)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.compare_models import is_schema_normalized_exact
from eval.run_bfcl_live_eval import load_bfcl_live_samples
from eval.run_expanded_indomain_eval import load_expanded_500_samples
from eval.run_multi_benchmark_eval import (
    load_all_eval_samples,
    score_bfcl_sample,
)
from rewards.rlla_reward import (
    compute_format_reward,
    compute_score,
    extract_assistant_response,
    parse_tool_calls,
)


def classify_rlla_error(sample: Dict[str, Any], pred_raw: str) -> str:
    gt = sample["ground_truth_str"]
    res = compute_score("rlla", pred_raw, gt)
    if res["exact_match"] == 1.0:
        return "CORRECT"

    norm_em = is_schema_normalized_exact(gt, pred_raw, res["exact_match"])
    if norm_em == 1.0:
        return "E7_Type_Quote_Mismatch_Only"

    clean = extract_assistant_response(pred_raw)
    fmt_r, _ = compute_format_reward(clean, gt)

    gt_has_tool = "<tool_call>" in gt
    pd_has_tool = "<tool_call>" in clean and "</tool_call>" in clean

    if gt_has_tool and not pd_has_tool:
        if "<response>" in clean and "</response>" in clean:
            return "E2_False_Abstention_Mode_Error"
        return "E1_Format_or_JSON_Syntax_Error"
    if not gt_has_tool and pd_has_tool:
        return "E2_False_Tool_Invocation_Error"
    if fmt_r == 0.0:
        return "E1_Format_or_JSON_Syntax_Error"

    try:
        gt_tools = parse_tool_calls(gt.split("<tool_call>")[1].split("</tool_call>")[0])
        pd_tools = parse_tool_calls(clean.split("<tool_call>")[1].split("</tool_call>")[0])
    except Exception:
        return "E1_Format_or_JSON_Syntax_Error"

    gt_names = [t["name"] for t in gt_tools]
    pd_names = [t["name"] for t in pd_tools]
    if sorted(gt_names) != sorted(pd_names):
        if len(pd_names) != len(gt_names):
            return "E3b_Tool_Call_Count_Error"
        return "E3a_Wrong_Tool_Selection"

    gt_sorted = sorted(gt_tools, key=lambda x: x["name"])
    pd_sorted = sorted(pd_tools, key=lambda x: x["name"])
    has_missing_key = False
    has_extra_key = False
    for g_t, p_t in zip(gt_sorted, pd_sorted):
        g_keys = set(g_t.get("parameters", {}).keys())
        p_keys = set(p_t.get("parameters", {}).keys())
        if g_keys - p_keys:
            has_missing_key = True
        if p_keys - g_keys:
            has_extra_key = True

    if has_missing_key:
        return "E4_Missing_Required_Param_Key"
    if has_extra_key:
        return "E5_Hallucinated_Extra_Param_Key"
    return "E6_Param_Value_or_Arithmetic_Error"


def classify_bfcl_error(sample: Dict[str, Any], pred_raw: str) -> str:
    res = score_bfcl_sample(pred_raw, sample["ground_truth_str"], sample["bfcl_possible_gt"])
    if res["strict_exact_match"] == 1.0:
        return "CORRECT"
    if res["schema_norm_exact_match"] == 1.0:
        return "E7_Type_Quote_Mismatch_Only"

    clean = extract_assistant_response(pred_raw)
    if "<tool_call>" not in clean or "</tool_call>" not in clean:
        if "<response>" in clean:
            return "E2_False_Abstention_Mode_Error"
        return "E1_Format_or_JSON_Syntax_Error"

    try:
        pd_tools = parse_tool_calls(clean.split("<tool_call>")[1].split("</tool_call>")[0])
    except Exception:
        return "E1_Format_or_JSON_Syntax_Error"

    possible_gt = sample["bfcl_possible_gt"]
    gt_names = [next(iter(item.keys())) for item in possible_gt if isinstance(item, dict)]
    pd_names = [t["name"] for t in pd_tools]

    if sorted(gt_names) != sorted(pd_names):
        if len(pd_names) != len(gt_names):
            return "E3b_Tool_Call_Count_Error"
        return "E3a_Wrong_Tool_Selection"

    has_missing = False
    has_extra = False
    for gt_item, pd_t in zip(possible_gt, pd_tools):
        if isinstance(gt_item, dict):
            fn = next(iter(gt_item.keys()))
            spec = gt_item[fn]
            req_keys = {k for k, v in spec.items() if "" not in v and None not in v}
            all_allowed_keys = set(spec.keys())
            p_keys = set(pd_t.get("parameters", {}).keys())
            if req_keys - p_keys:
                has_missing = True
            if p_keys - all_allowed_keys:
                has_extra = True
    if has_missing:
        return "E4_Missing_Required_Param_Key"
    if has_extra:
        return "E5_Hallucinated_Extra_Param_Key"
    return "E6_Param_Value_or_Arithmetic_Error"


def main() -> None:
    samples_1320 = load_all_eval_samples()
    rlla_samples = [s for s in samples_1320 if s["is_rlla"]]
    bfcl_samples = [s for s in samples_1320 if not s["is_rlla"] and s["benchmark"] != "bfcl_irrelevance"]
    live_samples = load_bfcl_live_samples()
    exp500_samples = load_expanded_500_samples()

    model_tags = [
        ("model0_zeroshot_base", "0. Zero-Shot Base"),
        ("model2a_pure_sft_ep1", "1a. Pure SFT (1.0 Ep, Min Eval-Loss)"),
        ("model2_pure_sft_ep2", "1b. Pure SFT (2.0 Ep, Min Train-Loss)"),
        ("model3_sft_plus_rl_step120", "2a. Full-SFT (2Ep) + GRPO (Negative Control)"),
        ("model4_coldstart_sft320_ep1", "2b. Cold-Start SFT (320, 1.0 Ep)"),
        ("model6_coldstart_sft_plus_rl_step100", "2c. Cold-Start SFT (320) + GRPO (Positive Control)"),
        ("model1_pure_rl_step120", "3a. Pure RL (4.0 Ep, v1 Reward)"),
        ("model5_purerl_v2_ep8_step120", "3b. Pure RL (8.0 Ep, v2 Reward)"),
    ]

    report = {}
    for tag, label in model_tags:
        p_multi = PROJECT_ROOT / "eval" / "multi_bench_dumps" / f"{tag}.jsonl"
        p_live = PROJECT_ROOT / "eval" / "live_bench_dumps" / f"{tag}.jsonl"
        p_exp500 = PROJECT_ROOT / "eval" / "indomain500_dumps" / f"{tag}.jsonl"
        if not p_multi.exists():
            continue
        preds_m = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p_multi)}
        rlla_counts = Counter(classify_rlla_error(s, preds_m[s["id"]]) for s in rlla_samples)
        bfcl_counts = Counter(classify_bfcl_error(s, preds_m[s["id"]]) for s in bfcl_samples)

        live_counts = Counter()
        if p_live.exists():
            preds_l = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p_live)}
            live_counts = Counter(classify_bfcl_error(s, preds_l[s["id"]]) for s in live_samples)

        exp500_counts = Counter()
        if p_exp500.exists():
            preds_e = {json.loads(l)["id"]: json.loads(l)["output"] for l in open(p_exp500)}
            exp500_counts = Counter(classify_rlla_error(s, preds_e[s["id"]]) for s in exp500_samples)

        report[tag] = {
            "label": label,
            "rlla_80": dict(rlla_counts),
            "indomain_500": dict(exp500_counts),
            "bfcl_1000": dict(bfcl_counts),
            "bfcl_live_1351": dict(live_counts),
        }
        print(f"\n=== {label} ({tag}) ===")
        print("  RLLA-4k (80)       :", dict(sorted(rlla_counts.items())))
        print("  In-Domain (500)    :", dict(sorted(exp500_counts.items())))
        print("  BFCL-v3 (1000)     :", dict(sorted(bfcl_counts.items())))
        print("  BFCL-Live (1351)   :", dict(sorted(live_counts.items())))

    out_path = PROJECT_ROOT / "eval" / "error_taxonomy_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full error taxonomy report to {out_path}")


if __name__ == "__main__":
    main()
