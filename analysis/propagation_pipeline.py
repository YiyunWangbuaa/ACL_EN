# -*- coding: utf-8 -*-
"""Dependency-graph error propagation statistics for the thinking-off A/B runs.

Reads results/nothink_pair/error_nodes.jsonl. States are recomputed from
is_wrong and logical_parents. first_error_nodes stay dependency candidates.
A gold-repair effect is not estimated: that needs a do(node=gold) rerun.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import statsmodels.api as sm
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "nothink_pair" / "error_nodes.jsonl"
OUT = ROOT / "results" / "nothink_pair" / "propagation"

GROUPS = ("standard", "standard-order", "alpha", "roll", "order", "mount", "body")
LABELS = {
    "standard": "统一默认",
    "standard-order": "球腕顺序",
    "alpha": "理论迎角正负",
    "roll": "机构滚转方向",
    "order": "机构角顺序",
    "mount": "正装反装",
    "body": "体轴系手性",
}
MATCHED = {
    "alpha": "standard",
    "roll": "standard",
    "mount": "standard",
    "body": "standard",
    "order": "standard-order",
}
STANDARDS = frozenset({"standard", "standard-order"})
# A rate gap is called material only when it is at least this large and the
# paired test is significant. 176 nodes make very small gaps "significant".
MATERIAL = 0.05
ALPHA = 0.05
BOOT = 1000
RNG = np.random.default_rng(20261102)


def _id_key(blank_id: str) -> int:
    return int(str(blank_id).removeprefix("ID"))


def load_rows() -> list[dict]:
    rows = []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _parents_of(rows: list[dict]) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for row in rows:
        parents = tuple(row["logical_parents"])
        previous = found.get(row["target_id"])
        if previous is not None and previous != parents:
            raise RuntimeError(f"{row['target_id']} 的逻辑父节点在 A/B 间不一致")
        found[row["target_id"]] = parents
    return found


def _children(parents: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    found: dict[str, list[str]] = defaultdict(list)
    for child, upstream in parents.items():
        for parent in upstream:
            if parent in parents:
                found[parent].append(child)
    return {node: tuple(sorted(kids, key=_id_key)) for node, kids in found.items()}


def _first_errors(wrong: dict[str, bool], parents: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    memo: dict[str, tuple[str, ...]] = {}

    def walk(node: str, stack: set[str]) -> tuple[str, ...]:
        if node in memo:
            return memo[node]
        if node in stack:
            return ()
        stack.add(node)
        upstream: list[str] = []
        for parent in parents.get(node, ()):
            if wrong.get(parent) is True:
                upstream.extend(walk(parent, stack))
        stack.remove(node)
        if wrong.get(node) and not upstream:
            result = (node,)
        else:
            result = tuple(sorted(set(upstream), key=_id_key))
        memo[node] = result
        return result

    for node in wrong:
        walk(node, set())
    return memo


def _distances(root: str, children: dict[str, tuple[str, ...]]) -> dict[str, int]:
    distance = {root: 0}
    queue = [root]
    while queue:
        current = queue.pop(0)
        for child in children.get(current, ()):
            if child in distance:
                continue
            distance[child] = distance[current] + 1
            queue.append(child)
    return distance


def label_run(rows: list[dict], gold_prefix: bool) -> dict[str, dict]:
    """Four states. Gold prefix does not inherit a model error from a parent."""
    wrong = {row["target_id"]: bool(row["is_wrong"]) for row in rows}
    parents = _parents_of(rows)
    present = set(wrong)
    children = _children(parents)
    sources = {node: () for node in present} if gold_prefix else _first_errors(wrong, parents)
    roots = [node for node, source in sources.items() if source == (node,)]
    distance_of = {root: _distances(root, children) for root in roots}
    labeled = {}
    for row in rows:
        node = row["target_id"]
        source = sources[node]
        if wrong[node] and (gold_prefix or source == (node,)):
            kind = "local_error"
        elif wrong[node]:
            kind = "propagated_error"
        elif source:
            kind = "recovered"
        else:
            kind = "correct"
        depth = 0
        if source and not gold_prefix and kind != "local_error":
            hops = [
                distance_of[origin][node]
                for origin in source
                if origin in distance_of and node in distance_of[origin]
            ]
            depth = min(hops) if hops else 0
        attributed = []
        if kind == "local_error" and not gold_prefix:
            for other, other_source in sources.items():
                if other != node and node in other_source:
                    attributed.append(other)
        parent_wrong = any(wrong.get(parent) is True for parent in parents.get(node, ()))
        labeled[node] = {
            "target_id": node,
            "is_wrong": wrong[node],
            "error_type": kind,
            "first_error_nodes": [node] if kind == "local_error" else list(source),
            "propagation_depth": depth,
            "parent_wrong": parent_wrong,
            "logical_parents": list(parents.get(node, ())),
            "process": row["process"],
            "error_category": row["error_category"],
            "definition_entries": row.get("definition_entries") or [],
            "attributed_descendants": sorted(attributed, key=_id_key),
            "candidate_only": True,
        }
    return labeled


def _rate(part: int, whole: int) -> float | None:
    if whole == 0:
        return None
    return round(part / whole, 4)


def run_metrics(labeled: dict[str, dict], gold_prefix: bool) -> dict:
    nodes = list(labeled.values())
    total = len(nodes)
    local = [item for item in nodes if item["error_type"] == "local_error"]
    propagated = [item for item in nodes if item["error_type"] == "propagated_error"]
    recovered = [item for item in nodes if item["error_type"] == "recovered"]
    wrong_n = len(local) + len(propagated)
    affected = len(propagated) + len(recovered)
    depths = [item["propagation_depth"] for item in propagated]
    root_depths = []
    for item in local:
        descendant_depths = [
            labeled[other]["propagation_depth"]
            for other in item["attributed_descendants"]
            if labeled[other]["error_type"] == "propagated_error"
        ]
        if descendant_depths:
            root_depths.append(max(descendant_depths))
    return {
        "nodes": total,
        "local_error": len(local),
        "propagated_error": len(propagated),
        "recovered": len(recovered),
        "correct": sum(item["error_type"] == "correct" for item in nodes),
        "wrong": wrong_n,
        "r_root": _rate(len(local), total),
        "r_node": _rate(wrong_n, total),
        "amplification": round(wrong_n / len(local), 4) if local else None,
        "recovery_rate": _rate(len(recovered), affected),
        "affected_by_prefix": affected,
        "mean_propagation_depth": round(sum(depths) / len(depths), 4) if depths else 0.0,
        "mean_root_reach": round(sum(root_depths) / len(root_depths), 4) if root_depths else 0.0,
        "multi_source": sum(len(item["first_error_nodes"]) > 1 for item in nodes),
        "gold_prefix": gold_prefix,
    }


def survival(labeled: dict[str, dict]) -> dict[str, dict]:
    """S(d): among descendants d edges from a first error, the share still wrong."""
    parents = {item["target_id"]: tuple(item["logical_parents"]) for item in labeled.values()}
    children = _children(parents)
    roots = [node for node, item in labeled.items() if item["error_type"] == "local_error"]
    distance_of = {root: _distances(root, children) for root in roots}
    pair_wrong: dict[int, int] = defaultdict(int)
    pair_total: dict[int, int] = defaultdict(int)
    nearest: dict[str, int] = {}
    for root in roots:
        for node, distance in distance_of[root].items():
            if distance == 0:
                continue
            pair_total[distance] += 1
            if labeled[node]["is_wrong"]:
                pair_wrong[distance] += 1
            nearest[node] = min(nearest.get(node, distance), distance)
    unique_wrong: dict[int, int] = defaultdict(int)
    unique_total: dict[int, int] = defaultdict(int)
    for node, distance in nearest.items():
        unique_total[distance] += 1
        if labeled[node]["is_wrong"]:
            unique_wrong[distance] += 1
    by_distance = {}
    for distance in sorted(set(pair_total) | set(unique_total)):
        by_distance[str(distance)] = {
            "distance_edges": distance,
            "pair_wrong": pair_wrong[distance],
            "pair_total": pair_total[distance],
            "s_pair": _rate(pair_wrong[distance], pair_total[distance]),
            "unique_wrong": unique_wrong[distance],
            "unique_total": unique_total[distance],
            "s_unique": _rate(unique_wrong[distance], unique_total[distance]),
        }
    intervals = _survival_interval(roots, distance_of, labeled) if roots else {}
    return {"by_distance": by_distance, "root_bootstrap": intervals, "roots": len(roots)}


def _survival_interval(roots: list[str], distance_of: dict, labeled: dict) -> dict[str, dict]:
    """Resamples first-error roots. This is not a sample-level bootstrap."""
    if len(roots) < 2:
        return {}
    root_ids = np.array(roots)
    draws = {str(distance): [] for distance in range(1, 6)}
    for _ in range(BOOT):
        chosen = RNG.choice(root_ids, size=len(root_ids), replace=True)
        wrong = defaultdict(int)
        total = defaultdict(int)
        for root in chosen:
            for node, distance in distance_of[root].items():
                if distance == 0 or distance > 5:
                    continue
                total[distance] += 1
                if labeled[node]["is_wrong"]:
                    wrong[distance] += 1
        for distance in range(1, 6):
            draws[str(distance)].append(
                wrong[distance] / total[distance] if total[distance] else np.nan
            )
    intervals = {}
    for distance, values in draws.items():
        clean = np.array([value for value in values if not np.isnan(value)])
        if len(clean) == 0:
            continue
        intervals[distance] = {
            "low": round(float(np.quantile(clean, 0.025)), 4),
            "high": round(float(np.quantile(clean, 0.975)), 4),
        }
    return intervals


def paired(left: dict[str, dict], right: dict[str, dict]) -> dict:
    both = sorted(set(left) & set(right), key=_id_key)
    cells = {"correct_correct": 0, "correct_wrong": 0, "wrong_wrong": 0, "wrong_correct": 0}
    sensitive_local = 0
    sensitive_propagated = 0
    for node in both:
        a_wrong = left[node]["is_wrong"]
        b_wrong = right[node]["is_wrong"]
        if not a_wrong and not b_wrong:
            cells["correct_correct"] += 1
        elif not a_wrong and b_wrong:
            cells["correct_wrong"] += 1
            if right[node]["error_type"] == "local_error":
                sensitive_local += 1
            else:
                sensitive_propagated += 1
        elif a_wrong and b_wrong:
            cells["wrong_wrong"] += 1
        else:
            cells["wrong_correct"] += 1
    discordant = cells["correct_wrong"] + cells["wrong_correct"]
    p_value = 1.0 if discordant == 0 else float(
        binomtest(cells["correct_wrong"], discordant, 0.5, alternative="two-sided").pvalue
    )
    return {
        **cells,
        "paired_nodes": len(both),
        "prefix_loss": cells["correct_wrong"] - cells["wrong_correct"],
        "sensitive_local": sensitive_local,
        "sensitive_propagated": sensitive_propagated,
        "mcnemar_p": round(p_value, 6),
    }


def _mcnemar_direction(base: dict[str, bool], other: dict[str, bool]) -> dict:
    only_other = only_base = 0
    for node in set(base) & set(other):
        if other[node] and not base[node]:
            only_other += 1
        elif base[node] and not other[node]:
            only_base += 1
    discordant = only_other + only_base
    p_value = 1.0 if discordant == 0 else float(
        binomtest(only_other, discordant, 0.5, alternative="two-sided").pvalue
    )
    whole = len(set(base) & set(other))
    return {
        "only_group": only_other,
        "only_standard": only_base,
        "p": round(p_value, 6),
        "rate_gap": round((only_other - only_base) / whole, 4) if whole else None,
        "material": bool(whole and abs(only_other - only_base) / whole >= MATERIAL and p_value < ALPHA),
        "higher": bool(only_other > only_base and p_value < ALPHA and whole and (only_other - only_base) / whole >= MATERIAL),
    }


def classify_group(group: str, runs: dict, pairs: dict) -> dict:
    if group not in MATCHED:
        return {"group": group, "role": "standard", "type": "标准参照"}
    base = MATCHED[group]
    a_here = {node: item["is_wrong"] for node, item in runs[(group, "A")].items()}
    a_base = {node: item["is_wrong"] for node, item in runs[(base, "A")].items()}
    loss_here = {
        node: (not runs[(group, "A")][node]["is_wrong"] and runs[(group, "B")][node]["is_wrong"])
        for node in runs[(group, "A")]
    }
    loss_base = {
        node: (not runs[(base, "A")][node]["is_wrong"] and runs[(base, "B")][node]["is_wrong"])
        for node in runs[(base, "A")]
    }
    local = _mcnemar_direction(a_base, a_here)
    propagation = _mcnemar_direction(loss_base, loss_here)
    amp = runs[(group, "B")]
    amp_base = runs[(base, "B")]
    # metrics objects are stored separately; this function receives labeled runs.
    # Amplification is filled by the caller. Placeholder keys are overwritten.
    return {
        "group": group,
        "matched_standard": base,
        "role": "heterogeneous",
        "local_vs_standard": local,
        "prefix_loss_vs_standard": propagation,
        "pending_structure": (amp, amp_base),
    }


def _finish_type(item: dict, metrics: dict) -> dict:
    if item["role"] == "standard":
        return item
    group_b = metrics[(item["group"], "B")]
    base_b = metrics[(item["matched_standard"], "B")]
    amp_up = (
        group_b["amplification"] is not None
        and base_b["amplification"] is not None
        and group_b["amplification"] > base_b["amplification"]
        and group_b["mean_propagation_depth"] > base_b["mean_propagation_depth"]
    )
    local_up = item["local_vs_standard"]["higher"]
    prop_up = item["prefix_loss_vs_standard"]["higher"]
    if local_up and prop_up and amp_up:
        kind = "结构放大型"
    elif local_up and prop_up:
        kind = "局部与传播同时增加"
    elif local_up:
        kind = "局部敏感型"
    elif prop_up:
        kind = "传播敏感型"
    else:
        kind = "与配对标准无显著差别"
    item["amplification_and_depth_higher"] = amp_up
    item["type"] = kind
    item.pop("pending_structure", None)
    return item


def _design(records: list[dict], exposed: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = []
    outcome = []
    for record in records:
        parent = record["parent_exposed"] if exposed else record["parent_wrong"]
        rows.append([
            1.0,
            record["self_prefix"],
            record["heterogeneity"],
            record["depth"],
            parent,
            record["matrix"],
            record["heterogeneity"] * parent,
        ])
        outcome.append(record["is_wrong"])
    names = [
        "intercept",
        "self_prefix",
        "heterogeneity",
        "depth",
        "parent_error",
        "matrix_node",
        "heterogeneity_x_parent_error",
    ]
    return np.asarray(rows, dtype=float), np.asarray(outcome, dtype=float), names


def _fit_logit(x_values: np.ndarray, y_values: np.ndarray, names: list[str], clusters: list[str]) -> dict:
    try:
        model = sm.Logit(y_values, x_values).fit(
            disp=False,
            maxiter=200,
            cov_type="cluster",
            cov_kwds={"groups": np.asarray(clusters)},
        )
    except Exception as exc:  # noqa: BLE001 - report the failed specification
        return {"converged": False, "error": str(exc)}
    coefficients = []
    for name, value, error, p_value in zip(names, model.params, model.bse, model.pvalues):
        coefficients.append({
            "term": name,
            "coef": round(float(value), 4),
            "odds_ratio": round(float(np.exp(value)), 4),
            "std_err": round(float(error), 4),
            "p": round(float(p_value), 6),
        })
    return {
        "converged": bool(model.mle_retvals.get("converged", True)),
        "n": int(model.nobs),
        "clusters": len(set(clusters)),
        "pseudo_r2": round(float(model.prsquared), 4),
        "coefficients": coefficients,
        "note": "聚类为单位定义稿。每稿只有一份样本，标准误不能当成多样本抽样误差。",
    }


def regression(flat: list[dict]) -> dict:
    full_x, full_y, names = _design(flat, exposed=False)
    exposed_x, exposed_y, _ = _design(flat, exposed=True)
    clusters = [record["group"] for record in flat]
    b_only = [record for record in flat if record["self_prefix"] == 1.0]
    b_x, b_y, b_names = _design(b_only, exposed=True)
    # self_prefix is constant on B. Drop it and the columns stay aligned by refitting names.
    keep = [index for index, name in enumerate(b_names) if name != "self_prefix"]
    b_names = [b_names[index] for index in keep]
    b_x = b_x[:, keep]
    return {
        "specified_parent_wrong": _fit_logit(full_x, full_y, names, clusters),
        "parent_exposed_on_self_prefix": _fit_logit(exposed_x, exposed_y, names, clusters),
        "self_prefix_only": _fit_logit(b_x, b_y, b_names, [record["group"] for record in b_only]),
        "reading": {
            "specified_parent_wrong": "父节点在同一次运行里答错。A 条件下这个错误没有写进子节点前缀。",
            "parent_exposed_on_self_prefix": "父节点答错，并且当前注入把模型自己的前缀交给了子节点。",
            "self_prefix_only": "只看 B。parent_error 表示错误前缀确实进入了当前节点。",
        },
    }


def intervention_candidates(labeled_b: dict[str, dict[str, dict]]) -> list[dict]:
    bucket: dict[str, dict] = {}
    for group, labeled in labeled_b.items():
        for node, item in labeled.items():
            slot = bucket.setdefault(node, {
                "target_id": node,
                "root_groups": [],
                "attributed_wrong": 0,
                "attributed_recovered": 0,
                "max_depth": 0,
                "confluence_groups": 0,
                "processes": set(),
                "entries": [],
                "category": "",
            })
            slot["entries"] = item["definition_entries"]
            slot["category"] = item["error_category"]
            if item["error_type"] == "local_error":
                slot["root_groups"].append(group)
                wrong_desc = [
                    other for other in item["attributed_descendants"]
                    if labeled[other]["error_type"] == "propagated_error"
                ]
                recovered_desc = [
                    other for other in item["attributed_descendants"]
                    if labeled[other]["error_type"] == "recovered"
                ]
                slot["attributed_wrong"] += len(wrong_desc)
                slot["attributed_recovered"] += len(recovered_desc)
                if wrong_desc:
                    slot["max_depth"] = max(
                        slot["max_depth"],
                        max(labeled[other]["propagation_depth"] for other in wrong_desc),
                    )
                slot["processes"].update(labeled[other]["process"] for other in item["attributed_descendants"])
                slot["processes"].add(item["process"])
            for origin in item["first_error_nodes"]:
                if len(item["first_error_nodes"]) <= 1:
                    continue
                origin_slot = bucket.setdefault(origin, {
                    "target_id": origin,
                    "root_groups": [],
                    "attributed_wrong": 0,
                    "attributed_recovered": 0,
                    "max_depth": 0,
                    "confluence_groups": 0,
                    "processes": set(),
                    "entries": [],
                    "category": "",
                })
                origin_slot["confluence_groups"] += 1
    ranked = []
    for slot in bucket.values():
        if not slot["root_groups"]:
            continue
        ranked.append({
            "target_id": slot["target_id"],
            "root_groups": slot["root_groups"],
            "root_frequency": len(slot["root_groups"]),
            "attributed_wrong_descendants": slot["attributed_wrong"],
            "attributed_recovered_descendants": slot["attributed_recovered"],
            "max_depth": slot["max_depth"],
            "confluence_groups": slot["confluence_groups"],
            "processes": sorted(slot["processes"]),
            "cross_process": len(slot["processes"]) > 1,
            "definition_entries": slot["entries"],
            "error_category": slot["category"],
            "causal_effect": None,
            "note": "描述性优先级。传播影响 I(v) 要等 do(该节点=gold) 之后才能计算。",
        })
    ranked.sort(key=lambda item: (
        -item["root_frequency"],
        -item["attributed_wrong_descendants"],
        -item["max_depth"],
        -item["confluence_groups"],
        _id_key(item["target_id"]),
    ))
    return ranked


def _stratum(runs: dict[tuple[str, str], dict[str, dict]]) -> dict:
    found: dict[str, dict] = {}
    for (group, injection), labeled in runs.items():
        for item in labeled.values():
            for key, value in (("process", item["process"]), ("error_category", item["error_category"])):
                slot = found.setdefault(f"{key}:{value}", {
                    "stratum": key,
                    "name": value,
                    "A_wrong": 0,
                    "A_nodes": 0,
                    "B_local": 0,
                    "B_wrong": 0,
                    "B_recovered": 0,
                    "B_affected": 0,
                    "B_nodes": 0,
                })
                if injection == "A":
                    slot["A_nodes"] += 1
                    slot["A_wrong"] += int(item["is_wrong"])
                else:
                    slot["B_nodes"] += 1
                    slot["B_wrong"] += int(item["is_wrong"])
                    slot["B_local"] += int(item["error_type"] == "local_error")
                    slot["B_recovered"] += int(item["error_type"] == "recovered")
                    slot["B_affected"] += int(item["error_type"] in {"propagated_error", "recovered"})
    rows = []
    for slot in found.values():
        rows.append({
            **slot,
            "A_error_rate": _rate(slot["A_wrong"], slot["A_nodes"]),
            "B_error_rate": _rate(slot["B_wrong"], slot["B_nodes"]),
            "B_r_root": _rate(slot["B_local"], slot["B_nodes"]),
            "B_amplification": round(slot["B_wrong"] / slot["B_local"], 4) if slot["B_local"] else None,
            "B_recovery_rate": _rate(slot["B_recovered"], slot["B_affected"]),
        })
    return {"rows": rows}


_TRANSITION = {
    (False, False): "保持正确",
    (False, True): "转错",
    (True, False): "转对",
    (True, True): "持续错误",
}


def _prefix_location(sample_ids: set[str]) -> dict[str, dict[str, tuple[str, int]]]:
    """Public id to (process, sentence index) in the manuscript actually scored.

    gold-trunc and self-trunc inject only earlier sentences of the same subprocess.
    """
    from cloze_eval import contexts_for

    path = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
    found: dict[str, dict[str, tuple[str, int]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") not in sample_ids:
            continue
        index: dict[str, tuple[str, int]] = {}
        for chain in contexts_for(row):
            for sentence_i, sentence in enumerate(chain["sentences"]):
                for blank_id in sentence["blank_ids"]:
                    place = (chain["process"], sentence_i)
                    bare = str(blank_id).removeprefix("ID")
                    index[str(blank_id)] = place
                    index[bare] = place
                    index[f"ID{bare}"] = place
        found[row["id"]] = index
    missing = sample_ids - set(found)
    if missing:
        raise RuntimeError(f"题面里没有这些样本：{sorted(missing)}")
    return found


def _exposed_wrong_parents(
    target: str,
    logical_parents: tuple[str, ...] | list[str],
    wrong: dict[str, bool],
    location: dict[str, tuple[str, int]],
) -> tuple[list[str], list[str]]:
    """Wrong logical parents whose answers were injected before this call."""
    here = location.get(target)
    exposed: list[str] = []
    withheld: list[str] = []
    for parent in logical_parents:
        if wrong.get(parent) is not True:
            continue
        parent_at = location.get(parent)
        injected = (
            here is not None
            and parent_at is not None
            and parent_at[0] == here[0]
            and parent_at[1] < here[1]
        )
        if injected:
            exposed.append(parent)
        else:
            withheld.append(parent)
    return exposed, withheld


def _wrong_ancestors(node: str, wrong: dict[str, bool], parents: dict[str, tuple[str, ...]]) -> list[str]:
    found = []
    seen = set()

    def walk(current: str) -> None:
        for parent in parents.get(current, ()):
            if parent in seen or parent not in wrong:
                continue
            seen.add(parent)
            if wrong[parent]:
                found.append(parent)
            walk(parent)

    walk(node)
    return sorted(found, key=_id_key)


def audit_rows(rows: list[dict]) -> dict:
    """Pair on sample_id + target_id. A missing score stays missing."""
    seen: dict[tuple[str, str, str], int] = {}
    duplicates = []
    unscored = []
    by_sample: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        sample = str(row.get("sample_id") or "")
        target = str(row.get("target_id") or "")
        injection = str(row.get("injection") or "")
        key = (sample, target, injection)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 2:
            duplicates.append({"sample_id": sample, "target_id": target, "injection": injection})
        if row.get("is_wrong") not in (True, False):
            unscored.append({"sample_id": sample, "target_id": target, "injection": injection})
        if sample and target and injection in {"A", "B"} and row.get("is_wrong") in (True, False):
            by_sample[sample][injection].add(target)
    missing = []
    for sample, sides in sorted(by_sample.items()):
        targets = sides.get("A", set()) | sides.get("B", set())
        for target in sorted(targets, key=_id_key):
            absent = [side for side in ("A", "B") if target not in sides.get(side, set())]
            if absent:
                missing.append({"sample_id": sample, "target_id": target, "missing_injection": absent})
        for side in ("A", "B"):
            count = len(sides.get(side, set()))
            if count != 176:
                missing.append({"sample_id": sample, "injection": side, "scored": count, "expected": 176})
    return {
        "rows": len(rows),
        "duplicate_keys": duplicates,
        "unscored": unscored,
        "incomplete_pairs": [item for item in missing if "target_id" in item],
        "unexpected_counts": [item for item in missing if "scored" in item],
        "complete": not duplicates and not unscored and not missing,
    }


def _accumulate(bucket: dict, a_wrong: bool, b_wrong: bool) -> None:
    bucket["A_correct" if not a_wrong else "A_wrong"] += 1
    if not a_wrong and b_wrong:
        bucket["to_wrong"] += 1


_CATEGORIES = ("semantic", "matrix", "sign", "expression", "vector")
_CATEGORY_LABEL = {
    "semantic": "语义",
    "matrix": "矩阵",
    "sign": "符号",
    "expression": "公式",
    "vector": "向量",
}


def _category_exposure(pair_rows: list[dict]) -> list[dict]:
    slots = {
        category: {"none_n": 0, "none_wrong": 0, "exposed_n": 0, "exposed_wrong": 0}
        for category in _CATEGORIES
    }
    for row in pair_rows:
        if row.get("transition") == "缺失" or not row.get("A_correct"):
            continue
        category = row.get("error_category") or ""
        slot = slots.setdefault(category, {"none_n": 0, "none_wrong": 0, "exposed_n": 0, "exposed_wrong": 0})
        side = "exposed" if row.get("B_wrong_direct_parent") else "none"
        slot[f"{side}_n"] += 1
        if row.get("transition") == "转错":
            slot[f"{side}_wrong"] += 1
    table = []
    for category, slot in slots.items():
        none_rate = _rate(slot["none_wrong"], slot["none_n"])
        exposed_rate = _rate(slot["exposed_wrong"], slot["exposed_n"])
        gap = None if none_rate is None or exposed_rate is None else round(exposed_rate - none_rate, 4)
        table.append({
            "category": category,
            "label": _CATEGORY_LABEL.get(category, category),
            **slot,
            "none_rate": none_rate,
            "exposed_rate": exposed_rate,
            "gap": gap,
        })
    return table


def _common_a_correct(pair_rows: list[dict]) -> list[dict]:
    by_group: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in pair_rows:
        if row.get("transition") == "缺失":
            continue
        by_group[row["group"]][row["target_id"]] = row
    found = []
    for hetero, base in MATCHED.items():
        left = by_group[hetero]
        right = by_group[base]
        common = [
            target for target in left
            if target in right and left[target].get("A_correct") and right[target].get("A_correct")
        ]

        def _side(table: dict[str, dict]) -> dict:
            wrong = sum(table[target]["transition"] == "转错" for target in common)
            return {"to_wrong": wrong, "to_wrong_rate": _rate(wrong, len(common))}

        hetero_side = _side(left)
        base_side = _side(right)
        gap = None
        if hetero_side["to_wrong_rate"] is not None and base_side["to_wrong_rate"] is not None:
            gap = round(hetero_side["to_wrong_rate"] - base_side["to_wrong_rate"], 4)
        found.append({
            "group": hetero,
            "label": LABELS[hetero],
            "standard": base,
            "standard_label": LABELS[base],
            "common_a_correct": len(common),
            "group_to_wrong": hetero_side["to_wrong"],
            "group_rate": hetero_side["to_wrong_rate"],
            "standard_to_wrong": base_side["to_wrong"],
            "standard_rate": base_side["to_wrong_rate"],
            "rate_gap": gap,
        })
    return found


def transition_analysis(rows: list[dict], labeled_b: dict[str, dict[str, dict]]) -> dict:
    audit = audit_rows(rows)
    located: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        if row.get("is_wrong") not in (True, False):
            continue
        located[(row["sample_id"], row["target_id"], row["injection"])] = row
    samples = sorted({row["sample_id"] for row in rows if row.get("injection") in {"A", "B"}})
    prefix_at = _prefix_location(set(samples))
    pair_rows = []
    withheld_edges = 0
    group_tables = []
    category: dict[str, dict] = {}
    parent_state: dict[str, dict] = {}
    by_group_category: dict[str, dict] = {}
    by_group_parent: dict[str, dict] = {}
    for sample in samples:
        targets = sorted(
            {target for (sid, target, _side) in located if sid == sample},
            key=_id_key,
        )
        group = next(row["group"] for row in rows if row.get("sample_id") == sample)
        cells = {"保持正确": 0, "转错": 0, "转对": 0, "持续错误": 0}
        b_labeled = labeled_b[group]
        parents = {node: tuple(item["logical_parents"]) for node, item in b_labeled.items()}
        wrong_b = {node: item["is_wrong"] for node, item in b_labeled.items()}
        for target in targets:
            left = located.get((sample, target, "A"))
            right = located.get((sample, target, "B"))
            if left is None or right is None:
                pair_rows.append({
                    "sample_id": sample,
                    "group": group,
                    "target_id": target,
                    "transition": "缺失",
                    "A_present": left is not None,
                    "B_present": right is not None,
                })
                continue
            a_wrong = bool(left["is_wrong"])
            b_wrong = bool(right["is_wrong"])
            kind = _TRANSITION[(a_wrong, b_wrong)]
            cells[kind] += 1
            b_item = b_labeled.get(target, {})
            direct, withheld = _exposed_wrong_parents(
                target, parents.get(target, ()), wrong_b, prefix_at[sample],
            )
            withheld_edges += len(withheld)
            ancestors = _wrong_ancestors(target, wrong_b, parents)
            record = {
                "sample_id": sample,
                "group": group,
                "target_id": target,
                "A_correct": not a_wrong,
                "B_correct": not b_wrong,
                "transition": kind,
                "error_category": right.get("error_category"),
                "process": right.get("process"),
                "B_wrong_direct_parent": bool(direct),
                "B_wrong_direct_parent_count": len(direct),
                "B_wrong_parent_not_injected": len(withheld),
                "B_wrong_ancestor": bool(ancestors),
                "B_error_type": b_item.get("error_type"),
            }
            pair_rows.append(record)
            if not a_wrong:
                cat = category.setdefault(record["error_category"], {"A_correct": 0, "to_wrong": 0})
                parent_key = "已注入的错误父节点" if direct else "没有已注入的错误父节点"
                parent_slot = parent_state.setdefault(parent_key, {"A_correct": 0, "to_wrong": 0})
                _accumulate(cat, a_wrong, b_wrong)
                _accumulate(parent_slot, a_wrong, b_wrong)
                group_cat = by_group_category.setdefault(group, {})
                group_parent = by_group_parent.setdefault(group, {})
                _accumulate(group_cat.setdefault(record["error_category"], {"A_correct": 0, "A_wrong": 0, "to_wrong": 0}), a_wrong, b_wrong)
                _accumulate(group_parent.setdefault(parent_key, {"A_correct": 0, "A_wrong": 0, "to_wrong": 0}), a_wrong, b_wrong)
        n11 = cells["保持正确"]
        n10 = cells["转错"]
        n01 = cells["转对"]
        n00 = cells["持续错误"]
        paired_n = n11 + n10 + n01 + n00
        group_tables.append({
            "group": group,
            "label": LABELS.get(group, group),
            "sample_id": sample,
            "n11_stay_correct": n11,
            "n10_to_wrong": n10,
            "n01_to_correct": n01,
            "n00_stay_wrong": n00,
            "paired": paired_n,
            "to_wrong_rate": _rate(n10, n11 + n10),
            "to_correct_rate": _rate(n01, n01 + n00),
            "net_loss": n10 - n01,
            "net_loss_rate": _rate(n10 - n01, paired_n),
            "mcnemar_p_descriptive": paired(
                {row["target_id"]: {"is_wrong": row["is_wrong"], "error_type": "local_error"} for row in rows if row.get("sample_id") == sample and row.get("injection") == "A" and row.get("is_wrong") in (True, False)},
                {row["target_id"]: {"is_wrong": row["is_wrong"], "error_type": "local_error"} for row in rows if row.get("sample_id") == sample and row.get("injection") == "B" and row.get("is_wrong") in (True, False)},
            )["mcnemar_p"],
        })
    def _rates(found: dict) -> list[dict]:
        rows_out = []
        for name, slot in found.items():
            rows_out.append({
                "name": name,
                "A_correct": slot["A_correct"],
                "to_wrong": slot["to_wrong"],
                "to_wrong_rate": _rate(slot["to_wrong"], slot["A_correct"]),
            })
        return rows_out
    group_tables.sort(key=lambda item: GROUPS.index(item["group"]) if item["group"] in GROUPS else 99)
    return {
        "audit": audit,
        "pairs": pair_rows,
        "groups": group_tables,
        "by_category": sorted(
            _rates(category),
            key=lambda item: ("semantic", "matrix", "sign", "expression", "vector").index(item["name"])
            if item["name"] in {"semantic", "matrix", "sign", "expression", "vector"} else 99,
        ),
        "by_parent": sorted(_rates(parent_state), key=lambda item: item["name"] != "没有已注入的错误父节点"),
        "by_category_exposure": _category_exposure(pair_rows),
        "common_a_correct": _common_a_correct(pair_rows),
        "withheld_wrong_parent_edges": withheld_edges,
        "by_group_category": {group: _rates(found) for group, found in by_group_category.items()},
        "by_group_parent": {group: _rates(found) for group, found in by_group_parent.items()},
        "note": "转对率是 A 错且 B 对。B 恢复率是同一次 B 运行里上游有错但当前题答对。父节点暴露只计算同一子过程中、当前句之前已经注入的错误答案。propagated_error 只表示该错误节点有错误上游。",
    }


def _pct(rate: float | None) -> str:
    if rate is None:
        return "—"
    from decimal import Decimal, ROUND_HALF_UP

    value = (Decimal(str(rate)) * Decimal(100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{value}%"


def write_report(transition: dict, path: Path) -> None:
    groups = transition["groups"]
    n11 = sum(item["n11_stay_correct"] for item in groups)
    n10 = sum(item["n10_to_wrong"] for item in groups)
    n01 = sum(item["n01_to_correct"] for item in groups)
    n00 = sum(item["n00_stay_wrong"] for item in groups)
    paired_n = n11 + n10 + n01 + n00
    a_correct = n11 + n10
    a_wrong = n01 + n00
    parent = {item["name"]: item for item in transition["by_parent"]}
    exposed = parent.get("已注入的错误父节点", {})
    clear = parent.get("没有已注入的错误父节点", {})
    exposed_still_correct = exposed.get("A_correct", 0) - exposed.get("to_wrong", 0)
    lines = [
        "# 关闭思考实验的逐题配对转移",
        "",
        "deepseek-flash，思考关闭，温度 0，无标记稿。A 是标准前缀、句号截断；B 是模型前缀、句号截断。配对键是样本编号加题号。下表的题次把七份定义稿叠在一起，不是 176 道独立题。",
        "",
        "父节点暴露只计入同一子过程里、当前句子之前已经注入的错误答案。同一句同时生成的答案，以及还没有出现在前缀里的逻辑父节点，不计为暴露。",
        "",
        f"配对检查：记录 {transition['audit']['rows']} 条，{'配对完整' if transition['audit']['complete'] else '配对不完整'}。未注入前缀的错误父节点边 {transition['withheld_wrong_parent_edges']} 条。",
        "",
        "## 四格合计",
        "",
        "| 配对结果 | 题次 |",
        "| --- | ---: |",
        f"| 保持正确 | {n11} |",
        f"| A 对、B 错 | {n10} |",
        f"| A 错、B 对 | {n01} |",
        f"| 持续错误 | {n00} |",
        "",
        f"总体转错率 {n10}/{a_correct} = {_pct(_rate(n10, a_correct))}。总体转对率 {n01}/{a_wrong} = {_pct(_rate(n01, a_wrong))}。总体净损失 {(n10 - n01)}/{paired_n} = {_pct(_rate(n10 - n01, paired_n))}。",
        "",
        f"A 中能够做对的题次里，有 {n10} 个在 B 中失败。同时有 {n01} 个题次是 A 错、B 对。这些转对不能直接解释成模型前缀有帮助，它们可能来自上下文、答案抵消或生成波动。净损失会隐藏这部分反向变化。",
        "",
        "## 各定义稿",
        "",
        "| 定义稿 | 保持正确 | 转错 | 转对 | 持续错误 | 转错率 | 转对率 | 净损失 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in groups:
        lines.append(
            f"| {item['label']} | {item['n11_stay_correct']} | {item['n10_to_wrong']} | "
            f"{item['n01_to_correct']} | {item['n00_stay_wrong']} | {_pct(item['to_wrong_rate'])} | "
            f"{_pct(item['to_correct_rate'])} | {item['net_loss']} |"
        )
    lines.extend([
        "",
        "体轴系手性的转错率接近统一默认。它的净损失更小，主要来自更多转对，而不是更少转错。因此不能把它解释为对模型前缀更稳健。",
        "",
        "正装反装的转错率高于统一默认，较大的净损失伴随着更多本来能做对、换成模型前缀后失败的题。机构滚转的转错率也较高，同时 A 本身错误较多，局部弱点不排除前缀敏感。",
        "",
        "各组转错率的分母是本组 A 做对的题，题目构成不完全相同。下表只保留异构稿和它的标准参照在 A 中都做对的题号。",
        "",
        "## 两组 A 都正确的共同题号",
        "",
        "| 定义稿 | 参照 | 共同题数 | 本组转错 | 本组转错率 | 参照转错 | 参照转错率 | 差值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in transition["common_a_correct"]:
        lines.append(
            f"| {item['label']} | {item['standard_label']} | {item['common_a_correct']} | "
            f"{item['group_to_wrong']} | {_pct(item['group_rate'])} | {item['standard_to_wrong']} | "
            f"{_pct(item['standard_rate'])} | {_pct(item['rate_gap'])} |"
        )
    lines.extend([
        "",
        "共同题号上，正装反装的转错率仍然高于统一默认。机构滚转也仍然更高。理论迎角和体轴系手性与统一默认的差距很小。机构角顺序在共同题号上的转错率低于球腕顺序。",
        "",
        "## 节点类别",
        "",
        "分母是 A 正确题次。公式和矩阵的转错率高于语义。",
        "",
        "| 节点类别 | A 正确题次 | B 转错 | 转错率 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for item in transition["by_category"]:
        lines.append(
            f"| {_CATEGORY_LABEL.get(item['name'], item['name'])} | {item['A_correct']} | "
            f"{item['to_wrong']} | {_pct(item['to_wrong_rate'])} |"
        )
    lines.extend([
        "",
        "## 已注入前缀的错误父节点",
        "",
        "| B 中已注入的直接父节点 | A 正确题次 | B 转错 | 转错率 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ("没有已注入的错误父节点", "已注入的错误父节点"):
        item = parent.get(name, {"A_correct": 0, "to_wrong": 0, "to_wrong_rate": None})
        lines.append(f"| {name} | {item['A_correct']} | {item['to_wrong']} | {_pct(item['to_wrong_rate'])} |")
    exposed_n = exposed.get("A_correct", 0)
    exposed_wrong = exposed.get("to_wrong", 0)
    clear_wrong = clear.get("to_wrong", 0)
    lines.extend([
        "",
        f"转错题次中，{exposed_wrong} 个有已注入的错误直接父节点，{clear_wrong} 个没有。前者是有错误父节点暴露的转错占比，不是传播造成的比例。没有错误直接父节点，也不等于整个前缀与 A 相同。",
        "",
        f"已注入错误父节点的 {exposed_n} 个题次里，{exposed_still_correct} 个在 B 中仍然正确。错误父节点并不必然导致当前失败。",
        "",
        "## 节点类别 × 父节点暴露",
        "",
        "只使用 A 正确题次。差值是有错误父节点的转错率减去没有错误父节点的转错率。",
        "",
        "| 节点类别 | 无错误父节点题次 | 无错误父节点转错率 | 有错误父节点题次 | 有错误父节点转错率 | 差值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for item in transition["by_category_exposure"]:
        lines.append(
            f"| {item['label']} | {item['none_n']} | {_pct(item['none_rate'])} | "
            f"{item['exposed_n']} | {_pct(item['exposed_rate'])} | {_pct(item['gap'])} |"
        )
    lines.extend([
        "",
        _cross_reading(transition["by_category_exposure"]),
        "",
        "## 可写入的描述",
        "",
        (
            f"在七份定义稿的 {a_correct} 个 gold 前缀正确题次中，{n10} 个在模型前缀下失败，"
            f"转错率为 {_pct(_rate(n10, a_correct))}。"
            + _category_sentence(transition["by_category"])
            + _parent_sentence(clear, exposed)
            + "这些结果表明，模型前缀下的性能损失集中于公式与矩阵应用，并与父节点错误暴露相关；该关联仍需结合节点类别分层和修复干预进一步解释。"
        ),
        "",
        "节点级 McNemar 的 p 值只随四格表保存，不用于给定义稿贴显著标签。题与题来自同一依赖流程，并不独立。",
        "",
        "恢复思考后沿用同一套表。需要分开看两件事：思考是否减少上游错误暴露，以及在已经暴露于错误父节点之后，失败概率是否下降。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _cross_reading(rows: list[dict]) -> str:
    found = {item["category"]: item for item in rows}
    formula = found["expression"]
    matrix = found["matrix"]
    semantic = found["semantic"]
    return (
        f"公式节点在没有已注入错误父节点时转错率为 {_pct(formula['none_rate'])}，"
        f"有错误父节点时为 {_pct(formula['exposed_rate'])}。"
        f"矩阵节点分别是 {_pct(matrix['none_rate'])} 和 {_pct(matrix['exposed_rate'])}。"
        "这两类的总体转错率高，既因为它们更常处在错误父节点之后，也因为同样暴露时失败更多。"
        f"语义节点在有错误父节点时的转错率是 {_pct(semantic['exposed_rate'])}，"
        "其转错主要出现在没有错误直接父节点的题次。"
        "若某一类在没有错误父节点时转错率仍然高，后续审阅应检查非父节点前缀、遗漏依赖和表达形式。"
    )


def _category_sentence(rows: list[dict]) -> str:
    found = {item["name"]: item for item in rows}
    formula = found.get("expression", {})
    matrix = found.get("matrix", {})
    return (
        f"公式和矩阵节点的转错率分别为 {_pct(formula.get('to_wrong_rate'))} 和 {_pct(matrix.get('to_wrong_rate'))}。"
    )


def _parent_sentence(clear: dict, exposed: dict) -> str:
    return (
        f"存在已注入的错误直接父节点时，转错率为 {_pct(exposed.get('to_wrong_rate'))}，"
        f"高于没有已注入错误直接父节点时的 {_pct(clear.get('to_wrong_rate'))}。"
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def main() -> None:
    rows = load_rows()
    by_run: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_run[(row["group"], row["injection"])].append(row)
    labeled_runs = {}
    metrics = {}
    survival_by_group = {}
    for group in GROUPS:
        for injection in ("A", "B"):
            labeled = label_run(by_run[(group, injection)], gold_prefix=(injection == "A"))
            labeled_runs[(group, injection)] = labeled
            metrics[(group, injection)] = run_metrics(labeled, gold_prefix=(injection == "A"))
        survival_by_group[group] = survival(labeled_runs[(group, "B")])
    pairs = {
        group: paired(labeled_runs[(group, "A")], labeled_runs[(group, "B")])
        for group in GROUPS
    }
    labeled_b = {group: labeled_runs[(group, "B")] for group in GROUPS}
    transition = transition_analysis(rows, labeled_b)
    flat = []
    for (group, injection), labeled in labeled_runs.items():
        for item in labeled.values():
            flat.append({
                "group": group,
                "self_prefix": 1.0 if injection == "B" else 0.0,
                "heterogeneity": 0.0 if group in STANDARDS else 1.0,
                "depth": float(item["propagation_depth"]),
                "parent_wrong": 1.0 if item["parent_wrong"] else 0.0,
                "parent_exposed": 1.0 if item["parent_wrong"] and injection == "B" else 0.0,
                "matrix": 1.0 if item["error_category"] == "matrix" else 0.0,
                "is_wrong": 1.0 if item["is_wrong"] else 0.0,
            })
    payload = {
        "source": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "scope": "关闭思考、deepseek-flash、温度 0、无标记稿、A=gold-trunc、B=self-trunc。",
        "claim_limit": "结果只描述这一设置下的观测。思考开关并未在其余因素保持不变时单独比较。",
        "causal_status": "first_error_nodes 是依赖图上的归因候选。有错误上游的错误节点不是已被证明的因果传播。I(v) 尚未估计。",
        "dependence_note": "176 题来自同一依赖流程，题与题不独立。McNemar 的 p 值随四格表保存，不作为确认性证据，也不用于给定义稿贴显著标签。",
        "groups": [],
        "transitions": transition["groups"],
        "transition_by_category": transition["by_category"],
        "transition_by_parent": transition["by_parent"],
        "transition_by_category_exposure": transition["by_category_exposure"],
        "common_a_correct": transition["common_a_correct"],
        "withheld_wrong_parent_edges": transition["withheld_wrong_parent_edges"],
        "transition_audit": transition["audit"],
        "strata": _stratum(labeled_runs),
        "regression": regression(flat),
        "intervention_candidates": intervention_candidates(labeled_b)[:25],
    }
    for group in GROUPS:
        payload["groups"].append({
            "group": group,
            "label": LABELS[group],
            "matched_standard": MATCHED.get(group),
            "A": metrics[(group, "A")],
            "B": metrics[(group, "B")],
            "pair": pairs[group],
            "survival": survival_by_group[group],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "transitions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in transition["pairs"]),
        encoding="utf-8",
    )
    write_report(transition, OUT / "report.md")
    print(OUT / "summary.json")
    print(OUT / "transitions.jsonl")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
