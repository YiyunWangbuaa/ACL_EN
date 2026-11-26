# -*- coding: utf-8 -*-
"""Binary fault signature matrix for one manuscript.

D[f, m] = 1 exactly when m is in the fault's graph reach (FaultInstance.potential).
The matrix does not inject a wrong answer and does not record predicted values.
Diagnosis recomputes a real observation with the local rule; that prediction is
not this matrix.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RES, RSN

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass

from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.fault_instance import (
    SIGN_COMPONENTS,
    FaultInstance,
    e1_census,
    e1_instances,
    instances,
    outside,
)
from domains.wind_tunnel.multi_slot import changed_definitions, changed_reads

# Half of the "one binary column, one block of 62" baseline. A block this
# large means the reach sets did not separate the recognition rows.
_BLOCK_LIMIT = 20


@dataclass(frozen=True)
class InstanceResult:
    instance: FaultInstance
    det: int
    reason: str
    entries: tuple[str, ...]


@dataclass(frozen=True)
class Report:
    sample: str
    results: tuple[InstanceResult, ...]
    digest: str
    summary: dict


def _support(item: FaultInstance) -> tuple[str, ...]:
    return tuple(sorted(item.potential))


def evaluate(sample_id: str) -> tuple[InstanceResult, ...]:
    """One binary row per library fault. The support is the graph reach."""
    found = []
    for item in instances(sample_id):
        support = _support(item)
        if support:
            found.append(InstanceResult(item, 1, "", support))
        else:
            found.append(InstanceResult(item, 0, "not injectable", ()))
    found.sort(key=lambda item: item.instance.id)
    return tuple(found)


def report(sample_id: str) -> Report:
    results = evaluate(sample_id)
    return Report(sample_id, results, table_hash(sample_id, results), summarize(sample_id, results))


def table_hash(sample_id: str, results: tuple[InstanceResult, ...]) -> str:
    payload = []
    for item in results:
        payload.append({
            "id": item.instance.id,
            "det": item.det,
            "reason": item.reason,
            "support": list(item.entries),
        })
    raw = json.dumps(
        {"sample": sample_id, "rows": payload},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def summarize(sample_id: str, results: tuple[InstanceResult, ...]) -> dict:
    counts = Counter(item.instance.klass for item in results)
    det1 = [item for item in results if item.det == 1]
    det0 = [item for item in results if item.det == 0]
    reasons = Counter(item.reason for item in det0)
    e1_det1 = [item for item in det1 if item.instance.klass == DIR]
    blocks = _blocks(e1_det1)
    largest = max(blocks, key=len) if blocks else ()
    reads = changed_reads(sample_id)
    probe_ids = [
        item.id for item in e1_instances(sample_id)
        if set(reads) & {blank for blank, _value in item.injector}
    ]
    detectable = [item.instance.id for item in e1_det1]
    return {
        "counts": dict(counts),
        "census": e1_census(sample_id),
        "changed": changed_definitions(sample_id),
        "changed_reads": reads,
        "changed_probes": probe_ids,
        "e1_detectable": detectable,
        "det1": len(det1),
        "det0": dict(reasons),
        "det0_ids": {reason: [item.instance.id for item in det0 if item.reason == reason] for reason in reasons},
        "e1_blocks": blocks,
        "e1_max_block": len(largest),
        "e1_max_reason": _why(largest),
        "e1_singletons": sum(len(block) == 1 for block in blocks),
        "isolation": _isolation(results),
        "outside": outside(),
    }


def audit(sample_ids: tuple[str, ...] = ("body-rule", "roll-rule")) -> tuple[Report, ...]:
    """The second build of each manuscript must hash the same."""
    reports = []
    for sample_id in sample_ids:
        first = report(sample_id)
        second = report(sample_id)
        _check(first)
        if first.digest != second.digest:
            raise RuntimeError(f"{sample_id} hash changed on the second build.")
        reports.append(first)
    return tuple(reports)


def _check(item: Report) -> None:
    sample = item.sample
    summary = item.summary
    seen = {result.instance.id for result in item.results}
    library = {row.id for row in instances(sample)}
    if seen != library:
        raise RuntimeError(f"{sample} support rows do not cover the instance library.")
    for result in item.results:
        if result.det not in (0, 1):
            raise RuntimeError(f"{result.instance.id} det is not 0 or 1.")
        if result.det == 0 and result.reason != "not injectable":
            raise RuntimeError(f"{result.instance.id} det=0 but the reason is {result.reason!r}.")
        if result.det == 1 and result.reason:
            raise RuntimeError(f"{result.instance.id} det=1 still carries a reason.")
        if result.det == 1 and result.entries != tuple(sorted(result.instance.potential)):
            raise RuntimeError(f"{result.instance.id} support is not the graph reach.")
        for mid in result.entries:
            if not mid:
                raise RuntimeError(f"{result.instance.id} support contains an empty measurement.")
    e1_rows = summary["census"]["rows"]
    if summary["e1_max_block"] > _BLOCK_LIMIT or summary["e1_max_block"] >= e1_rows:
        raise RuntimeError(
            f"{sample} largest DIR ambiguity group is {summary['e1_max_block']}, not clearly below the single-column baseline of 62."
        )
    if RES in summary["counts"]:
        raise RuntimeError(f"{sample} treats RES as a row.")
    math_rows = sum(
        1 for node_id, node in NODE_BY_ID.items()
        if node.answer_type in {"matrix", "vector", "expression"} and node_id not in SIGN_COMPONENTS
    )
    if summary["counts"].get(ALG, 0) != math_rows:
        raise RuntimeError(f"{sample} ALG should be {math_rows} rows, got {summary['counts'].get(ALG, 0)}.")


def _blocks(rows: list[InstanceResult]) -> tuple[tuple[str, ...], ...]:
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for item in rows:
        grouped[item.entries].append(item.instance.id)
    blocks = [tuple(sorted(ids)) for ids in grouped.values()]
    blocks.sort(key=lambda ids: (-len(ids), ids))
    return tuple(blocks)


def _why(block: tuple[str, ...]) -> str:
    if len(block) <= 1:
        return ""
    return "shared reach"


def _isolation(results: tuple[InstanceResult, ...]) -> dict[str, dict]:
    found = {}
    for klass in (DIR, RSN, EXE, ALG):
        rows = [item for item in results if item.instance.klass == klass and item.det == 1]
        blocks = _blocks(rows)
        singles = sum(len(block) == 1 for block in blocks)
        found[klass] = {
            "detectable": len(rows),
            "blocks": len(blocks),
            "singletons": singles,
            "max_block": max((len(block) for block in blocks), default=0),
        }
    return found
