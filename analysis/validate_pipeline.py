# -*- coding: utf-8 -*-
"""Four-domain lightweight acceptance. Read-only.

One representative manuscript per domain. No model call and no session fork.
Layer 0 is the gold regression. Layer 1 is the signature plus the composer
exposure audit. Layer 2 is one synthetic answer sheet through the conflict
explainer. Layer 3 is the repair kernels and the three rankings.

Wind repair search reads a stored self-trunc observation file. When that file
is absent, layer 3 reads the already written repair table and does not search
again. The other three domains call ``minimal_fixes`` directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIND_SAMPLE = "body-rule"
_OPEN = {"≥3", "单修失败", "全上游仍失败"}
_NAMED_GRADES = {"单因素", "无上游", "单修失败", "全上游仍失败", "≥3"}


def _known_grade(grade: str) -> bool:
    if grade in _NAMED_GRADES:
        return True
    head = grade[:-1]
    return grade.endswith("元") and head.isdigit() and int(head) >= 2
_SWAPS = {
    "正": "负", "负": "正",
    "arcsin": "arctan", "arctan": "arcsin",
    "相同": "相反", "相反": "相同",
}


def _block_count(summary: dict) -> int:
    blocks = summary.get("e1_blocks", ())
    if isinstance(blocks, int):
        return blocks
    return len(blocks)


def _exposure(module) -> tuple[int, int, int]:
    """Sum latent, alias, and unfilled composers. ``check`` is the gate."""
    if hasattr(module, "audit"):
        rows = module.audit()
        latent = sum(int(row["latent"] or 0) for row in rows)
        alias = sum(int(row["alias"] or 0) for row in rows)
        pending = sum(1 for row in rows if row["status"] == "未填")
        if hasattr(module, "check"):
            module.check()
        return latent, alias, pending
    module.check()
    latent = sum(module.latent(table) for table in module.FACTORS.values())
    alias = sum(
        module.alias(table, module.RENDER[name]) for name, table in module.FACTORS.items()
    )
    pending = len(getattr(module, "PENDING", ()))
    return latent, alias, pending


def _exposure_cell(latent: int, alias: int, pending: int) -> str:
    text = f"{latent}/{alias}"
    marks = []
    if latent:
        marks.append("待修")
    if alias:
        marks.append("碰撞")
    if pending:
        marks.append(f"未填{pending}")
    if marks:
        text += " " + " ".join(marks)
    return text


def _link_counts(report: dict) -> tuple[int, int, set[str], float]:
    states = set(report.get("states", {}).values())
    correct = int(report.get("correct_count") or 0)
    if correct:
        states.add("correct")
    readings = report.get("readings") or []
    scored = correct + len(report.get("states") or {})
    coverage = scored / len(readings) if readings else float(scored > 0)
    causal = sum(1 for row in report.get("chain_attributions") or [] if row.get("strength") == "因果")
    associative = sum(1 for row in report.get("chain_attributions") or [] if row.get("strength") == "关联")
    return causal, associative, states, coverage


def _faithful_sheet(view) -> dict[str, str]:
    """One sheet: a faithful chain from a declaration, plus one local slip outside it."""
    gold = dict(view.gold())
    children: dict[str, list[str]] = defaultdict(list)
    for node in view.node_ids:
        for parent in view.parents(node):
            children[parent].append(node)

    def other(node: str) -> str | None:
        options = view.question(node).get("options") or []
        return next((item for item in options if item != gold.get(node)), None)

    def cone(root: str) -> set[str]:
        seen: set[str] = set()
        pending = list(children[root])
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(children[node])
        return seen

    root = ""
    alt = ""
    reached: set[str] = set()
    for node in view.declaration_roots:
        choice = other(node)
        if not choice or not children[node]:
            continue
        trial = dict(gold)
        trial[node] = choice
        moved = False
        for child in children[node]:
            got = view.answer_under(child, trial)
            if got is not None and not view.equal(str(got), str(gold[child]), child):
                moved = True
                break
        if not moved:
            continue
        root, alt, reached = node, choice, cone(node)
        break
    if not root:
        raise RuntimeError(f"{view.name} 没有能带动后文的声明根")

    answers = dict(gold)
    answers[root] = alt
    order = [node for node in view.node_ids if node in reached]
    for _ in range(len(order) + 1):
        changed = False
        for node in order:
            got = view.answer_under(node, answers)
            if got is not None and got != answers.get(node):
                answers[node] = got
                changed = True
        if not changed:
            break

    for node in view.node_ids:
        if node in reached or node == root or node in view.declaration_roots:
            continue
        choice = other(node)
        if choice is None:
            continue
        answers[node] = choice
        break
    return answers


def _boards(orderings: dict[str, list[str]]) -> tuple[bool, str]:
    if any(not rows for rows in orderings.values()):
        return False, "空"
    heads = [rows[0] for rows in orderings.values()]
    return True, "分歧" if len(set(heads)) > 1 else "一致"


def _rank(faults: list[str], targets_of, depth_of, harm_of) -> tuple[bool, str]:
    if not faults:
        return False, "空"
    unique = sorted(set(faults), key=str)
    orderings = {
        "频率": sorted(unique, key=lambda node: (-targets_of(node), str(node))),
        "深度": sorted(unique, key=lambda node: (-depth_of(node), str(node))),
        "危害": sorted(unique, key=lambda node: (-harm_of(node), str(node))),
    }
    return _boards(orderings)


class DomainRow:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cells = {
            "金标": "—",
            "det": "—",
            "latent/alias": "—",
            "五态": "—",
            "覆盖率": "—",
            "因果/关联": "—",
            "核": "—",
            "三榜": "—",
        }
        self.ok = {key: False for key in ("0", "1", "2", "3")}
        self.notes: list[str] = []

    def fail(self, layer: str, exc: BaseException) -> None:
        self.notes.append(f"第{layer}层：{exc}")


def _wind_record(sample: str) -> dict:
    path = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == sample:
            return row
    raise RuntimeError(f"样本 {sample} 不在 pair/samples.jsonl")


def _wind_layer0(sample: str):
    from body_frame_model import BodyFrameModel
    from domains.wind_tunnel.contract_verifiers import check_verifiers
    from domains.wind_tunnel.slot_signature import manuscript_answers

    record = _wind_record(sample)
    model = BodyFrameModel().declare(record["spec"])
    gold = manuscript_answers(sample)
    check_verifiers(model, gold)
    return model, gold


def _seed_wind_library(sample: str, results) -> None:
    from domains.wind_tunnel.conflict_diagnosis import _LIBRARIES

    by_id = {}
    effects = {}
    injected = {}
    for result in results:
        fault = result.instance
        by_id[fault.id] = fault
        if result.det != 1:
            continue
        effects[fault.id] = frozenset(result.entries)
        injected[fault.id] = frozenset(node for node, _value in fault.injector)
    _LIBRARIES[sample] = (by_id, effects, injected)


def _wind_sheet(model, gold: dict[str, str]) -> dict[str, str]:
    from domains.wind_tunnel.contract_verifiers import derive
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    children: dict[str, list[str]] = defaultdict(list)
    for node, item in NODE_BY_ID.items():
        for parent in item.depends_on:
            children[parent].append(node)

    def cone(root: str) -> set[str]:
        seen: set[str] = set()
        pending = list(children[root])
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            pending.extend(children[node])
        return seen

    root = ""
    alt = ""
    reached: set[str] = set()
    for node, item in NODE_BY_ID.items():
        if item.depends_on:
            continue
        choice = _SWAPS.get(str(gold.get(node, "")).strip())
        if not choice:
            continue
        trial = dict(gold)
        trial[node] = choice
        moved = False
        for child in children[node]:
            got = derive(model, child, trial)
            if got is not None and got.ok and str(got.matrix) != str(gold.get(child, "")):
                moved = True
                break
        if not moved:
            continue
        root, alt, reached = node, choice, cone(node)
        break
    if not root:
        raise RuntimeError("风洞没有能带动后文的声明根")

    answers = dict(gold)
    answers[root] = alt
    order = sorted(reached, key=int)
    for _ in range(len(order) + 1):
        changed = False
        for node in order:
            got = derive(model, node, answers)
            if got is not None and got.ok and str(got.matrix) != str(answers.get(node, "")):
                answers[node] = str(got.matrix)
                changed = True
        if not changed:
            break
    for node, item in NODE_BY_ID.items():
        if not item.depends_on or node in reached:
            continue
        choice = _SWAPS.get(str(gold.get(node, "")).strip())
        if choice is None:
            continue
        answers[node] = choice
        break
    return answers


def _wind_observations(sample: str, gold: dict[str, str], answers: dict[str, str]):
    from domains.wind_tunnel.contract_labels import TARGETS
    from domains.wind_tunnel.error_graph import NODE_BY_ID
    from domains.wind_tunnel.error_workflow import Observation

    found = []
    for node, item in NODE_BY_ID.items():
        model_answer = str(answers.get(node, gold.get(node, "")))
        gold_answer = str(gold.get(node, ""))
        if not model_answer and not gold_answer:
            continue
        label, subtype = TARGETS.get(node, ("", ""))
        found.append(Observation(
            sample_id=sample,
            condition="synthetic",
            prefix="model",
            process=item.process,
            node_id=node,
            gold_answer=gold_answer,
            model_answer=model_answer,
            standard_answer=gold_answer,
            options=[],
            format_code=item.answer_type,
            parse_status="ok",
            correct=model_answer == gold_answer,
            error_kind="",
            label=label,
            subtype=subtype,
        ))
    return found


def _wind_kernels(sample: str) -> tuple[str, list[str], dict[str, list[str]], str]:
    folder = ROOT / "results" / "contract_consistency" / "error_workflow" / f"{sample}_self-trunc"
    path = folder / "observations.jsonl"
    if path.is_file():
        from domains.wind_tunnel.t_final import minimal_fixes

        fixes = minimal_fixes(sample)
        faults: list[str] = []
        targets: dict[str, list[str]] = defaultdict(list)
        open_count = 0
        for item in fixes:
            if not _known_grade(item.grade):
                raise RuntimeError(f"ID{item.target} 的级别无法归类：{item.grade}")
            if item.minimal and item.grade not in _OPEN:
                for group in item.minimal:
                    for fault in group:
                        faults.append(fault)
                        targets[fault].append(item.target)
            else:
                open_count += 1
            if item.direct:
                faults.append(item.target)
                targets[item.target].append(item.target)
        cell = f"{len(fixes) - open_count}核/{open_count}未解开"
        return cell, faults, targets, "现场"
    from analysis.rank_harm import _load_fixes

    rows = _load_fixes(ROOT / "analysis" / "CR最小修复集.md").get(sample) or {}
    if not rows:
        raise RuntimeError(f"{sample} 没有 self-trunc 观测，存档修复表里也没有这一份")
    faults = []
    targets = defaultdict(list)
    open_count = 0
    for target, row in rows.items():
        grade = row.get("grade", "")
        if not _known_grade(grade):
            raise RuntimeError(f"ID{target} 的存档级别无法归类：{grade}")
        groups = row.get("minimal") or ()
        if grade in _OPEN or not groups:
            open_count += 1
        for group in groups:
            for fault in group:
                faults.append(fault)
                targets[fault].append(target)
        if row.get("direct"):
            faults.append(target)
            targets[target].append(target)
    cell = f"{len(rows) - open_count}核/{open_count}未解开"
    return cell, faults, targets, "存档"


def _wind_depth(node: str) -> int:
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    item = NODE_BY_ID[node]
    reached = [target for target in ("80", "81", "111", "112", "113", "114", "115", "116", "176", "177", "178", "195", "196", "197", "198", "199", "200") if target == node or target in item.descendants]
    deepest = max((NODE_BY_ID[target].dependency_depth for target in reached), default=item.dependency_depth)
    return deepest - item.dependency_depth


def run_wind(sample: str = WIND_SAMPLE) -> DomainRow:
    row = DomainRow("wind_tunnel")
    print(f"风洞 {sample} 第0层", flush=True)
    try:
        model, gold = _wind_layer0(sample)
        row.cells["金标"] = "✓"
        row.ok["0"] = True
    except Exception as exc:
        row.cells["金标"] = "失败"
        row.fail("0", exc)
        return row

    print("风洞 第1层", flush=True)
    try:
        from analysis import audit_composers
        from domains.wind_tunnel.signature import evaluate, summarize

        results = evaluate(sample)
        summary = summarize(sample, results)
        _seed_wind_library(sample, results)
        blocks = _block_count(summary)
        row.cells["det"] = str(summary["det1"])
        latent, alias, pending = _exposure(audit_composers)
        row.cells["latent/alias"] = _exposure_cell(latent, alias, pending)
        row.ok["1"] = summary["det1"] > 0 and blocks > 0
        if not row.ok["1"]:
            row.notes.append(f"det={summary['det1']}，模糊组={blocks}")
    except Exception as exc:
        row.cells["det"] = "失败"
        row.fail("1", exc)
        return row

    print("风洞 第2层", flush=True)
    try:
        from domains.wind_tunnel.conflict_diagnosis import explain_observations

        answers = _wind_sheet(model, gold)
        report = explain_observations(model, _wind_observations(sample, gold, answers), gold)
        causal, associative, states, coverage = _link_counts(report)
        row.cells["五态"] = f"{len(states)}态"
        row.cells["覆盖率"] = f"{coverage:.2f}"
        row.cells["因果/关联"] = f"{causal}/{associative}"
        row.ok["2"] = bool(states) and coverage > 0 and causal > 0
    except Exception as exc:
        row.cells["五态"] = "失败"
        row.fail("2", exc)

    print("风洞 第3层", flush=True)
    try:
        from analysis.rank_harm import WEIGHT

        cell, faults, targets, source = _wind_kernels(sample)
        row.cells["核"] = cell
        ok, text = _rank(
            faults,
            lambda node: len(targets[node]),
            _wind_depth,
            lambda node: sum(WEIGHT.get(target, 1) for target in targets[node]),
        )
        row.cells["三榜"] = text
        row.ok["3"] = ok
        if source == "存档":
            row.notes.append("核读取已有修复表，未重跑搜索")
    except Exception as exc:
        row.cells["核"] = "失败"
        row.fail("3", exc)
    return row


def _view_depth(view, node: str, memo: dict[str, int]) -> int:
    if node in memo:
        return memo[node]
    parents = view.parents(node)
    memo[node] = 0 if not parents else 1 + max(_view_depth(view, parent, memo) for parent in parents)
    return memo[node]


def run_view(name: str, stage0, view_of, composers, fixes_of, targets) -> DomainRow:
    row = DomainRow(name)
    print(f"{name} 第0层", flush=True)
    try:
        stage0()
        row.cells["金标"] = "✓"
        row.ok["0"] = True
    except Exception as exc:
        row.cells["金标"] = "失败"
        row.fail("0", exc)
        return row

    print(f"{name} 第1层", flush=True)
    try:
        from domains.mbd_kernel import audit_stage4

        view = view_of()
        summary = audit_stage4(view)
        blocks = _block_count(summary)
        row.cells["det"] = str(summary["det1"])
        latent, alias, pending = _exposure(composers)
        row.cells["latent/alias"] = _exposure_cell(latent, alias, pending)
        row.ok["1"] = summary["det1"] > 0 and blocks > 0
        if not row.ok["1"]:
            row.notes.append(f"det={summary['det1']}，模糊组={blocks}")
    except Exception as exc:
        row.cells["det"] = "失败"
        row.fail("1", exc)
        return row

    print(f"{name} 第2层", flush=True)
    try:
        from domains.conflict_diagnosis import explain

        report = explain(view, _faithful_sheet(view), targets, prefix="model")
        causal, associative, states, coverage = _link_counts(report)
        row.cells["五态"] = f"{len(states)}态"
        row.cells["覆盖率"] = f"{coverage:.2f}"
        row.cells["因果/关联"] = f"{causal}/{associative}"
        row.ok["2"] = bool(states) and coverage > 0 and causal > 0
    except Exception as exc:
        row.cells["五态"] = "失败"
        row.fail("2", exc)

    print(f"{name} 第3层", flush=True)
    try:
        faults: list[str] = []
        targets_of: dict[str, list[str]] = defaultdict(list)
        open_count = 0
        for target in targets:
            groups = fixes_of(target)
            if not groups:
                open_count += 1
                continue
            for group in groups:
                for fault in group:
                    faults.append(fault)
                    targets_of[fault].append(target)
        row.cells["核"] = f"{len(targets) - open_count}核/{open_count}未解开"
        memo: dict[str, int] = {}
        ok, text = _rank(
            faults,
            lambda node: len(targets_of[node]),
            lambda node: _view_depth(view, node, memo),
            lambda node: float(len(targets_of[node])),
        )
        row.cells["三榜"] = text
        row.ok["3"] = ok and (faults or open_count == len(targets))
    except Exception as exc:
        row.cells["核"] = "失败"
        row.fail("3", exc)
    return row


def run_flight() -> DomainRow:
    from analysis import audit_composers_flight
    from domains.aerodynamics.aligned import view
    from domains.aerodynamics.signed_lists import T_FINAL
    from domains.aerodynamics.stage0_check import check_stage0
    from domains.aerodynamics.t_final import minimal_fixes

    return run_view("aerodynamics", check_stage0, view, audit_composers_flight, minimal_fixes, T_FINAL)


def _pair(text: str) -> tuple[int, int] | None:
    head = text.split()[0]
    if head.count("/") != 1:
        return None
    left, right = head.split("/")
    if left.isdigit() and right.isdigit():
        return int(left), int(right)
    return None


def _merge_domain(name: str, parts: list[DomainRow]) -> DomainRow:
    """One domain row. The parts are sections of the same dataset."""
    row = DomainRow(name)
    for layer in ("0", "1", "2", "3"):
        row.ok[layer] = all(part.ok[layer] for part in parts)
    row.cells["金标"] = "✓" if row.ok["0"] else "失败"
    dets = [int(part.cells["det"]) for part in parts if part.cells["det"].isdigit()]
    row.cells["det"] = str(sum(dets)) if len(dets) == len(parts) else "失败"
    exposures = [_pair(part.cells["latent/alias"]) for part in parts]
    if all(item is not None for item in exposures):
        latent = sum(item[0] for item in exposures)
        alias = sum(item[1] for item in exposures)
        pending = 0
        for part in parts:
            for token in part.cells["latent/alias"].split():
                if token.startswith("未填") and token.removeprefix("未填").isdigit():
                    pending += int(token.removeprefix("未填"))
        row.cells["latent/alias"] = _exposure_cell(latent, alias, pending)
    links = [_pair(part.cells["因果/关联"]) for part in parts]
    if all(item is not None for item in links):
        row.cells["因果/关联"] = f"{sum(item[0] for item in links)}/{sum(item[1] for item in links)}"
    kernels = []
    for part in parts:
        cell = part.cells["核"]
        if "核/" not in cell or "未解开" not in cell:
            kernels = []
            break
        left, right = cell.split("核/", 1)
        right = right.removesuffix("未解开")
        if left.isdigit() and right.isdigit():
            kernels.append((int(left), int(right)))
        else:
            kernels = []
            break
    if kernels:
        row.cells["核"] = f"{sum(item[0] for item in kernels)}核/{sum(item[1] for item in kernels)}未解开"
    boards = [part.cells["三榜"] for part in parts]
    if all(part.ok["3"] for part in parts):
        row.cells["三榜"] = "一致" if set(boards) == {"一致"} else "分歧"
    row.cells["五态"] = "通过" if row.ok["2"] else "失败"
    coverages = []
    for part in parts:
        try:
            coverages.append(float(part.cells["覆盖率"]))
        except ValueError:
            coverages = []
            break
    if coverages:
        row.cells["覆盖率"] = f"{min(coverages):.2f}"
    for part in parts:
        row.notes.extend(part.notes)
    return row


def run_vision() -> DomainRow:
    from analysis import audit_composers_scene, audit_composers_vision
    from domains.computer_vision.aligned import view as theory_view
    from domains.computer_vision.scene_aligned import view as scene_view
    from domains.computer_vision.scene_signed_lists import T_FINAL as SCENE_FINAL
    from domains.computer_vision.scene_stage0_check import check_stage0 as check_scene
    from domains.computer_vision.scene_t_final import minimal_fixes as scene_fixes
    from domains.computer_vision.signed_lists import T_FINAL
    from domains.computer_vision.stage0_check import check_stage0
    from domains.computer_vision.t_final import minimal_fixes

    theory = run_view(
        "机器视觉", check_stage0, theory_view, audit_composers_vision, minimal_fixes, T_FINAL,
    )
    scene = run_view(
        "机器视觉", check_scene, scene_view, audit_composers_scene, scene_fixes, SCENE_FINAL,
    )
    return _merge_domain("computer_vision", [theory, scene])


def run_kinematics() -> DomainRow:
    from analysis import audit_composers_kinematics
    from domains.robotics.kin_aligned import view
    from domains.robotics.kin_signed_lists import T_FINAL
    from domains.robotics.kin_stage0_check import check_stage0
    from domains.robotics.kin_t_final import minimal_fixes

    return run_view("robotics", check_stage0, view, audit_composers_kinematics, minimal_fixes, T_FINAL)


_RUNNERS = {
    "wind_tunnel": run_wind,
    "aerodynamics": run_flight,
    "computer_vision": run_vision,
    "robotics": run_kinematics,
}


def _print_table(rows: list[DomainRow]) -> None:
    headers = ("域", "金标", "det", "latent/alias", "五态", "覆盖率", "因果/关联", "核", "三榜")
    keys = headers[1:]
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [row.name] + [row.cells[key] for key in keys]
        print("| " + " | ".join(values) + " |")
    print()
    passed = all(all(row.ok.values()) for row in rows)
    diagnosable = all(row.ok["1"] for row in rows)
    attributable = all(row.ok["2"] for row in rows)
    explained = all(row.ok["3"] for row in rows)
    feasible = diagnosable and attributable and explained
    print(f"跑通：{'是' if passed else '否'}")
    print(
        "方案可行："
        + ("是" if feasible else "否")
        + f"（可诊断性 {'是' if diagnosable else '否'}，"
        + f"可归因 {'是' if attributable else '否'}，"
        + f"可解释影响 {'是' if explained else '否'}）"
    )
    for row in rows:
        for note in row.notes:
            print(f"{row.name}：{note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="四域轻量验收。只读，不调用模型。")
    parser.add_argument("domains", nargs="*", choices=tuple(_RUNNERS), help="缺省时跑四个域")
    args = parser.parse_args(argv)
    names = args.domains or list(_RUNNERS)
    rows = [_RUNNERS[name]() for name in names]
    _print_table(rows)
    return 0 if all(all(row.ok.values()) for row in rows) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
