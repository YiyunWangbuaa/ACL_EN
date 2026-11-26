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
_OPEN = {"≥3", "single-repair-failed", "all-upstream-still-fail"}
_NAMED_GRADES = {
    "single-factor", "no-upstream", "single-repair-failed", "all-upstream-still-fail", "≥3",
}


def _known_grade(grade: str) -> bool:
    if grade in _NAMED_GRADES:
        return True
    if not grade.endswith("-factor"):
        return False
    head = grade[: -len("-factor")]
    return head.isdigit() and int(head) >= 2
_SWAPS = {
    "positive": "negative", "negative": "positive",
    "arcsin": "arctan", "arctan": "arcsin",
    "same": "opposite", "opposite": "same",
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
        pending = sum(1 for row in rows if row["status"] == "unfilled")
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
        marks.append("to-repair")
    if alias:
        marks.append("collision")
    if pending:
        marks.append(f"unfilled{pending}")
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
    causal = sum(1 for row in report.get("chain_attributions") or [] if row.get("strength") == "causal")
    associative = sum(1 for row in report.get("chain_attributions") or [] if row.get("strength") == "associative")
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
        raise RuntimeError(f"{view.name} has no declaration root that moves a later answer")

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
        return False, "empty"
    heads = [rows[0] for rows in orderings.values()]
    return True, "split" if len(set(heads)) > 1 else "agree"


def _rank(faults: list[str], targets_of, depth_of, harm_of) -> tuple[bool, str]:
    if not faults:
        return False, "empty"
    unique = sorted(set(faults), key=str)
    orderings = {
        "frequency": sorted(unique, key=lambda node: (-targets_of(node), str(node))),
        "depth": sorted(unique, key=lambda node: (-depth_of(node), str(node))),
        "harm": sorted(unique, key=lambda node: (-harm_of(node), str(node))),
    }
    return _boards(orderings)


class DomainRow:
    def __init__(self, name: str) -> None:
        self.name = name
        self.cells = {
            "gold": "—",
            "det": "—",
            "latent/alias": "—",
            "five-class": "—",
            "coverage": "—",
            "causal/associative": "—",
            "kernel": "—",
            "rankings": "—",
        }
        self.ok = {key: False for key in ("0", "1", "2", "3")}
        self.notes: list[str] = []

    def fail(self, layer: str, exc: BaseException) -> None:
        self.notes.append(f"layer {layer}: {exc}")


def _wind_record(sample: str) -> dict:
    path = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] == sample:
            return row
    raise RuntimeError(f"sample {sample} is not in pair/samples.jsonl")


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
        raise RuntimeError("wind tunnel has no declaration root that moves a later answer")

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
                raise RuntimeError(f"ID{item.target} grade cannot be classified: {item.grade}")
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
        cell = f"{len(fixes) - open_count} kernel/{open_count} unresolved"
        return cell, faults, targets, "live"
    from analysis.rank_harm import _load_fixes

    rows = _load_fixes(ROOT / "doc" / "CR_minimal_repair_sets.md").get(sample) or {}
    if not rows:
        raise RuntimeError(f"{sample} has no self-trunc observation, and the archived repair table has no copy")
    faults = []
    targets = defaultdict(list)
    open_count = 0
    for target, row in rows.items():
        grade = row.get("grade", "")
        if not _known_grade(grade):
            raise RuntimeError(f"ID{target} archived grade cannot be classified: {grade}")
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
    cell = f"{len(rows) - open_count} kernel/{open_count} unresolved"
    return cell, faults, targets, "archive"


def _wind_depth(node: str) -> int:
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    item = NODE_BY_ID[node]
    reached = [target for target in ("80", "81", "111", "112", "113", "114", "115", "116", "176", "177", "178", "195", "196", "197", "198", "199", "200") if target == node or target in item.descendants]
    deepest = max((NODE_BY_ID[target].dependency_depth for target in reached), default=item.dependency_depth)
    return deepest - item.dependency_depth


def run_wind(sample: str = WIND_SAMPLE) -> DomainRow:
    row = DomainRow("wind_tunnel")
    print(f"wind tunnel {sample} layer 0", flush=True)
    try:
        model, gold = _wind_layer0(sample)
        row.cells["gold"] = "✓"
        row.ok["0"] = True
    except Exception as exc:
        row.cells["gold"] = "fail"
        row.fail("0", exc)
        return row

    print("wind tunnel layer 1", flush=True)
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
            row.notes.append(f"det={summary['det1']}, ambiguity groups={blocks}")
    except Exception as exc:
        row.cells["det"] = "fail"
        row.fail("1", exc)
        return row

    print("wind tunnel layer 2", flush=True)
    try:
        from domains.wind_tunnel.conflict_diagnosis import explain_observations

        answers = _wind_sheet(model, gold)
        report = explain_observations(model, _wind_observations(sample, gold, answers), gold)
        causal, associative, states, coverage = _link_counts(report)
        row.cells["five-class"] = f"{len(states)} states"
        row.cells["coverage"] = f"{coverage:.2f}"
        row.cells["causal/associative"] = f"{causal}/{associative}"
        row.ok["2"] = bool(states) and coverage > 0 and causal > 0
    except Exception as exc:
        row.cells["five-class"] = "fail"
        row.fail("2", exc)

    print("wind tunnel layer 3", flush=True)
    try:
        from analysis.rank_harm import WEIGHT

        cell, faults, targets, source = _wind_kernels(sample)
        row.cells["kernel"] = cell
        ok, text = _rank(
            faults,
            lambda node: len(targets[node]),
            _wind_depth,
            lambda node: sum(WEIGHT.get(target, 1) for target in targets[node]),
        )
        row.cells["rankings"] = text
        row.ok["3"] = ok
        if source == "archive":
            row.notes.append("kernel read the existing repair table and did not rerun the search")
    except Exception as exc:
        row.cells["kernel"] = "fail"
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
    print(f"{name} layer 0", flush=True)
    try:
        stage0()
        row.cells["gold"] = "✓"
        row.ok["0"] = True
    except Exception as exc:
        row.cells["gold"] = "fail"
        row.fail("0", exc)
        return row

    print(f"{name} layer 1", flush=True)
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
            row.notes.append(f"det={summary['det1']}, ambiguity groups={blocks}")
    except Exception as exc:
        row.cells["det"] = "fail"
        row.fail("1", exc)
        return row

    print(f"{name} layer 2", flush=True)
    try:
        from domains.conflict_diagnosis import explain

        report = explain(view, _faithful_sheet(view), targets, prefix="model")
        causal, associative, states, coverage = _link_counts(report)
        row.cells["five-class"] = f"{len(states)} states"
        row.cells["coverage"] = f"{coverage:.2f}"
        row.cells["causal/associative"] = f"{causal}/{associative}"
        row.ok["2"] = bool(states) and coverage > 0 and causal > 0
    except Exception as exc:
        row.cells["five-class"] = "fail"
        row.fail("2", exc)

    print(f"{name} layer 3", flush=True)
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
        row.cells["kernel"] = f"{len(targets) - open_count} kernel/{open_count} unresolved"
        memo: dict[str, int] = {}
        ok, text = _rank(
            faults,
            lambda node: len(targets_of[node]),
            lambda node: _view_depth(view, node, memo),
            lambda node: float(len(targets_of[node])),
        )
        row.cells["rankings"] = text
        row.ok["3"] = ok and (faults or open_count == len(targets))
    except Exception as exc:
        row.cells["kernel"] = "fail"
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
    row.cells["gold"] = "✓" if row.ok["0"] else "fail"
    dets = [int(part.cells["det"]) for part in parts if part.cells["det"].isdigit()]
    row.cells["det"] = str(sum(dets)) if len(dets) == len(parts) else "fail"
    exposures = [_pair(part.cells["latent/alias"]) for part in parts]
    if all(item is not None for item in exposures):
        latent = sum(item[0] for item in exposures)
        alias = sum(item[1] for item in exposures)
        pending = 0
        for part in parts:
            for token in part.cells["latent/alias"].split():
                if token.startswith("unfilled") and token.removeprefix("unfilled").isdigit():
                    pending += int(token.removeprefix("unfilled"))
        row.cells["latent/alias"] = _exposure_cell(latent, alias, pending)
    links = [_pair(part.cells["causal/associative"]) for part in parts]
    if all(item is not None for item in links):
        row.cells["causal/associative"] = f"{sum(item[0] for item in links)}/{sum(item[1] for item in links)}"
    kernels = []
    for part in parts:
        cell = part.cells["kernel"]
        if " kernel/" not in cell or "unresolved" not in cell:
            kernels = []
            break
        left, right = cell.split(" kernel/", 1)
        right = right.removesuffix("unresolved").strip()
        if left.isdigit() and right.isdigit():
            kernels.append((int(left), int(right)))
        else:
            kernels = []
            break
    if kernels:
        row.cells["kernel"] = f"{sum(item[0] for item in kernels)} kernel/{sum(item[1] for item in kernels)} unresolved"
    boards = [part.cells["rankings"] for part in parts]
    if all(part.ok["3"] for part in parts):
        row.cells["rankings"] = "agree" if set(boards) == {"agree"} else "split"
    row.cells["five-class"] = "pass" if row.ok["2"] else "fail"
    coverages = []
    for part in parts:
        try:
            coverages.append(float(part.cells["coverage"]))
        except ValueError:
            coverages = []
            break
    if coverages:
        row.cells["coverage"] = f"{min(coverages):.2f}"
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
        "computer vision", check_stage0, theory_view, audit_composers_vision, minimal_fixes, T_FINAL,
    )
    scene = run_view(
        "computer vision", check_scene, scene_view, audit_composers_scene, scene_fixes, SCENE_FINAL,
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
    headers = ("domain", "gold", "det", "latent/alias", "five-class", "coverage", "causal/associative", "kernel", "rankings")
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
    print(f"passed: {'yes' if passed else 'no'}")
    print(
        "plan feasible: "
        + ("yes" if feasible else "no")
        + f" (diagnosable {'yes' if diagnosable else 'no'}, "
        + f"attributable {'yes' if attributable else 'no'}, "
        + f"explained effect {'yes' if explained else 'no'})"
    )
    for row in rows:
        for note in row.notes:
            print(f"{row.name}: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Four-domain lightweight acceptance. Read-only. Does not call a model.")
    parser.add_argument("domains", nargs="*", choices=tuple(_RUNNERS), help="runs all four domains when omitted")
    args = parser.parse_args(argv)
    names = args.domains or list(_RUNNERS)
    rows = [_RUNNERS[name]() for name in names]
    _print_table(rows)
    return 0 if all(all(row.ok.values()) for row in rows) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
