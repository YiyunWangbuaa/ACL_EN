# -*- coding: utf-8 -*-
"""Conflict sets stay above the forward verifier. No model calls."""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.conflict_diagnosis import (  # noqa: E402
    assert_report,
    conflict_faults,
    explain,
    field_state,
    minimal_hitting_sets,
    upstream_faults,
)
from domains.phi_result import Contradiction, from_execution  # noqa: E402
from domains.mbd_kernel import DomainView  # noqa: E402
from domains.observation import annotate_trajectory, diagnose  # noqa: E402


def _view(name, nodes, parents, gold, questions, roots, rule, reference=None):
    def question(node):
        row = dict(questions[node])
        row.setdefault("stem", f"definition of node {node}")
        return row

    return DomainView(
        name=name,
        node_ids=tuple(nodes),
        parents=lambda node: tuple(parents.get(node, ())),
        gold=lambda: dict(gold),
        question=question,
        answer_under=rule,
        equal=lambda left, right, node: str(left).strip() == str(right).strip(),
        declaration_roots=frozenset(roots),
        reference=reference,
    )


def _fid(view, klass: str, node: str) -> str:
    from domains.mbd_kernel import instances

    found = [
        item.id
        for item in instances(view)
        if item.klass == klass and tuple(blank for blank, _value in item.injector) == (node,)
    ]
    if len(found) != 1:
        listed = [(item.klass, item.id, item.injector) for item in instances(view)]
        raise AssertionError(f"{klass}@{node} -> {found}; library={listed}")
    return found[0]


def _chain():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": [], "format": "expression"},
        "C": {"options": [], "format": "expression"},
    }

    def rule(node, assignment):
        if node == "B":
            return "1" if assignment.get("A") == "yes" else "0"
        if node == "C":
            return assignment.get("B")
        return None

    return _view(
        "hitting-set chain",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        rule,
    )


def _pair():
    questions = {
        "D": {"options": [], "format": "expression"},
        "E": {"options": [], "format": "expression"},
    }

    def rule(node, assignment):
        return None

    return _view(
        "hitting-set two locals",
        ("D", "E"),
        {"D": (), "E": ()},
        {"D": "1", "E": "2"},
        questions,
        (),
        rule,
    )


def test_hitting_sets_match_the_conflict_example():
    found, truncated = minimal_hitting_sets([
        frozenset({"f1", "f3"}),
        frozenset({"f4"}),
    ])
    assert found == [["f1", "f4"], ["f3", "f4"]]
    assert truncated is False
    smaller, _truncated = minimal_hitting_sets([
        frozenset({"f1"}),
        frozenset({"f1", "f2"}),
    ])
    assert smaller == [["f1"]]
    empty, _truncated = minimal_hitting_sets([frozenset()])
    assert empty == []


def test_conflict_keeps_local_and_upstream_apart():
    assert conflict_faults("wrong_and_locally_inconsistent", ["local"], ["upstream"]) == ["local", "upstream"]
    assert conflict_faults("wrong_but_locally_consistent", ["local"], ["upstream"]) == ["upstream"]
    assert conflict_faults("wrong_without_local_rule", ["local"], ["upstream"]) == ["local", "upstream"]
    assert conflict_faults("wrong_without_local_rule", [], ["upstream"]) == ["upstream"]
    assert conflict_faults("correct", ["local"], ["upstream"]) is None
    assert conflict_faults("wrong_but_locally_consistent", ["local"], []) is None
    assert conflict_faults("unparseable_or_unscorable", ["local"], ["upstream"]) is None
    assert conflict_faults(
        "contract_contradiction",
        [f"{ALG}:B", f"{EXE}:B", f"{RSN}:B", f"{DIR}:B"],
        [f"{DIR}:A", f"{ALG}:up"],
    ) == [f"{ALG}:B", f"{ALG}:up", f"{DIR}:A", f"{DIR}:B", f"{EXE}:B", f"{RSN}:B"]
    injected = {"local": frozenset({"B"}), "upstream": frozenset({"A"}), "side": frozenset({"S"})}
    assert upstream_faults(["local", "upstream", "side"], injected, {"A"}) == ["upstream"]
    assert field_state("wrong_and_locally_inconsistent", True) == "wrong_but_locally_consistent"
    assert field_state("wrong_but_locally_consistent", False) == "wrong_and_locally_inconsistent"
    assert field_state("wrong_without_local_rule", True) == "wrong_without_local_rule"
    assert field_state("wrong_and_locally_inconsistent", None) == "wrong_and_locally_inconsistent"


def test_chain_is_ambiguous_and_repair_stays_separate():
    view = _chain()
    report = explain(view, {"A": "no", "B": "0", "C": "0"}, ("C",))
    assert_report(report)
    assert report["status"] == "attributed"
    assert report["validation"] == []
    assert report["validated_diagnoses"] == []
    assert report["diagnosis_cardinality"] is None
    assert report["repair_cardinality"] == 1
    assert report["repairs"][0]["sets"] == [["A"]]
    e1a = _fid(view, DIR, "A")
    cases = annotate_trajectory(view, {"A": "no", "B": "0", "C": "0"}, ("C",))
    own = next(case for case in cases if case.node == "A")
    propagated = next(case for case in cases if case.node == "B")
    assert own.final_class == "ROOT_ERROR"
    assert propagated.final_class == "PROPAGATED_ERROR"
    assert own.conflict_sets
    assert own.covering_diagnoses
    propagated_faults = {
        fault
        for record in report["conflict_records"]
        if record["node"] == "B"
        for fault in record["faults"]
    }
    assert e1a in propagated_faults
    assert _fid(view, ALG, "B") not in propagated_faults
    causal = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert causal["strength"] == "causal"
    assert causal["chain_class"] == "attributable"
    part = next(item for item in report["candidate_partitions"] if item["target"] == "C")
    assert e1a in part["candidates"]
    assert e1a in part["causal"]
    assert e1a not in part["associative"]
    assert e1a not in part["unresolved"]
    assert all(record["state"] == "wrong_but_locally_consistent" for record in report["conflict_records"] if record["node"] == "B")


def test_two_local_faults_have_diagnosis_two_and_repair_one():
    view = _pair()
    report = explain(view, {"D": "-(1)", "E": "-(2)"}, ("D", "E"))
    assert_report(report)
    assert report["status"] == "attributed"
    assert report["validation"] == []
    assert report["diagnosis_cardinality"] is None
    assert report["repair_cardinality"] == 1
    e4d = _fid(view, ALG, "D")
    e4e = _fid(view, ALG, "E")
    buckets = {row["target"]: row for row in report["candidate_partitions"]}
    assert e4d in buckets["D"]["candidates"]
    assert e4d in buckets["D"]["causal"]
    assert e4e in buckets["E"]["candidates"]
    assert e4e in buckets["E"]["causal"]
    assert {tuple(row["sets"][0]) for row in report["repairs"]} == {("D",), ("E",)}


def test_fields_split_local_consistency():
    identity = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    propagated = "[[0, 0, 0], [0, 1, 0], [0, 0, 1]]"
    observed = "[[0, 9, 0], [0, 1, 0], [0, 0, 1]]"
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": [], "format": "matrix"},
    }

    def rule(node, assignment):
        if node == "B":
            return identity if assignment.get("A") == "yes" else propagated
        return None

    view = _view(
        "hitting-set fields",
        ("A", "B"),
        {"A": (), "B": ("A",)},
        {"A": "yes", "B": identity},
        questions,
        ("A",),
        rule,
    )
    report = explain(view, {"A": "no", "B": observed}, ("B",))
    assert_report(report)
    by_measurement = {record["measurement"]: record for record in report["conflict_records"]}
    consistent = by_measurement["IDB.cell(0,0)"]
    inconsistent = by_measurement["IDB.cell(0,1)"]
    assert consistent["state"] == "wrong_but_locally_consistent"
    e1a = _fid(view, DIR, "A")
    e4b = _fid(view, ALG, "B")
    assert e4b not in consistent["faults"]
    assert e1a in consistent["faults"]
    assert inconsistent["state"] == "wrong_and_locally_inconsistent"
    assert e4b in inconsistent["faults"]
    assert e1a in inconsistent["faults"]


def test_unknown_local_text_stays_a_graph_candidate():
    view = _pair()
    report = explain(view, {"D": "9", "E": "2"}, ("D",))
    assert_report(report)
    assert report["status"] == "attributed"
    assert report["validation"] == []
    assert report["validated_diagnoses"] == []
    assert report["diagnosis_cardinality"] is None
    row = next(item for item in report["candidate_partitions"] if item["target"] == "D")
    assert _fid(view, ALG, "D") in row["candidates"]
    assert _fid(view, ALG, "D") in row["causal"]


def test_clean_and_gold_prefix_do_not_invent_faults():
    view = _chain()
    clean = explain(view, {"A": "yes", "B": "1", "C": "1"}, ("C",))
    assert_report(clean)
    assert clean["status"] == "consistent"
    assert clean["conflicts"] == []
    assert clean["targets"][0]["chain_class"] == "consistent"
    assert clean["targets"][0]["chain_strength"] == ""
    gold = explain(view, {"A": "no", "B": "0", "C": "0"}, ("C",), prefix="gold")
    assert_report(gold)
    assert gold["status"] == "gold_prefix"
    assert gold["minimal_diagnoses"] == []


def test_contradiction_is_propagation_union_local_exe():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "D": {"options": [], "format": "expression"},
    }

    def rule(node, assignment):
        if node == "D":
            return "1" if assignment.get("A") == "yes" else "0"
        return None

    def reference(node, assignment):
        if node == "B":
            return Contradiction("option conflicts with the parent")
        return rule(node, assignment)

    view = _view(
        "contradiction union",
        ("A", "B", "D"),
        {"A": (), "B": ("A",), "D": ("A",)},
        {"A": "yes", "B": "1", "D": "1"},
        questions,
        ("A",),
        rule,
        reference,
    )
    answers = {"A": "no", "B": "0", "D": "9"}
    report = explain(view, answers, ("B", "D"))
    assert_report(report)
    by_node = {}
    for record in report["conflict_records"]:
        by_node.setdefault(record["node"], set()).update(record["faults"])
        if record["node"] == "B":
            assert record["state"] == "contract_contradiction"
        if record["node"] == "D":
            assert record["state"] != "contract_contradiction"
    e1a = _fid(view, DIR, "A")
    assert e1a in by_node["B"]
    assert _fid(view, ALG, "D") not in by_node["B"]
    assert _fid(view, ALG, "D") in by_node["D"]
    cases = diagnose(view, answers)
    root = next(case for case in cases if case.node == "A")
    stuck = next(case for case in cases if case.node == "B")
    algebra = next(case for case in cases if case.node == "D")
    assert root.final_class == "ROOT_ERROR"
    assert stuck.final_class == "UNRESOLVED"
    assert stuck.note == "local contradiction"
    assert ("A",) in stuck.candidate_fault_sets
    assert ("B",) in stuck.candidate_fault_sets
    assert algebra.final_class == "NEW_LOCAL_ERROR"
    assert report["targets"]


def test_missing_phi_is_not_a_contradiction():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
    }
    view = _view(
        "no rule",
        ("A", "B"),
        {"A": (), "B": ("A",)},
        {"A": "yes", "B": "1"},
        questions,
        ("A",),
        lambda node, assignment: None,
    )
    report = explain(view, {"A": "no", "B": "0"}, ("B",))
    assert_report(report)
    assert report["states"]["B"] == "unparseable_or_unscorable"
    assert all(record["node"] != "B" for record in report["conflict_records"])


def test_contradiction_correct_has_no_symptom_and_flags_the_target():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return Contradiction("option conflicts with the parent")
        if node == "C":
            return assignment.get("B") or "0"
        return "yes"

    view = _view(
        "contradiction correct",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        lambda node, assignment: None,
        reference,
    )
    report = explain(view, {"A": "yes", "B": "1"}, ("B",))
    assert_report(report)
    assert report["conflicts"] == []
    target = report["targets"][0]
    assert target["reading"] == "contradiction_correct"
    assert target["reliability"] == "contradiction"
    assert target["process_faults"]
    assert target["process_attribution"]
    assert report["process_records"][0]["node"] == "B"
    cases = diagnose(view, {"A": "yes", "B": "1"})
    stuck = next(case for case in cases if case.node == "B")
    assert stuck.final_class == "UNRELIABLE_CORRECT"


def test_inherited_contradiction_does_not_rewrite_a_defined_child():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return Contradiction("option conflicts with the parent")
        if node == "C":
            return "0" if assignment.get("B") == "0" else "1"
        return None

    view = _view(
        "inherited",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "0", "C": "0"}, ("C",))
    assert_report(report)
    child = next(record for record in report["conflict_records"] if record["node"] == "C")
    assert child["state"] == "wrong_but_locally_consistent"
    assert _fid(view, ALG, "C") not in child["faults"]
    target = report["targets"][0]
    assert target["reading"] == "faithful_wrong"
    assert target["reliability"] == "inherited"
    assert target["answer_class"] == "unattributable"
    assert target["candidate_mode"] == "upstream"
    from domains.conflict_diagnosis import target_note

    assert "unattributable" in target_note(report["targets"])


def test_formula_domain_contradiction_keeps_the_local_fault():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "D": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "D":
            return Contradiction("empty domain")
        return None

    view = _view(
        "formula contradiction",
        ("A", "D"),
        {"A": (), "D": ("A",)},
        {"A": "yes", "D": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "D": "9"}, ("D",))
    assert_report(report)
    record = next(item for item in report["conflict_records"] if item["node"] == "D")
    assert record["state"] == "contract_contradiction"
    assert _fid(view, ALG, "D") in record["faults"]
    assert _fid(view, DIR, "A") in record["faults"]
    assert report["repairs"][0]["sets"] == [["A", "D"]]


def test_gold_parents_tell_parent_contradiction_from_local():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            if assignment.get("A") == "yes":
                return "1"
            return Contradiction("caused by a parent")
        if node == "C":
            return Contradiction("the node itself")
        return None

    view = _view(
        "contradiction source",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "0", "C": "9"}, ("B", "C"))
    assert_report(report)
    by_node = {item["node"]: item for item in report["readings"]}
    assert by_node["B"]["contradiction_source"] == "parents"
    assert by_node["C"]["contradiction_source"] == "local"
    assert by_node["B"]["candidate_mode"] == "union"
    sets = {row["target"]: row["sets"] for row in report["repairs"]}
    assert sets["B"] == [["A"]]
    assert sets["C"] == [["A", "C"]]


def test_absorbed_gold_keeps_the_upstream_fault():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return "1"
        if node == "C":
            return "1" if assignment.get("B") == "1" else "0"
        return None

    view = _view(
        "gold cannot cut the chain",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "1", "C": "1"}, ("C",))
    assert_report(report)
    assert all(record["node"] != "B" for record in report["conflict_records"])
    assert all(record["node"] != "C" for record in report["conflict_records"])
    by_node = {row["node"]: row for row in report["process_records"]}
    e1a = _fid(view, DIR, "A")
    assert e1a in by_node["B"]["faults"]
    assert e1a in by_node["C"]["faults"]
    assert report["targets"][0]["answer_class"] == "traceable_masking"
    masked = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert masked["strength"] == "causal"
    assert masked["chain_class"] == "traceable_masking"
    part = next(item for item in report["candidate_partitions"] if item["target"] == "C")
    assert e1a in part["causal"]
    cases = annotate_trajectory(view, {"A": "no", "B": "1", "C": "1"}, ("C",))
    kept = next(case for case in cases if case.node == "C")
    assert kept.final_class == "UNRELIABLE_CORRECT"
    assert e1a in kept.process_faults


def test_accidental_gold_stays_causal():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return "1" if assignment.get("A") == "yes" else "0"
        if node == "C":
            return "1" if assignment.get("B") == "1" else "0"
        return None

    view = _view(
        "local inconsistency still propagates",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "1", "C": "1"}, ("C",))
    assert_report(report)
    kept = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert kept["strength"] == "causal"
    assert kept["chain_class"] == "traceable_masking"
    assert report["targets"][0]["chain_strength"] == "causal"
    assert report["targets"][0]["chain_class"] == "traceable_masking"
    part = next(item for item in report["candidate_partitions"] if item["target"] == "C")
    e1a = _fid(view, DIR, "A")
    assert e1a in part["causal"]
    assert e1a not in part["associative"]


def test_one_unfaithful_side_path_stays_causal():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "D": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node in {"B", "D"}:
            return "1" if assignment.get("A") == "yes" else "0"
        if node == "C":
            return "1" if assignment.get("B") == "1" and assignment.get("D") == "1" else "0"
        return None

    view = _view(
        "side-path local deviation",
        ("A", "B", "D", "C"),
        {"A": (), "B": ("A",), "D": ("A",), "C": ("B", "D")},
        {"A": "yes", "B": "1", "D": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "0", "D": "1", "C": "0"}, ("C",))
    assert_report(report)
    kept = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert kept["strength"] == "causal"
    assert kept["chain_class"] == "attributable"
    assert report["targets"][0]["chain_strength"] == "causal"
    assert report["targets"][0]["chain_class"] == "attributable"
    part = next(item for item in report["candidate_partitions"] if item["target"] == "C")
    e1a = _fid(view, DIR, "A")
    assert e1a in part["causal"]
    assert e1a not in part["associative"]


def test_local_value_break_is_a_causal_candidate():
    questions = {
        "B": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        del node, assignment
        return "1"

    view = _view(
        "local deviation is a cause",
        ("B",),
        {"B": ()},
        {"B": "1"},
        questions,
        (),
        reference,
        reference,
    )
    report = explain(view, {"B": "9"}, ("B",))
    assert_report(report)
    part = next(item for item in report["candidate_partitions"] if item["target"] == "B")
    assert _fid(view, ALG, "B") in part["causal"]
    assert part["associative"] == []
    assert part["unresolved"] == []


def test_contradiction_on_the_chain_is_associative():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return Contradiction("parent is undefined")
        if node == "C":
            return assignment.get("B") or "0"
        return None

    view = _view(
        "contradiction chain is associative",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "0", "C": "0"}, ("C",))
    assert_report(report)
    linked = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert linked["strength"] == "associative"
    assert linked["chain_class"] == "unattributable"
    assert report["targets"][0]["chain_class"] == "unattributable"
    part = next(item for item in report["candidate_partitions"] if item["target"] == "C")
    e1a = _fid(view, DIR, "A")
    assert e1a in part["associative"]
    assert e1a not in part["causal"]
    assert e1a not in part["unresolved"]


def test_recognition_enters_contract_cognition_without_being_a_parent():
    """A wrong recognition answer is the contract, even for a node that does not list it."""
    from domains.contract_cognition import working_texts

    texts = working_texts(
        {"A": "right", "B": "1", "C": "R", "D": "gold"},
        {"A": "left", "B": "1", "C": "L", "D": "zzz"},
        ("A",),
        ("B",),
    )
    assert texts["A"] == "left"
    assert texts["B"] == "1"
    assert texts["D"] == "gold"

    questions = {
        "A": {"options": ["right", "left"], "format": "choice"},
        "B": {"options": ["0", "1"], "format": "choice"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "B":
            return "1"
        if node == "C":
            return "L" if assignment.get("A") == "left" else "R"
        return None

    view = _view(
        "contract cognition across nodes",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "right", "B": "1", "C": "R"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "left", "B": "1", "C": "L"}, ("C",))
    assert_report(report)
    assert report["states"]["C"] == "wrong_but_locally_consistent"
    assert report["targets"][0]["chain_class"] == "attributable"


def test_unresolved_ancestor_does_not_cover_a_closed_node():
    questions = {
        "A": {"options": ["yes", "no"], "format": "choice"},
        "B": {"options": [], "format": "expression"},
        "C": {"options": [], "format": "expression"},
    }

    def reference(node, assignment):
        if node == "C":
            return "1"
        return None

    view = _view(
        "unresolved ancestor does not cover a closed node",
        ("A", "B", "C"),
        {"A": (), "B": ("A",), "C": ("B",)},
        {"A": "yes", "B": "1", "C": "1"},
        questions,
        ("A",),
        reference,
        reference,
    )
    report = explain(view, {"A": "no", "B": "1", "C": "1"}, ("C",))
    assert_report(report)
    target = report["targets"][0]
    assert target["chain_class"] == "consistent"
    assert target["chain_strength"] == "causal"
    assert {"source": "A", "strength": "unresolved"} in target["side_paths"]
    opened = next(item for item in report["chain_attributions"] if item["source"] == "A" and item["target"] == "C")
    assert opened["strength"] == "unresolved"


def test_wrong_choice_without_a_rule_is_its_own_fault():
    from domains.conflict_diagnosis import _apply_chains, _five_class

    readings = [{
        "node": "ID23",
        "reading": "no_rule",
        "matches_gold": False,
        "process_faults": [],
    }]
    _apply_chains(readings, [{
        "target": "23",
        "source": "23",
        "strength": "unresolved",
        "chain_class": "unresolved",
    }])
    assert readings[0]["chain_class"] == "attributable"
    assert readings[0]["chain_strength"] == "causal"
    assert readings[0]["side_paths"] == []
    assert _five_class(readings[0]) == "attributable"


def test_shape_failure_is_not_a_contradiction():
    class Execution:
        def __init__(self, status, matrix="", reason=""):
            self.status = status
            self.matrix = matrix
            self.ok = status == "derived"
            self.reason = reason

    assert from_execution(Execution("derived", "1")) == "1"
    assert from_execution(Execution("contradictory", reason="total load matrix is not 3×3.")) is None
    assert from_execution(Execution("contradictory", reason="ID12 no calculator")) is None
    got = from_execution(Execution("contradictory", reason="The arcsine argument is outside the domain."))
    assert isinstance(got, Contradiction)


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_") and callable(value)]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok {test.__name__}")
    if failed:
        raise SystemExit(f"{failed} failed")
    print(f"{len(tests)} passed")


if __name__ == "__main__":
    main()
