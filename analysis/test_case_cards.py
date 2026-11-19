# -*- coding: utf-8 -*-
"""Checks for the case-card compiler. No model calls."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.case_cards import (  # noqa: E402
    LABELS,
    MECHANISM_LABELS,
    check_standard_answers,
    compile_case,
    compile_catalog,
    load_catalog,
    render_report,
    score_formula,
    score_matrix,
)


STANDARD_MATRIX = "[[1, 0, 0], [0, cos(γ_M - γ_tp), sin(γ_M - γ_tp)], [0, -sin(γ_M - γ_tp), cos(γ_M - γ_tp)]]"
CONTRACT_MATRIX = "[[1, 0, 0], [0, cos(γ_M + γ_tp), -sin(γ_M + γ_tp)], [0, sin(γ_M + γ_tp), cos(γ_M + γ_tp)]]"
SLOTS = [{"id": "r23", "row": 2, "col": 3}, {"id": "r32", "row": 3, "col": 2}]


def _blank(answer, prediction, options=None, role="", parents=None, prompt=""):
    return {
        "answer": answer,
        "prediction": prediction,
        "options": options or [],
        "role": role,
        "depends_on": parents or [],
        "prompt": prompt,
    }


def _spec(**overrides):
    spec = {
        "case_id": "T",
        "name": "测试",
        "question": "父节点与契约一致时，目标矩阵是否仍使用该方向？",
        "source": {"records": "memory", "sample_id": "toy", "id_scheme": "current"},
        "contract_definition": "反常方向",
        "standard_definition": "右手方向",
        "stem": "旋转矩阵",
        "leakage_note": "题干没有符号。",
        "error_cancellation": "no",
        "target": "85",
        "direct_parents": ["84"],
        "relevant_parents": ["84"],
        "contrast_sources": ["84"],
        "standard_answers": {"84": "正", "85": STANDARD_MATRIX},
        "nodes": [
            {"id": "84", "content": "方向", "node_type": "sign", "decides_target": "yes"},
            {"id": "85", "content": "矩阵", "node_type": "matrix", "decides_target": "yes"},
        ],
        "evidence_elements": {"kind": "matrix_sin", "slots": SLOTS},
        "probe": {"after": "84", "before": "85", "write_back": False, "question": "r23 的 sin 取正还是负？"},
        "repairs": {"non_ancestor": {"policy": "one_wrong_non_ancestor"}},
    }
    spec.update(overrides)
    return spec


def test_matrix_signs_ignore_the_angle_argument():
    scored = score_matrix(STANDARD_MATRIX, CONTRACT_MATRIX, STANDARD_MATRIX, SLOTS)
    assert scored["relation"] == "standard"
    assert [row["verdict"] for row in scored["rows"]] == ["标准一致", "标准一致"]
    assert scored["full_relation"] == "standard"


def test_binary_output_is_not_an_isolated_pattern():
    from domains.wind_tunnel.case_cards import behavior_label

    label, reason = behavior_label("standard", "yes", "yes", "contract", False)
    assert label == "underdetermined"
    assert "两个选项" in reason


def test_sign_target_binary_stays_underdetermined():
    spec = _spec(
        target="1",
        direct_parents=["0"],
        relevant_parents=["0"],
        contrast_sources=["0"],
        standard_answers={"0": "保持", "1": "正"},
        nodes=[
            {"id": "0", "content": "定义", "node_type": "semantic", "decides_target": "no"},
            {"id": "1", "content": "符号", "node_type": "sign", "decides_target": "yes"},
        ],
    )
    # Reuse the matrix scorer only when the target text is a matrix. For a sign
    # target the catalog uses formula or the node relation. Here the target is
    # replaced by a matrix so the binary gate can be exercised on a real sign
    # node that is also the contrast source of a matrix.
    spec = _spec()
    observations = {
        "84": _blank("逆", "正", ["正", "逆"], "该滚转相对坐标轴正方向的转向", ["82"], "方向"),
        "85": _blank(CONTRACT_MATRIX, STANDARD_MATRIX, [], "安装滚转矩阵", ["84"], "矩阵 " + STANDARD_MATRIX),
    }
    card = compile_case(spec, observations, {"condition": "self-trunc"})
    assert card["direct_parent_correct"] == "no"
    assert card["output_relation"] == "standard"
    assert card["label"] == "underdetermined"
    assert card["evidence"]["evidence_strength"] == "insufficient"
    assert card["evidence"]["not_binary"] == "yes"
    assert card["evidence_for_fallback"] == "insufficient"


def test_clean_matrix_is_standard_consistent_without_a_mechanism_label():
    observations = {
        "84": _blank("逆", "逆", ["正", "逆"], "该滚转相对坐标轴正方向的转向", []),
        "85": _blank(CONTRACT_MATRIX, STANDARD_MATRIX, [], "安装滚转矩阵", ["84"]),
    }
    card = compile_case(_spec(), observations, {"condition": "session-model", "model": "toy"})
    assert card["direct_parent_edge"] == "yes"
    assert card["direct_parent_correct"] == "yes"
    assert card["label"] == "standard_consistent"
    assert card["evidence"]["evidence_strength"] == "strong"
    assert card["label"] not in MECHANISM_LABELS
    assert card["experiment"]["probe"]["write_back"] is False
    assert card["organization"]["checks"]["strength_without_mechanism"] is True


def test_declared_parent_missing_from_the_record_edge():
    observations = {
        "84": _blank("逆", "逆", ["正", "逆"], "该滚转相对坐标轴正方向的转向", ["82"]),
        "85": _blank(CONTRACT_MATRIX, STANDARD_MATRIX, [], "安装滚转矩阵", ["82", "83"]),
    }
    card = compile_case(_spec(), observations, {})
    assert card["direct_parent_edge"] == "no"
    assert card["direct_parent_correct"] == "not_applicable"
    assert card["evidence"]["evidence_strength"] == "insufficient"
    assert "记录依赖" in card["label_reason"]


def test_formula_with_the_sign_hidden_in_the_arm_is_mixed():
    scored = score_formula(
        "CN·(Xg - Xr)/L_RT",
        "CN·(Xr - Xg)/L_RT",
        "-CN·(Xr - Xg)/L_RT",
    )
    assert scored["algebraic"] == "standard"
    assert scored["format_split"] is True
    assert scored["relation"] == "mixed"
    assert next(row["model"] for row in scored["rows"] if row["id"] == "format") == "sign_absorbed"
    written = score_formula(
        "-CN*(Xr-Xg)/L_RT",
        "CN·(Xr - Xg)/L_RT",
        "-CN·(Xr - Xg)/L_RT",
    )
    assert written["format_split"] is False
    assert written["relation"] == "standard"
    assert next(row["verdict"] for row in written["rows"] if row["id"] == "outer_sign") == "标准一致"
    spec = _spec(
        target="198",
        direct_parents=["192"],
        relevant_parents=["192"],
        contrast_sources=["192"],
        standard_answers={"192": "负", "198": "-CN·(Xr - Xg)/L_RT"},
        nodes=[
            {"id": "192", "content": "符号", "node_type": "sign", "decides_target": "yes"},
            {"id": "198", "content": "公式", "node_type": "expression", "decides_target": "yes"},
        ],
        evidence_elements={"kind": "formula"},
        probe={"after": "192", "before": "198", "write_back": False, "question": "整体符号？"},
    )
    observations = {
        "192": _blank("正", "正", ["正", "负"], "等效力矩MCN_CZ的符号", []),
        "198": _blank("CN·(Xr - Xg)/L_RT", "CN·(Xg - Xr)/L_RT", [], "等效力矩MCN_CZ", ["192"]),
    }
    card = compile_case(spec, observations, {})
    assert card["label"] == "mixed_representation"
    assert card["direct_parent_correct"] == "yes"


def test_standard_child_of_a_standard_parent_is_not_isolated():
    observations = {
        "84": _blank("逆", "正", ["正", "逆"], "该滚转相对坐标轴正方向的转向", []),
        "85": _blank(CONTRACT_MATRIX, STANDARD_MATRIX, [], "安装滚转矩阵", ["84"]),
    }
    card = compile_case(_spec(), observations, {})
    assert card["contrast_source_relation"] == "standard"
    assert card["label"] == "underdetermined"
    assert card["evidence"]["model_matches_standard"] == "yes"
    assert card["evidence"]["direct_parent_correct"] == "no"


def test_contract_child_after_a_wrong_parent_is_local_recovery():
    observations = {
        "84": _blank("逆", "正", ["正", "逆"], "该滚转相对坐标轴正方向的转向", []),
        "85": _blank(CONTRACT_MATRIX, CONTRACT_MATRIX, [], "安装滚转矩阵", ["84"]),
    }
    card = compile_case(_spec(), observations, {})
    assert card["output_relation"] == "contract"
    assert card["label"] == "local_recovery"


def test_mechanism_label_is_rejected():
    try:
        compile_case(_spec(forced_label="fallback"), {}, {})
    except RuntimeError as exc:
        assert "机制词" in str(exc)
    else:
        raise AssertionError("机制标签不应被接受")


def test_probe_answer_cannot_be_written_back():
    spec = _spec()
    spec["probe"] = dict(spec["probe"], write_back=True)
    try:
        compile_case(spec, {
            "84": _blank("逆", "逆", ["正", "逆"], "该滚转相对坐标轴正方向的转向", []),
            "85": _blank(CONTRACT_MATRIX, STANDARD_MATRIX, [], "安装滚转矩阵", ["84"]),
        }, {})
    except RuntimeError as exc:
        assert "写回" in str(exc)
    else:
        raise AssertionError("写回探针应被拒绝")


def test_catalog_matches_records_and_standard_samples():
    catalog = load_catalog(ROOT / "data" / "wind_tunnel" / "cases" / "catalog.json")
    problems = check_standard_answers(catalog, ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl")
    assert problems == []
    cards = {card["case_id"]: card for card in compile_catalog(catalog, ROOT)}
    assert set(cards) == {"C1", "C2a", "C2b", "C3"}
    for card in cards.values():
        assert card["label"] in LABELS
        assert card["label"] not in MECHANISM_LABELS
        assert card["evidence_for_fallback"] == "insufficient"
        assert card["experiment"]["probe"]["write_back"] is False
        assert card["identity_ok"] is True
        assert card["organization"]["complete"] is True
        assert tuple(card["evidence"]) == tuple(dict(cards["C1"]["evidence"]))

    roll = cards["C1"]
    assert roll["id_scheme"] == "legacy_pair"
    assert roll["target_canonical"] == "88"
    assert roll["output_relation"] == "standard"
    assert roll["direct_parent_edge"] == "no"
    assert roll["direct_parent_correct"] == "not_applicable"
    assert roll["recorded_parents"] == ["82", "83"]
    assert roll["recorded_parent_correct"] == "no"
    assert roll["label"] == "underdetermined"
    assert roll["evidence"]["evidence_strength"] == "insufficient"
    assert roll["evidence"]["answer_leakage"] == "yes"
    assert roll["evidence"]["model_matches_standard"] == "yes"
    assert [row["verdict"] for row in roll["elements"]] == ["标准一致", "标准一致"]
    direction = next(node for node in roll["nodes"] if node["id"] == "84")
    assert direction["model"] == "正"
    assert direction["contract"] == "逆"
    assert {item["claim"] for item in roll["narrative_mismatches"]} >= {"direct_parent_edge", "ID84.model", "ID82.model"}

    beta = cards["C2a"]
    gamma = cards["C2b"]
    assert beta["label"] == "standard_consistent"
    assert gamma["label"] == "standard_consistent"
    assert beta["contrast_source_correct"] == "yes"
    assert beta["direct_parent_correct"] == "yes"
    assert beta["direct_parent_discriminative"] == "no"
    assert beta["relevant_parent_wrong"] == "yes"
    assert beta["evidence"]["evidence_strength"] == "weak"
    assert beta["same_chain_selective_failure"] == "no"
    assert gamma["same_chain_selective_failure"] == "no"
    assert beta["handedness"]["handedness_used_in_beta_matrix"] == "no"
    assert beta["handedness"]["handedness_used_in_gamma_matrix"] == "no"
    assert beta["handedness"]["handedness_contract_visible_at_definition"] == "yes"
    assert any(item["claim"] == "same_chain_selective_failure" for item in gamma["narrative_mismatches"])

    torque = cards["C3"]
    assert torque["target_canonical"] == "198"
    assert torque["output_relation"] == "standard"
    assert torque["label"] == "underdetermined"
    assert torque["direct_parent_correct"] == "no"
    assert torque["direct_parent_edge"] == "yes"
    assert torque["evidence"]["evidence_strength"] == "insufficient"
    sign = next(node for node in torque["nodes"] if node["id"] == "192")
    lever = next(node for node in torque["nodes"] if node["id"] == "184")
    assert sign["model"] == "负"
    assert sign["contract"] == "正"
    assert lever["wrapped"] is True
    assert lever["relation"] == "both"
    facets = {row["id"]: row["verdict"] for row in torque["elements"]}
    assert facets["outer_sign"] == "标准一致"
    assert facets["arm"] == "无判别力"
    assert facets["force"] == "无判别力"
    assert any(item["claim"] == "direct_parent_correct" for item in torque["narrative_mismatches"])

    report = render_report(list(cards.values()))
    assert "fallback" not in report
    assert "C1" in report and "C2a" in report and "C3" in report


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
