# -*- coding: utf-8 -*-
"""Run robotics theory stages 0–7 gates in order."""
from __future__ import annotations

from analysis.audit_composers_robotics import check as check_stage7
from domains.robotics.answer_form import check_answer_form
from domains.robotics.contract_verifiers import check_verifiers
from domains.robotics.dmatrix import check_dmatrix
from domains.robotics.error_graph import check_graph
from domains.robotics.error_workflow import check_attribution
from domains.robotics.stage0_check import check_stage0
from domains.robotics.t_final import check_t_final


def main() -> None:
    raise SystemExit("机器人学理论稿已删除：这些题已由运动学 1–156 覆盖。请使用 python -m domains.robotics.kin_pipeline_check。")
    check_stage0()
    check_graph()
    check_answer_form()
    check_verifiers()
    check_dmatrix()
    check_attribution()
    check_t_final()
    check_stage7()
    print("机器人学理论首块阶段 0–7 全部通过。", flush=True)


if __name__ == "__main__":
    main()
