# -*- coding: utf-8 -*-
"""Run kinematics stages 0–7 gates in order."""
from __future__ import annotations

from analysis.audit_composers_kinematics import check as check_stage7
from domains.robotics.kin_answer_form import check_answer_form
from domains.robotics.kin_contract_verifiers import check_verifiers
from domains.robotics.kin_dmatrix import check_dmatrix
from domains.robotics.kin_error_graph import check_graph
from domains.robotics.kin_error_workflow import check_attribution
from domains.robotics.kin_stage0_check import check_stage0
from domains.robotics.kin_t_final import check_t_final


def main() -> None:
    check_stage0()
    check_graph()
    check_answer_form()
    check_verifiers()
    check_dmatrix()
    check_attribution()
    check_t_final()
    check_stage7()
    print("Robotics kinematics stages 0-7 all passed.", flush=True)


if __name__ == "__main__":
    main()
