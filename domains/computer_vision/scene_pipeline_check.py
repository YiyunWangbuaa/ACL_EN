# -*- coding: utf-8 -*-
"""Run vision scene / object+stereo stages 0–7 gates in order."""
from __future__ import annotations

from analysis.audit_composers_scene import check as check_stage7
from domains.computer_vision.scene_answer_form import check_answer_form
from domains.computer_vision.scene_contract_verifiers import check_verifiers
from domains.computer_vision.scene_dmatrix import check_dmatrix
from domains.computer_vision.scene_error_graph import check_graph
from domains.computer_vision.scene_error_workflow import check_attribution
from domains.computer_vision.scene_stage0_check import check_stage0
from domains.computer_vision.scene_t_final import check_t_final


def main() -> None:
    check_stage0()
    check_graph()
    check_answer_form()
    check_verifiers()
    check_dmatrix()
    check_attribution()
    check_t_final()
    check_stage7()
    print("机器视觉阶段 0–7 通过（物体与双目环节）。", flush=True)


if __name__ == "__main__":
    main()
