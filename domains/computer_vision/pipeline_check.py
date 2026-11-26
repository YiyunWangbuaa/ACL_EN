# -*- coding: utf-8 -*-
"""Run vision basic stages 0–7 gates in order."""
from __future__ import annotations

from analysis.audit_composers_vision import check as check_stage7
from domains.computer_vision.answer_form import check_answer_form
from domains.computer_vision.contract_verifiers import check_verifiers
from domains.computer_vision.dmatrix import check_dmatrix
from domains.computer_vision.error_graph import check_graph
from domains.computer_vision.error_workflow import check_attribution
from domains.computer_vision.stage0_check import check_stage0
from domains.computer_vision.t_final import check_t_final


def main() -> None:
    check_stage0()
    check_graph()
    check_answer_form()
    check_verifiers()
    check_dmatrix()
    check_attribution()
    check_t_final()
    check_stage7()
    print("Computer vision stages 0-7 passed (vision basics).", flush=True)


if __name__ == "__main__":
    main()
