# -*- coding: utf-8 -*-
"""Run flight stages 0–7 gates in order. Stops on the first failure."""
from __future__ import annotations

from analysis.audit_composers_flight import check as check_stage7
from domains.aerodynamics.answer_form import check_answer_form
from domains.aerodynamics.contract_verifiers import check_verifiers
from domains.aerodynamics.dmatrix import check_dmatrix
from domains.aerodynamics.error_graph import check_graph
from domains.aerodynamics.error_workflow import check_attribution
from domains.aerodynamics.stage0_check import check_stage0
from domains.aerodynamics.t_final import check_t_final


def main() -> None:
    check_stage0()
    check_graph()
    check_answer_form()
    check_verifiers()
    check_dmatrix()
    check_attribution()
    check_t_final()
    check_stage7()
    print("Flight mechanics 1-131, stages 0-7, all passed.", flush=True)


if __name__ == "__main__":
    main()
