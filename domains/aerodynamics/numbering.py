# -*- coding: utf-8 -*-
"""Continuous blank numbers across the three flight-mechanics subprocesses.

Generators keep a local number inside each subprocess. The number written into
a sample is the public number, the same way the wind-tunnel sample runs from
1 through 194 across its four subprocesses.
"""

from domains.aerodynamics.body_air_generate import BLANK_COUNT as AIR_BLANKS
from domains.aerodynamics.body_air_generate import PROCESS as AIR_PROCESS
from domains.aerodynamics.ground_body_generate import BLANK_COUNT as BODY_BLANKS
from domains.aerodynamics.ground_body_generate import PROCESS as BODY_PROCESS
from domains.aerodynamics.ground_track_air_generate import BLANK_COUNT as TRACK_BLANKS
from domains.aerodynamics.ground_track_air_generate import PROCESS as TRACK_PROCESS

PROCESS_SIZE = {
    BODY_PROCESS: BODY_BLANKS,
    TRACK_PROCESS: TRACK_BLANKS,
    AIR_PROCESS: AIR_BLANKS,
}
PROCESS_OFFSET: dict[str, int] = {}
_cursor = 0
for _name, _size in PROCESS_SIZE.items():
    PROCESS_OFFSET[_name] = _cursor
    _cursor += _size
TOTAL_BLANKS = _cursor


def public_id(process: str, local_id: str) -> str:
    return str(int(local_id) + PROCESS_OFFSET[process])


def public_span(process: str) -> tuple[int, int]:
    start = PROCESS_OFFSET[process] + 1
    return start, start + PROCESS_SIZE[process] - 1
