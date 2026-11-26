# -*- coding: utf-8 -*-
"""Logical dependency graph for the 207 wind-flight cloze blanks.

An edge u→v means the gold derivation of blank v uses the result of blank u.
The edge follows the calculation, not the order of the sentences. After
bind_reading_order, a parent asked in a later sentence is a future dependency:
a truncated prefix stops at the child's sentence and cannot carry that parent.
A definition edge d→q records where an engineering definition enters a blank.
Downstream closure then follows the answer edges, so a wrong blank can be
traced back to the definition that can affect it.

Repeated questions that only read the declaration again are separate roots.
A later blank does not depend on an earlier one when a nearer result already
screens it off. Single-point injection, which keeps every other prefix answer
correct, therefore follows the direct edges. Natural model-prefix propagation
follows the transitive descendants, because those intermediate answers are the
model's own.

The contract layer sits on this graph and does not add answer edges. Contract
instances use the C_ namespace. Binding rules are metadata for joint fields;
they are not nodes and do not change depth or error propagation. See
contract_graph.py.

Run::

    python -m domains.wind_tunnel.error_graph
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "wind_tunnel" / "dependency_graph.json"

WIND_TO_BODY = "wind frame to body-axis frame"
BALANCE_TO_BODY = "balance frame to body-axis frame"
BALANCE_TO_WIND = "balance frame to wind frame"
REFERENCE_POINT = "moment reference point"

PROCESS_OFFSET = {
    WIND_TO_BODY: 0,
    BALANCE_TO_BODY: 81,
    BALANCE_TO_WIND: 116,
    REFERENCE_POINT: 180,
}
PROCESS_SIZE = {
    WIND_TO_BODY: 81,
    BALANCE_TO_BODY: 35,
    BALANCE_TO_WIND: 64,
    REFERENCE_POINT: 33,
}
# Same as dataset_generate.REMOVED_SLOTS. The expansions were removed and later blank numbers were not shifted.
REMOVED_SLOTS = {
    WIND_TO_BODY: frozenset({52, 53, 54}),
    BALANCE_TO_BODY: frozenset({101, 102, 103}),
}

ANSWER_TYPE = {
    "semantic": "choice",
    "sign": "sign",
    "matrix": "matrix",
    "expression": "expression",
    "vector": "vector",
}
FORMAT_OF_ANSWER = {
    "choice": {"choice"},
    "sign": {"choice"},
    "matrix": {"matrix"},
    "vector": {"vector"},
    "expression": {"expression", "arctan", "arcsin", "distance", "moment-arm"},
}
PUBLIC_KEYS = (
    "depends_on",
    "dependency_depth",
    "min_dependency_depth",
    "target_definition",
    "answer_type",
    "error_type",
    "future_dependencies",
    "role",
    "dependency_edge_types",
    "definition_relations",
    "rule_mention",
    "rule_definition",
    "rule_definition_name",
)
_SLOT = re.compile(r"@@(\d+)@@")
# process -> public id -> sentence index. Empty until the manuscript is bound.
_SENTENCE_OF: dict[str, dict[str, int]] = {}

# id, label, error type, target, role, then optional "| parent parent".
_TABLES = {
    WIND_TO_BODY: """
1 C2 semantic frame_conversion frame conversion direction
2 C1 semantic wind_normal_axis wind-frame absolute normal
3 C1 semantic wind_axial_axis wind-frame absolute axial
4 C1 semantic wind_lateral_axis wind-frame absolute lateral
5 C1 semantic body_normal_axis airframe absolute normal
6 C1 semantic body_axial_axis airframe absolute axial
7 C1 semantic body_lateral_axis airframe absolute lateral
8 A2 matrix rp1 pre-matrix RP1 from wind frame to body-axis frame | 2 3 4 5 6 7
9 C2 semantic rp1_aligned_frame frame aligned by RP1
10 C2 semantic rp1_source_frame frame compared by RP1
11 C1 semantic placement model installation
12 A1 sign mount_relation whether normal and lateral agree after installation | 11
13 A2 matrix rp2 mounting pre-matrix RP2 | 5 6 7 12
14 C2 semantic rp2_aligned_frame frame aligned by RP2
15 C2 semantic rp2_source_frame frame compared by RP2
16 A1 semantic euler_first first item of the intrinsic order
17 A1 semantic euler_second second item of the intrinsic order
18 A1 semantic euler_third third item of the intrinsic order
19 C1 semantic placement_alpha installation reread for the angle of attack
20 A1 semantic alpha_m_airspeed airspeed side at a positive mechanism angle of attack | 19
21 C1 semantic alpha_airspeed airspeed side at a positive theoretical angle of attack
22 A1 sign alpha_m_sign sign of the theoretical angle of attack under a positive mechanism angle of attack | 20 21
23 C2 semantic alpha_frame frame of the angle of attack rotation
24 C1 semantic alpha_axis airframe coordinate axis of the angle of attack | 5 6 7 23
25 A1 sign alpha_turn turning sense of the angle of attack about that coordinate axis | 5 6 7 20 24
26 A2 matrix r_alpha angle of attack rotation matrix | 24 25
27 C1 semantic placement_beta installation reread for sideslip
28 A1 semantic beta_m_airspeed airspeed side at positive mechanism sideslip | 27
29 C1 semantic beta_airspeed airspeed side at positive theoretical sideslip
30 A1 sign beta_m_sign sign of the theoretical sideslip under positive mechanism sideslip | 28 29
31 C2 semantic beta_frame frame of the sideslip rotation
32 C1 semantic beta_axis airframe coordinate axis of the sideslip | 5 6 7 31
33 A1 sign beta_turn turning sense of the sideslip about that coordinate axis | 5 6 7 28 32
34 A2 matrix r_beta sideslip rotation matrix | 32 33
35 C1 sign gamma_m_hand relation of mechanism roll to the right-hand rule
36 A1 semantic gamma_m_wing right-wing side at positive mechanism roll | 35
37 C1 sign gamma_hand relation of theoretical roll to the right-hand rule
38 A1 semantic gamma_wing right-wing side at positive theoretical roll | 37
39 A1 sign gamma_sign sign of the theoretical roll under positive mechanism roll | 36 38
40 C2 semantic gamma_frame frame of the roll rotation
41 C1 semantic gamma_axis airframe coordinate axis of the roll | 6 40
42 A1 sign gamma_turn turning sense of the roll about that coordinate axis | 6 35 41
43 A2 matrix r_gamma roll rotation matrix | 41 42
44 C2 semantic passive_transform active or passive transform
45 C2 semantic moving_axes moving axes or fixed axes for the body-axis frame
46 C2 semantic intrinsic_rotation intrinsic or extrinsic rotation | 45
47 A2 semantic product_r1 R1 in the attitude product | 16 17 18 44 46
48 A2 semantic product_r2 R2 in the attitude product | 16 17 18 44 46
49 A2 semantic product_r3 R3 in the attitude product | 16 17 18 44 46
50 A2 semantic product_r4 R4 in the attitude product | 44 46
51 A2 semantic product_r5 R5 in the attitude product | 44 46
55 A2 matrix r_t total matrix of the three attitude rotations | 26 34 43 47 48 49
56 A1 vector level_airspeed airspeed coordinates in the level airframe | 3 8 13
57 C2 semantic level_frame frame studied next | 56
58 A1 vector level_airspeed_vector level airspeed unit vector used in the substitution | 56
59 A2 expression xtz_component XTZ component of airspeed in body axes | 55 58
60 A2 expression ytz_component YTZ component of airspeed in body axes | 55 58
61 A2 expression ztz_component ZTZ component of airspeed in body axes | 55 58
62 A1 sign xtz_sign_alpha sign of XTZ when only the angle of attack is positive | 59
63 A1 sign ytz_sign_alpha sign of YTZ when only the angle of attack is positive | 60
64 A1 sign ztz_sign_alpha sign of ZTZ when only the angle of attack is positive | 61
65 A1 sign xtz_sign_beta sign of XTZ when only sideslip is positive | 59
66 A1 sign ytz_sign_beta sign of YTZ when only sideslip is positive | 60
67 A1 sign ztz_sign_beta sign of ZTZ when only sideslip is positive | 61
68 A1 sign alpha_t_sign outer sign of the theoretical angle of attack formula | 22
69 A1 semantic alpha_inverse inverse function used by the theoretical angle of attack
70 A1 sign alpha_num_sign numerator sign of the theoretical angle of attack | 62 63 64 71
71 A1 semantic alpha_num_axis numerator component of the theoretical angle of attack | 5 6 7 21
72 A1 sign alpha_den_sign denominator sign of the theoretical angle of attack | 62 63 64 73
73 A1 semantic alpha_den_axis denominator component of the theoretical angle of attack | 6
74 A1 sign beta_t_sign outer sign of the theoretical sideslip formula | 30
75 A1 semantic beta_inverse inverse function used by the theoretical sideslip
76 A1 sign beta_num_sign numerator sign of the theoretical sideslip | 65 66 67 77
77 A1 semantic beta_num_axis numerator component of the theoretical sideslip | 5 6 7 29
78 A1 sign beta_den_sign denominator sign of the theoretical sideslip | 75
79 A1 semantic beta_den_axis denominator component of the theoretical sideslip | 75
80 A2 expression alpha_t theoretical angle of attack formula | 59 60 61 68 69 70 71 72 73
81 A2 expression beta_t theoretical sideslip formula | 59 60 61 74 75 76 77 78 79
""",
    BALANCE_TO_BODY: """
82 C2 semantic load_conversion force and moment conversion direction
83 C1 sign gamma_tp_hand sign of the balance installation angle relative to the right-hand rule
84 C1 sign gamma_body_hand sign of airframe roll relative to the right-hand rule
85 A1 expression roll_difference roll difference between the balance frame and the airframe | 83 84 90
86 A1 semantic balance_roll_axis balance coordinate axis for that roll | 90
87 A1 sign balance_roll_turn turning sense of that roll about the positive coordinate direction | 85
88 A2 matrix balance_roll_matrix installation roll matrix | 83 84 86 87
89 C1 semantic balance_normal_axis balance absolute normal
90 C1 semantic balance_axial_axis balance absolute axial
91 C1 semantic balance_lateral_axis balance absolute lateral
92 C1 semantic body_normal_axis_load airframe absolute normal
93 C1 semantic body_axial_axis_load airframe absolute axial
94 C1 semantic body_lateral_axis_load airframe absolute lateral
95 A2 matrix rtp_tz1 alignment matrix of axes and absolute directions | 89 90 91 92 93 94
96 C1 semantic placement_load upright or inverted model installation
97 A2 matrix rtp_tz2 pre-matrix for upright or inverted installation | 92 93 94 96
98 A2 semantic load_r1 R1 in the load product
99 A2 semantic load_r2 R2 in the load product
100 A2 semantic load_r3 R3 in the load product
104 A2 matrix load_matrix total load-transform matrix | 88 95 97 98 99 100
105 C1 semantic xtz_load body-axis load associated with XTZ | 92 93 94
106 C1 semantic ytz_load body-axis load associated with YTZ | 92 93 94
107 C1 semantic ztz_load body-axis load associated with ZTZ | 92 93 94
108 C1 semantic xtp_load balance load associated with XTP | 89 90 91
109 C1 semantic ytp_load balance load associated with YTP | 89 90 91
110 C1 semantic zyp_load balance load associated with ZYP | 89 90 91
111 A2 expression cn_tz body-axis normal force | 104 105 106 107 108 109 110
112 A2 expression ca_tz body-axis axial force | 104 105 106 107 108 109 110
113 A2 expression cz_tz body-axis lateral force | 104 105 106 107 108 109 110
114 A2 expression mn_tz body-axis normal moment | 104 105 106 107 108 109 110
115 A2 expression ma_tz body-axis axial moment | 104 105 106 107 108 109 110
116 A2 expression mz_tz body-axis lateral moment | 104 105 106 107 108 109 110
""",
    BALANCE_TO_WIND: """
117 C2 semantic elastic_conversion conversion direction of the elastic increment
118 C1 semantic wind_axial_forward tunnel absolute axial
119 C1 semantic balance_axial_forward balance absolute axial
120 C1 semantic wind_normal_fd wind-frame absolute normal
121 C1 semantic wind_lateral_fd wind-frame absolute lateral
122 C1 semantic wind_axial_fd wind-frame absolute axial
123 C1 semantic balance_normal_fd balance absolute normal
124 C1 semantic balance_lateral_fd balance absolute lateral
125 C1 semantic balance_axial_fd balance absolute axial
126 A2 matrix rp_tp_fd alignment matrix from balance coordinates to wind coordinates | 120 121 122 123 124 125
127 C2 semantic rp_aligned_frame frame the alignment matrix acts on
128 C1 semantic rp_matched_frame frame whose coordinate directions match it
129 A1 semantic install_absolute_axis absolute axis of a positive installation angle
130 C1 semantic install_axis coordinate axis of a positive installation angle | 123 124 125
131 A1 sign install_turn turning sense of the installation angle about that coordinate axis | 123 124 125 130
132 A2 matrix r_gamma_tp installation-angle rotation matrix | 130 131
133 A2 semantic elastic_r1 R1 in the elastic product
134 A2 semantic elastic_r2 R2 in the elastic product
135 A1 vector balance_forward_vector undeformed balance axial unit vector | 123 124 125
136 C1 semantic alpha_e_bend bend direction of a positive elastic angle of attack
137 C1 semantic beta_e_bend bend direction of a positive elastic sideslip
138 A1 semantic gamma_e_clock twist direction of a positive elastic roll
139 C1 semantic alpha_e_axis rotation axis of the elastic angle of attack | 123 124 125
140 C1 semantic beta_e_axis rotation axis of the elastic sideslip | 123 124 125
141 C1 semantic gamma_e_axis rotation axis of the elastic roll | 123 124 125
142 A1 sign xtp_e_sign sign of the deformed XTP component | 135 136 137 139 140
143 A1 sign ytp_e_sign sign of the deformed YTP component | 135 136 137 139 140
144 A1 sign ztp_e_sign sign of the deformed ZTP component | 135 136 137 139 140
145 A1 vector deformed_axis balance axial vector after the small-angle approximation | 135 136 137 139 140
146 A2 expression xfd_elastic XFD of the deformed axis in the wind frame | 126 132 133 134 145
147 A2 expression yfd_elastic YFD of the deformed axis in the wind frame | 126 132 133 134 145
148 A2 expression zfd_elastic ZFD of the deformed axis in the wind frame | 126 132 133 134 145
149 A1 semantic delta_alpha_component wind-frame component for the elastic angle of attack increment | 120 121 122 155
150 A1 semantic delta_beta_component wind-frame component for the elastic sideslip increment | 120 121 122 156
151 C1 semantic placement_elastic installation reread in the elastic section
152 A1 semantic alpha_e_airspeed whether a positive elastic angle of attack places airspeed above or below the airframe | 136 151
153 A1 semantic beta_e_airspeed whether a positive elastic sideslip places airspeed left or right of the airframe | 137 151
154 A1 sign gamma_e_hand relation of elastic roll to the right-hand rule
155 C1 semantic theory_alpha_dir airspeed direction at a positive theoretical angle of attack
156 C1 semantic theory_beta_dir airspeed direction at a positive theoretical sideslip
157 C1 sign theory_gamma_hand sign of theoretical roll relative to the right-hand rule
158 A1 sign delta_alpha_sign sign of the elastic angle of attack increment | 152 155
159 A1 sign delta_beta_sign sign of the elastic sideslip increment | 153 156
160 A1 sign delta_gamma_sign sign of the elastic roll increment | 154 157
161 A1 sign xfd_sign sign of XFD when the installation angle is zero | 146
162 A1 sign yfd_sign sign of YFD when the installation angle is zero | 147
163 A1 sign zfd_sign sign of ZFD when the installation angle is zero | 148
164 A1 sign delta_alpha_outer outer sign of the elastic angle of attack increment formula | 158
165 A1 sign delta_alpha_num_sign numerator sign of the elastic angle of attack increment | 161 162 163 166
166 A1 semantic delta_alpha_num_axis numerator component of the elastic angle of attack increment | 149
167 A1 sign delta_alpha_den_sign denominator sign of the elastic angle of attack increment | 161 162 163 168
168 A1 semantic delta_alpha_den_axis denominator component of the elastic angle of attack increment | 120 121 122
169 A1 sign delta_beta_outer outer sign of the elastic sideslip increment formula | 159
170 A1 sign delta_beta_num_sign numerator sign of the elastic sideslip increment | 161 162 163 171
171 A1 semantic delta_beta_num_axis numerator component of the elastic sideslip increment | 150
172 A1 sign delta_beta_den_sign denominator sign of the elastic sideslip increment | 161 162 163 173
173 A1 semantic delta_beta_den_axis denominator component of the elastic sideslip increment | 120 121 122
174 A2 expression delta_alpha_formula elastic angle of attack increment written with wind-frame components | 146 147 148 164 165 166 167 168
175 A2 expression delta_beta_formula elastic sideslip increment written with wind-frame components | 146 147 148 169 170 171 172 173
176 A2 expression delta_alpha_elastic elastic angle of attack increment written with elastic angles | 146 147 148 174
177 A2 expression delta_beta_elastic elastic sideslip increment written with elastic angles | 146 147 148 175
178 A1 expression delta_gamma elastic roll increment | 160
179 A1 sign delta_alpha_v sign of the flow compensation for angle of attack | 155
180 A1 sign delta_beta_v sign of the flow compensation for sideslip | 156
""",
    REFERENCE_POINT: """
181 C2 semantic moment_origin location of the body-axis origin
182 A1 expression xr distance from the balance center to the model nose
183 C1 semantic engineering_slot engineering component whose positive direction differs from the mathematical coordinate
184 A1 vector reference_coordinate mathematical body-axis coordinates of the moment reference point | 182 183
185 C2 semantic moment_from start of the moment transfer
186 C2 semantic moment_to end of the moment transfer
187 C2 semantic force_place location of the force that produces the extra moment
188 C2 semantic moment_place location where the extra moment acts
189 A1 sign sign_mcn_ca sign of equivalent moment MCN_CA | 184 185 186
190 A1 sign sign_mcz_ca sign of equivalent moment MCZ_CA | 184 185 186
191 A1 sign sign_mca_cz sign of equivalent moment MCA_CZ | 184 185 186
192 A1 sign sign_mcn_cz sign of equivalent moment MCN_CZ | 184 185 186
193 A1 sign sign_mca_cn sign of equivalent moment MCA_CN | 184 185 186
194 A1 sign sign_mcz_cn sign of equivalent moment MCZ_CN | 184 185 186
195 A2 expression mcn_ca equivalent moment MCN_CA | 189 201 207 213
196 A2 expression mcz_ca equivalent moment MCZ_CA | 190 202 208 213
197 A2 expression mca_cz equivalent moment MCA_CZ | 191 203 209 213
198 A2 expression mcn_cz equivalent moment MCN_CZ | 192 204 210 213
199 A2 expression mca_cn equivalent moment MCA_CN | 193 205 211 213
200 A2 expression mcz_cn equivalent moment MCZ_CN | 194 206 212 213
201 A1 semantic moment_force_mcn_ca numerator force coefficient of equivalent moment MCN_CA
202 A1 semantic moment_force_mcz_ca numerator force coefficient of equivalent moment MCZ_CA
203 A1 semantic moment_force_mca_cz numerator force coefficient of equivalent moment MCA_CZ
204 A1 semantic moment_force_mcn_cz numerator force coefficient of equivalent moment MCN_CZ
205 A1 semantic moment_force_mca_cn numerator force coefficient of equivalent moment MCA_CN
206 A1 semantic moment_force_mcz_cn numerator force coefficient of equivalent moment MCZ_CN
207 A1 semantic moment_arm_mcn_ca numerator coordinate component of equivalent moment MCN_CA | 184
208 A1 semantic moment_arm_mcz_ca numerator coordinate component of equivalent moment MCZ_CA | 184
209 A1 semantic moment_arm_mca_cz numerator coordinate component of equivalent moment MCA_CZ | 184
210 A1 semantic moment_arm_mcn_cz numerator coordinate component of equivalent moment MCN_CZ | 184
211 A1 semantic moment_arm_mca_cn numerator coordinate component of equivalent moment MCA_CN | 184
212 A1 semantic moment_arm_mcz_cn numerator coordinate component of equivalent moment MCZ_CN | 184
213 A1 semantic moment_denominator denominator arm of the equivalent moment coefficient | 182
""",
}

# Direct paths used to show what the graph does and does not claim.
DIRECT_PATHS = (
    ("23", "24", "25", "26"),
    ("31", "32", "33", "34"),
    ("40", "41", "42", "43"),
    ("35", "42", "43", "55", "60", "80"),
    ("60", "63", "70", "80"),
    ("22", "68", "80"),
    ("11", "12", "13", "56", "58", "60"),
    ("83", "85", "87", "88", "104", "111"),
    ("151", "152", "158", "164", "174", "176"),
    ("182", "184", "189", "195"),
)


@dataclass(frozen=True)
class BlankNode:
    process: str
    blank_id: str
    label: str
    error_type: str
    answer_type: str
    target_definition: str
    role: str
    depends_on: tuple[str, ...]
    dependency_depth: int
    min_dependency_depth: int
    future_dependencies: tuple[str, ...]
    direct_dependents: tuple[str, ...]
    ancestors: tuple[str, ...]
    descendants: tuple[str, ...]

    def late_parents(self) -> tuple[str, ...]:
        """Parents whose sentence is later than this blank."""
        _ensure_bound()
        sentences = _SENTENCE_OF[self.process]
        here = sentences[self.blank_id]
        return tuple(parent for parent in self.depends_on if sentences[parent] > here)

    def public_fields(self) -> dict:
        return {
            "depends_on": list(self.depends_on),
            "dependency_depth": self.dependency_depth,
            "min_dependency_depth": self.min_dependency_depth,
            "target_definition": self.target_definition,
            "answer_type": self.answer_type,
            "error_type": self.error_type,
            "future_dependencies": list(self.late_parents()),
            "role": self.role,
            "dependency_edge_types": [_hop_type(parent) for parent in self.depends_on],
            "definition_relations": definition_relations(self.blank_id),
            **rule_mention_fields(self.blank_id),
            **annotation_for(self.blank_id),
        }


def _parse_table(process: str, text: str) -> dict[str, dict]:
    rows = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        left, _, right = line.partition("|")
        blank_id, label, error_type, target, *role_words = left.split()
        role = " ".join(role_words)
        depends_on = tuple(right.split())
        if blank_id in rows:
            raise RuntimeError(f"{process} ID{blank_id} is duplicated.")
        if error_type not in ANSWER_TYPE:
            raise RuntimeError(f"{process} ID{blank_id} has an unrecognized error type {error_type}.")
        if label not in {"C1", "C2", "A1", "A2"}:
            raise RuntimeError(f"{process} ID{blank_id} has an unrecognized label {label}.")
        rows[blank_id] = {
            "process": process,
            "blank_id": blank_id,
            "label": label,
            "error_type": error_type,
            "answer_type": ANSWER_TYPE[error_type],
            "target_definition": target,
            "role": role,
            "depends_on": depends_on,
        }
    start = PROCESS_OFFSET[process] + 1
    removed = REMOVED_SLOTS.get(process, frozenset())
    expected = [str(number) for number in range(start, start + PROCESS_SIZE[process]) if number not in removed]
    if list(rows) != expected:
        missing = [item for item in expected if item not in rows]
        extra = [item for item in rows if item not in expected]
        raise RuntimeError(f"{process} ids are not contiguous; missing {missing}, extra {extra}.")
    return rows


def _closure(process: str, rows: dict[str, dict]) -> dict[str, BlankNode]:
    for blank_id, row in rows.items():
        for parent in row["depends_on"]:
            if parent not in rows:
                raise RuntimeError(f"{process} ID{blank_id} depends on missing ID{parent}.")
            if parent == blank_id:
                raise RuntimeError(f"{process} ID{blank_id} depends on itself.")

    visiting: set[str] = set()
    depth: dict[str, int] = {}
    short: dict[str, int] = {}

    def walk(blank_id: str) -> int:
        if blank_id in depth:
            return depth[blank_id]
        if blank_id in visiting:
            raise RuntimeError(f"The {process} dependency graph has a cycle through ID{blank_id}.")
        visiting.add(blank_id)
        parents = rows[blank_id]["depends_on"]
        if parents:
            depth[blank_id] = 1 + max(walk(parent) for parent in parents)
            short[blank_id] = 1 + min(short[parent] for parent in parents)
        else:
            depth[blank_id] = 0
            short[blank_id] = 0
        visiting.remove(blank_id)
        return depth[blank_id]

    for blank_id in rows:
        walk(blank_id)

    children: dict[str, list[str]] = {blank_id: [] for blank_id in rows}
    for blank_id, row in rows.items():
        for parent in row["depends_on"]:
            children[parent].append(blank_id)

    def reachable(blank_id: str, direction: dict[str, list[str]]) -> tuple[str, ...]:
        seen: list[str] = []
        stack = list(direction[blank_id])
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            seen.append(current)
            stack.extend(direction[current])
        return tuple(sorted(seen, key=int))

    parents_of = {
        blank_id: list(rows[blank_id]["depends_on"]) for blank_id in rows
    }
    nodes = {}
    for blank_id, row in rows.items():
        future = tuple(
            parent for parent in row["depends_on"] if int(parent) > int(blank_id)
        )
        nodes[blank_id] = BlankNode(
            process=process,
            blank_id=blank_id,
            label=row["label"],
            error_type=row["error_type"],
            answer_type=row["answer_type"],
            target_definition=row["target_definition"],
            role=row["role"],
            depends_on=row["depends_on"],
            dependency_depth=depth[blank_id],
            min_dependency_depth=short[blank_id],
            future_dependencies=future,
            direct_dependents=tuple(sorted(children[blank_id], key=int)),
            ancestors=reachable(blank_id, parents_of),
            descendants=reachable(blank_id, children),
        )
    return nodes


def build_graph() -> dict[str, dict[str, BlankNode]]:
    return {
        process: _closure(process, _parse_table(process, text))
        for process, text in _TABLES.items()
    }


def _sentence_break(chunk: str) -> bool:
    """A Chinese period, or an ASCII period before whitespace, closes a sentence."""
    if "。" in chunk:
        return True
    for index, char in enumerate(chunk):
        if char != ".":
            continue
        nxt = chunk[index + 1] if index + 1 < len(chunk) else ""
        if nxt in ("", " ", "\n", "\t"):
            return True
    return False


def _sentence_index(text: str, offset: int) -> dict[str, int]:
    """Public id to sentence index, using the same period rule as the cloze."""
    groups: list[list[str]] = []
    current: list[str] = []
    cursor = 0
    for match in _SLOT.finditer(text):
        if _sentence_break(text[cursor:match.start()]) and current:
            groups.append(current)
            current = []
        current.append(match.group(1))
        cursor = match.end()
    if current:
        groups.append(current)
    located: dict[str, int] = {}
    for index, group in enumerate(groups):
        for local in group:
            public = str(int(local) + offset)
            if public in located:
                raise RuntimeError(f"The problem text repeats ID{public}.")
            located[public] = index
    return located


def _ensure_bound() -> None:
    """Bind this module's sentence index.

    ``python -m`` loads this file as ``__main__`` and also as the package
    module. Binding has to happen on the copy that owns the graph being read.
    """
    if _SENTENCE_OF:
        return
    from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET, PROCESS_TEXT

    if _SENTENCE_OF:
        return
    bind_reading_order(PROCESS_TEXT, PROCESS_OFFSET)


def bind_reading_order(process_text: dict[str, str], offsets: dict[str, int]) -> None:
    """Record where each blank sits, and reject a parent asked later or together."""
    global _SENTENCE_OF
    located = {
        process: _sentence_index(text, offsets[process])
        for process, text in process_text.items()
    }
    late: list[str] = []
    same: list[str] = []
    for process, nodes in GRAPH.items():
        sentences = located.get(process)
        if sentences is None:
            raise RuntimeError(f"The problem text is missing the subprocess {process}.")
        if set(sentences) != set(nodes):
            missing = sorted(set(nodes) - set(sentences), key=int)
            extra = sorted(set(sentences) - set(nodes), key=int)
            raise RuntimeError(f"The {process} blanks do not match the dependency graph; missing {missing}, extra {extra}.")
        for item in nodes.values():
            here = sentences[item.blank_id]
            for parent in item.depends_on:
                parent_at = sentences[parent]
                if parent_at > here:
                    late.append(f"ID{item.blank_id}←ID{parent}")
                elif parent_at == here:
                    same.append(f"ID{item.blank_id}←ID{parent}")
    if late or same:
        parts = []
        if late:
            parts.append("appears later: " + ", ".join(late))
        if same:
            parts.append("in the same sentence: " + ", ".join(same))
        raise RuntimeError("The problem text still has a dependency that cannot enter the prefix. " + " ".join(parts))
    _SENTENCE_OF = located


GRAPH = build_graph()
NODE_BY_ID = {
    item.blank_id: item
    for nodes in GRAPH.values()
    for item in nodes.values()
}
# φ already reads these blanks. The prefix graph leaves them off because the
# sentence is later or the blank sits in another subprocess.
PHI_READS = {
    "43": ("5", "7"),
    "88": ("89", "91"),
    "132": ("89", "90", "91"),
}


def diagnosis_parents(node_id: str) -> tuple[str, ...]:
    """Prefix parents plus the blanks φ reads that the sentence order cannot list."""
    node = NODE_BY_ID.get(str(node_id))
    if node is None:
        return ()
    return tuple(dict.fromkeys((*node.depends_on, *PHI_READS.get(str(node_id), ()))))


def _diagnosis_descendants() -> dict[str, tuple[str, ...]]:
    children: dict[str, list[str]] = {node_id: [] for node_id in NODE_BY_ID}
    for node_id in NODE_BY_ID:
        for parent in diagnosis_parents(node_id):
            if parent in children:
                children[parent].append(node_id)
    found = {}
    for node_id in NODE_BY_ID:
        seen: list[str] = []
        stack = list(children[node_id])
        visited: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            seen.append(current)
            stack.extend(children[current])
        found[node_id] = tuple(sorted(seen, key=int))
    return found


DIAGNOSIS_DESCENDANTS = _diagnosis_descendants()


def diagnosis_descendants(node_id: str) -> tuple[str, ...]:
    return DIAGNOSIS_DESCENDANTS.get(str(node_id), ())


if len(NODE_BY_ID) != sum(len(nodes) for nodes in GRAPH.values()):
    raise RuntimeError("Blank ids repeat across subprocesses, so the definition graph cannot be traced by id.")


# Reads are the blanks that ask the definition itself. Scope guards are blanks
# that must keep this definition off another frame; their gold answer stays put.
# Alignment guards are the axis-matching matrices that must not absorb installation.
_FRAME_AXES = {
    "wind": ("2", "3", "4", "120", "121", "122"),
    "body": ("5", "6", "7", "92", "93", "94"),
    "balance": ("89", "90", "91", "123", "124", "125"),
}
_DEFINITIONS = (
    {
        "key": "theory_alpha",
        "name": "positive direction of the theoretical angle of attack",
        "standard": "positive when airspeed points downward on the airframe",
        "perturbed": "positive when airspeed points upward on the airframe",
        "reads": ("21", "155"),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "theory_beta",
        "name": "positive direction of the theoretical sideslip",
        "standard": "positive when airspeed points right on the airframe",
        "perturbed": "positive when airspeed points left on the airframe",
        "reads": ("29", "156"),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "theory_gamma",
        "name": "positive direction of the theoretical roll",
        "standard": "right wing downward agrees with the right-hand rule",
        "perturbed": "right wing upward agrees with the right-hand rule",
        "reads": ("37", "157"),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "mechanism_alpha",
        "name": "positive direction of the mechanism angle of attack",
        "standard": "positive when airspeed points downward on the airframe",
        "perturbed": "positive when airspeed points upward on the airframe",
        "reads": ("20",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "mechanism_beta",
        "name": "positive direction of the mechanism sideslip",
        "standard": "positive when airspeed points right on the airframe",
        "perturbed": "positive when airspeed points left on the airframe",
        "reads": ("28",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "mechanism_roll",
        "name": "positive direction of the mechanism roll",
        "standard": "right wing downward agrees with the right-hand rule about the airframe axis",
        "perturbed": "right wing upward agrees with the right-hand rule about the airframe axis",
        "reads": ("35", "84"),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "placement",
        "name": "upright or inverted installation",
        "standard": "upright",
        "perturbed": "inverted",
        "reads": ("11", "19", "27", "96", "151"),
        "guard_frames": ("wind", "balance"),
        "alignment_guards": ("8", "95", "126"),
    },
    {
        "key": "body_handedness",
        "name": "handedness of the body-axis frame",
        "standard": "right-handed; XTZ points forward on the airframe and YTZ points upward on the airframe",
        "perturbed": "left-handed; XTZ still points forward on the airframe and YTZ still points upward on the airframe",
        "reads": ("5", "6", "7", "92", "93", "94"),
        "guard_frames": ("wind", "balance"),
        "alignment_guards": (),
    },
    {
        "key": "euler_order",
        "name": "order of the mechanism angles",
        "standard": "angle of attack, sideslip, roll",
        "perturbed": "sideslip, angle of attack, roll",
        "reads": ("16", "17", "18"),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "balance_install",
        "name": "positive direction of the balance installation angle",
        "standard": "same as the right-hand rule about the balance axis",
        "perturbed": "opposite the right-hand rule about the balance axis",
        "reads": ("83",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "wind_axes",
        "name": "wind-frame coordinate axes",
        "standard": "YFD points upward along the tunnel normal",
        "perturbed": "YFD points right along the tunnel lateral",
        "reads": ("2", "3", "4", "120", "121", "122"),
        "guard_frames": ("body", "balance"),
        "alignment_guards": (),
    },
    {
        "key": "balance_handedness",
        "name": "handedness of the balance frame",
        "standard": "right-handed; XTP points forward on the balance and YTP points upward on the balance",
        "perturbed": "left-handed; XTP still points forward on the balance and YTP still points upward on the balance",
        "reads": ("89", "90", "91", "123", "124", "125"),
        "guard_frames": ("wind", "body"),
        "alignment_guards": (),
    },
    {
        "key": "elastic_alpha",
        "name": "positive direction of the elastic angle of attack",
        "standard": "positive when the balance axis bends upward",
        "perturbed": "positive when the balance axis bends downward",
        "reads": ("136",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "elastic_beta",
        "name": "positive direction of the elastic sideslip",
        "standard": "positive when the balance axis bends left",
        "perturbed": "positive when the balance axis bends right",
        "reads": ("137",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "elastic_gamma",
        "name": "positive direction of the elastic roll",
        "standard": "positive when clockwise looking along the balance axis",
        "perturbed": "positive when counterclockwise looking along the balance axis",
        "reads": ("138",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "flow_alpha",
        "name": "positive direction of the flow-offset angle of attack",
        "standard": "positive when airspeed points downward on the airframe",
        "perturbed": "positive when airspeed points upward on the airframe",
        "reads": ("179",),
        "guard_frames": (),
        "alignment_guards": (),
    },
    {
        "key": "flow_beta",
        "name": "positive direction of the flow-offset sideslip",
        "standard": "positive when airspeed points right on the airframe",
        "perturbed": "positive when airspeed points left on the airframe",
        "reads": ("180",),
        "guard_frames": (),
        "alignment_guards": (),
    },
)
# Fixed conventions. They stay in the declaration. Recall blanks in the process can be dropped by class.
_CONVENTIONS = (
    {"key": "transform_convention", "name": "transform convention"},
    {"key": "attitude_frame", "name": "frame of the rotation"},
    {"key": "rotation_axis", "name": "roll rotation axis"},
    {"key": "coordinate_axis", "name": "coordinate axis of the rotation"},
    {"key": "level_frame", "name": "intermediate airspeed frame"},
    {"key": "alpha_quotient", "name": "template of the theoretical angle of attack"},
    {"key": "beta_quotient", "name": "template of the theoretical sideslip"},
    {"key": "delta_alpha_quotient", "name": "template of the elastic angle of attack increment"},
    {"key": "delta_beta_quotient", "name": "template of the elastic sideslip increment"},
    {"key": "load_mapping", "name": "correspondence of force and moment components"},
    {"key": "engineering_coordinate", "name": "positive direction of the engineering coordinates"},
    {"key": "moment_transfer", "name": "moment transfer"},
)
# Recall: the question only confirms or restates a rule. The no-rule-mention build drops these blanks by definition.
# Method hint: the stem states the method. The blank stays a scored target and is kept by default.
_RULE_MENTION_GROUPS = (
    (("16", "17", "18"), "recall", "euler_order"),
    (("11", "19", "27", "96", "151"), "recall", "placement"),
    (("21", "149", "155"), "recall", "theory_alpha"),
    (("29", "150", "156"), "recall", "theory_beta"),
    (("37", "157"), "recall", "theory_gamma"),
    (("35", "42", "84"), "recall", "mechanism_roll"),
    (("85",), "method", "mechanism_roll"),
    (("83", "87", "129", "131"), "recall", "balance_install"),
    (("136", "139"), "recall", "elastic_alpha"),
    (("176",), "method", "elastic_alpha"),
    (("137", "140"), "recall", "elastic_beta"),
    (("177",), "method", "elastic_beta"),
    (("138", "141", "154"), "recall", "elastic_gamma"),
    (("178",), "method", "elastic_gamma"),
    (("179",), "recall", "flow_alpha"),
    (("180",), "recall", "flow_beta"),
    (("44", "45", "46"), "recall", "transform_convention"),
    (("23", "31", "40"), "recall", "attitude_frame"),
    (("41",), "recall", "rotation_axis"),
    (("86", "130"), "recall", "coordinate_axis"),
    (("57",), "method", "level_frame"),
    (("68", "69", "70", "71", "72", "73"), "recall", "alpha_quotient"),
    (("74", "75", "76", "77", "78", "79"), "recall", "beta_quotient"),
    (("164", "165", "166", "167", "168"), "recall", "delta_alpha_quotient"),
    (("174",), "method", "delta_alpha_quotient"),
    (("169", "170", "171", "172", "173"), "recall", "delta_beta_quotient"),
    (("175",), "method", "delta_beta_quotient"),
    (("105", "106", "107", "108", "109", "110"), "recall", "load_mapping"),
    (("183",), "recall", "engineering_coordinate"),
    (("185", "186", "187", "188"), "recall", "moment_transfer"),
)
_EFFECTS: dict | None = None


def _rule_definition_names() -> dict[str, str]:
    names = {spec["key"]: spec["name"] for spec in _DEFINITIONS}
    names.update({spec["key"]: spec["name"] for spec in _CONVENTIONS})
    return names


def _rule_mentions() -> dict[str, tuple[str, str]]:
    names = _rule_definition_names()
    found: dict[str, tuple[str, str]] = {}
    for blank_ids, kind, key in _RULE_MENTION_GROUPS:
        if kind not in {"recall", "method"}:
            raise RuntimeError(f"Unrecognized rule-mention kind {kind}.")
        if key not in names:
            raise RuntimeError(f"Rule-mention definition {key} has no name.")
        for blank_id in blank_ids:
            if blank_id in found:
                raise RuntimeError(f"ID{blank_id} is marked as a rule mention twice.")
            if blank_id not in NODE_BY_ID:
                raise RuntimeError(f"Rule-mention ID{blank_id} is not in the dependency graph.")
            found[blank_id] = (kind, key)
    kept = {
        "36", "38", "39", "43", "47", "48", "49", "50", "51", "55",
        "80", "81", "88", "97", "98", "99", "100", "104",
        "111", "112", "113", "114", "115", "116",
        "132", "133", "134", "152", "153", "158", "159",
        "160", "161", "162", "163",
        "181", "182", "184",
        "189", "190", "191", "192", "193", "194",
        "195", "196", "197", "198", "199", "200",
        "201", "202", "203", "204", "205", "206",
        "207", "208", "209", "210", "211", "212", "213",
    }
    recalled = {blank_id for blank_id, (kind, _key) in found.items() if kind == "recall"}
    overlap = sorted(kept & recalled, key=int)
    if overlap:
        raise RuntimeError(f"These derivation blanks were marked as recall questions: {', '.join(overlap)}")
    return found


RULE_MENTIONS = _rule_mentions()
RULE_DEFINITION_NAMES = _rule_definition_names()


def rule_mention_fields(blank_id: str) -> dict[str, str]:
    found = RULE_MENTIONS.get(str(blank_id))
    if found is None:
        return {"rule_mention": "", "rule_definition": "", "rule_definition_name": ""}
    kind, key = found
    return {
        "rule_mention": kind,
        "rule_definition": key,
        "rule_definition_name": RULE_DEFINITION_NAMES[key],
    }


def _rule_mention_catalog() -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for blank_id, (kind, key) in RULE_MENTIONS.items():
        entry = catalog.setdefault(key, {
            "key": key,
            "name": RULE_DEFINITION_NAMES[key],
            "recall": [],
            "method": [],
        })
        entry[kind].append(blank_id)
    for entry in catalog.values():
        entry["recall"] = sorted(entry["recall"], key=int)
        entry["method"] = sorted(entry["method"], key=int)
    return catalog


def mentions_to_drop(definitions: tuple[str, ...] | list[str] | None = None) -> tuple[str, ...]:
    """Recall-question ids. Given definition keys, drop only the confirm and recall questions of those definitions."""
    selected = None if definitions is None else set(definitions)
    unknown = [] if selected is None else sorted(selected - set(RULE_DEFINITION_NAMES))
    if unknown:
        raise KeyError(f"No such definition classes: {', '.join(unknown)}")
    found = [
        blank_id
        for blank_id, (kind, key) in RULE_MENTIONS.items()
        if kind == "recall" and (selected is None or key in selected)
    ]
    return tuple(sorted(found, key=int))


def _hop_type(parent_id: str) -> str:
    """What the parent answer contributes to the child."""
    parent = NODE_BY_ID[str(parent_id)]
    target = parent.target_definition
    if parent.error_type == "sign":
        return "sign"
    if parent.error_type == "matrix":
        return "matrix"
    if parent.error_type == "vector":
        return "value"
    if parent.error_type == "expression":
        return "formula"
    if (
        target.startswith("euler")
        or target.startswith("product")
        or "transform" in target
        or target.endswith("rotation")
    ):
        return "method"
    if parent.label == "C2":
        return "method"
    if any(piece in target for piece in ("axis", "frame", "placement", "hand")):
        return "scope"
    if parent.label.startswith("A"):
        return "selector"
    return "value"


def _direct_type(blank_id: str, changed: bool) -> str:
    item = NODE_BY_ID[blank_id]
    target = item.target_definition
    method = (
        item.label == "C2"
        or target.startswith("euler")
        or target.startswith("product")
        or "transform" in target
        or target.endswith("rotation")
    )
    if method:
        return "direct_method"
    if not changed:
        return "direct_scope"
    return "direct_answer"


def _error_candidate(relation_type: str) -> str:
    return {
        "direct_answer": "definition_unused",
        "direct_method": "method_unused",
        "direct_scope": "scope_leakage",
        "downstream": "propagated_candidate",
    }[relation_type]


def _semantic_code(relation_type: str) -> int:
    return {
        "direct_answer": 1,
        "direct_method": 2,
        "direct_scope": 3,
        "downstream": 4,
    }[relation_type]


def _effect_code(relation_type: str, changed: bool, edge_types: list[str]) -> int:
    """0 stable, 1 direct gold change, 3 downstream gold change, 4 formula or matrix change."""
    if not changed:
        return 0
    if relation_type != "downstream":
        return 1
    if any(kind in edge_types for kind in ("formula", "matrix")):
        return 4
    return 3


def _relation(
    definition_id: str,
    blank_id: str,
    relation_type: str,
    changed: bool,
    path: list[str],
) -> dict:
    edge_types = [_hop_type(parent) for parent in path[:-1]]
    return {
        "definition_id": definition_id,
        "question_id": blank_id,
        "relation_type": relation_type,
        "direct": relation_type != "downstream",
        "semantic_relation": True,
        "semantic_code": _semantic_code(relation_type),
        "ground_truth_changes": changed,
        "effect_code": _effect_code(relation_type, changed, edge_types),
        "requires_scope_binding": relation_type == "direct_scope",
        "source_question": path[0],
        "path": path,
        "edge_types": edge_types,
        "distance": len(path) - 1,
        "error_candidate": _error_candidate(relation_type),
    }


def _mutate(spec: dict, key: str) -> None:
    from copy import deepcopy

    if key == "theory_alpha":
        spec["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}
    elif key == "theory_beta":
        spec["beta"] = {"airspeed_toward": "body_left", "sign": "positive"}
    elif key == "theory_gamma":
        spec["gamma"] = {"right_wing_toward": "body_up", "sign": "positive"}
    elif key == "mechanism_alpha":
        spec["alpha_m"] = {"airspeed_toward": "body_up", "sign": "positive"}
    elif key == "mechanism_beta":
        spec["beta_m"] = {"airspeed_toward": "body_left", "sign": "positive"}
    elif key == "mechanism_roll":
        spec["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    elif key == "placement":
        spec["placement"] = "inverted"
    elif key == "body_handedness":
        spec["body_frame"] = {
            "chirality": "left",
            "euler_order": ["alpha", "beta", "gamma"],
            "XTZ": {"along": "body_axial", "points": "body_forward"},
            "YTZ": {"along": "body_normal", "points": "body_up"},
        }
    elif key == "euler_order":
        frame = deepcopy(spec["body_frame"])
        frame["euler_order"] = ["beta", "alpha", "gamma"]
        spec["body_frame"] = frame
    elif key == "balance_install":
        spec["gamma_tp"] = {"sense": "opposite"}
    elif key == "wind_axes":
        spec["wind_frame"] = {
            "chirality": "right",
            "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
            "YFD": {"along": "tunnel_lateral", "points": "tunnel_right"},
        }
    elif key == "balance_handedness":
        spec["balance_frame"] = {
            "chirality": "left",
            "XTP": {"along": "balance_axial", "points": "balance_forward"},
            "YTP": {"along": "balance_normal", "points": "balance_up"},
        }
    elif key == "elastic_alpha":
        spec["alpha_e"] = {"axial_bends": "balance_down", "sign": "positive"}
    elif key == "elastic_beta":
        spec["beta_e"] = {"axial_bends": "balance_right", "sign": "positive"}
    elif key == "elastic_gamma":
        spec["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    elif key == "flow_alpha":
        spec["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    elif key == "flow_beta":
        spec["beta_v"] = {"airspeed_toward": "body_left", "sign": "positive"}
    else:
        raise KeyError(key)


def _gold_by_definition() -> dict[str, set[str]]:
    """Blanks whose stored answer changes under each isolated definition."""
    from copy import deepcopy

    from domains.wind_tunnel.dataset_generate import PROCESSES, _public_id
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

    def collect(spec: dict) -> dict[str, str]:
        found = {}
        for process, generate in PROCESSES:
            payload = generate(deepcopy(spec))
            for item in payload["items"]:
                found[_public_id(process, item["manuscript_id"])] = item["answer"]
        return found

    base = collect(STANDARD_SPEC)
    changed = {}
    for spec in _DEFINITIONS:
        trial = deepcopy(STANDARD_SPEC)
        _mutate(trial, spec["key"])
        found = collect(trial)
        changed[spec["key"]] = {
            blank_id for blank_id, answer in base.items() if answer != found[blank_id]
        }
    return changed


def _changed_path(blank_id: str, changed: set[str], seeds: set[str]) -> list[str]:
    """Shortest walk back through answers that change, until a definition entry."""
    memo: dict[str, list[str]] = {}
    visiting: set[str] = set()

    def walk(current: str) -> list[str]:
        if current in memo:
            return memo[current]
        if current in visiting:
            raise RuntimeError(f"Definition traceback cycles at ID{current}.")
        if current in seeds:
            memo[current] = [current]
            return memo[current]
        visiting.add(current)
        parents = [
            parent for parent in NODE_BY_ID[current].depends_on if parent in changed
        ]
        if not parents:
            raise RuntimeError(f"The gold answer of ID{current} changed, but no changed upstream question can be traced.")
        best = None
        for parent in parents:
            candidate = walk(parent) + [current]
            if best is None or (len(candidate), candidate) < (len(best), best):
                best = candidate
        visiting.remove(current)
        memo[current] = best
        return best

    return walk(blank_id)


def _paths_from(seeds: list[str]) -> dict[str, list[str]]:
    paths = {seed: [seed] for seed in seeds}
    queue = list(seeds)
    while queue:
        current = queue.pop(0)
        for child in NODE_BY_ID[current].direct_dependents:
            if child in paths:
                continue
            paths[child] = paths[current] + [child]
            queue.append(child)
    return paths


def _build_effects() -> dict:
    changed_of = _gold_by_definition()
    by_definition = {}
    by_question: dict[str, list[dict]] = {blank_id: [] for blank_id in NODE_BY_ID}
    for spec in _DEFINITIONS:
        key = spec["key"]
        definition_id = f"D_{key}"
        changed = changed_of[key]
        missing = [blank_id for blank_id in spec["reads"] if blank_id not in NODE_BY_ID]
        if missing:
            raise RuntimeError(f"{definition_id} is missing its entry questions: {missing}")
        for blank_id in spec["alignment_guards"]:
            if blank_id not in NODE_BY_ID:
                raise RuntimeError(f"{definition_id} is missing its scope question: {blank_id}")
        seeds = {
            blank_id for blank_id in changed
            if blank_id in spec["reads"] or not any(
                parent in changed for parent in NODE_BY_ID[blank_id].depends_on
            )
        }
        relations = {}
        paths = _paths_from(sorted(seeds, key=int))
        for blank_id in sorted(seeds, key=int):
            relations[blank_id] = _relation(
                definition_id, blank_id, _direct_type(blank_id, True), True, [blank_id],
            )
        for blank_id, path in paths.items():
            if blank_id in relations:
                continue
            if blank_id in changed:
                path = _changed_path(blank_id, changed, seeds)
            relations[blank_id] = _relation(
                definition_id, blank_id, "downstream", blank_id in changed, path,
            )
        for blank_id in spec["reads"]:
            if blank_id in relations:
                continue
            relations[blank_id] = _relation(
                definition_id, blank_id, _direct_type(blank_id, False), False, [blank_id],
            )
        guards = list(spec["alignment_guards"])
        for frame in spec["guard_frames"]:
            guards.extend(_FRAME_AXES[frame])
        for blank_id in guards:
            if blank_id in relations:
                continue
            relations[blank_id] = _relation(
                definition_id, blank_id, "direct_scope", False, [blank_id],
            )
        for item in relations.values():
            path = item["path"]
            for parent, child in zip(path, path[1:]):
                if parent not in NODE_BY_ID[child].depends_on:
                    raise RuntimeError(
                        f"{definition_id} traceback ID{parent}->ID{child} is not an answer-dependency edge."
                    )
        accounted = {
            blank_id for blank_id, item in relations.items() if item["ground_truth_changes"]
        }
        if accounted != changed:
            missing_ids = sorted(changed - accounted, key=int)
            extra_ids = sorted(accounted - changed, key=int)
            raise RuntimeError(
                f"Not every answer change of {definition_id} entered the definition graph; "
                f"missing {missing_ids}, extra {extra_ids}."
            )
        ordered = [relations[blank_id] for blank_id in sorted(relations, key=int)]
        for item in ordered:
            by_question[item["question_id"]].append(item)
        by_definition[key] = {
            "id": definition_id,
            "key": key,
            "name": spec["name"],
            "perturbable": True,
            "standard": spec["standard"],
            "perturbed": spec["perturbed"],
            "direct_questions": [
                item["question_id"] for item in ordered if item["direct"]
            ],
            "changed_questions": sorted(changed, key=int),
            "questions": {item["question_id"]: item for item in ordered},
        }
    for relations in by_question.values():
        relations.sort(key=lambda item: (item["distance"], item["definition_id"]))
    return {"by_definition": by_definition, "by_question": by_question}


def effects() -> dict:
    global _EFFECTS
    if _EFFECTS is None:
        _EFFECTS = _build_effects()
    return _EFFECTS


def definition_relations(blank_id: str) -> list[dict]:
    """Definition edges that enter this blank or reach it through answer edges."""
    return list(effects()["by_question"].get(str(blank_id), ()))


_FACTOR_OF_SAMPLE = {
    "alpha": "theory_alpha",
    "beta": "theory_beta",
    "body": "body_handedness",
    "roll": "mechanism_roll",
    "order": "euler_order",
    "mount": "placement",
}


def _definition_spec(definition_id: str) -> dict:
    key = definition_id.removeprefix("D_")
    found = effects()["by_definition"].get(key)
    if found is None:
        raise KeyError(definition_id)
    return found


def _observation_correct(observation) -> bool | None:
    if observation is None:
        return None
    if isinstance(observation, bool):
        return observation
    if observation.get("normalized") is not None:
        return bool(observation["normalized"])
    if "correct" in observation:
        return bool(observation["correct"])
    if "exact" in observation:
        return bool(observation["exact"])
    return None


def _where_error_appears(steps: list[dict]) -> str:
    """Where, walking forward from the definition entry, the first wrong answer sits.

    A missing blank before that wrong answer blocks the claim: the deleted
    question may already have been wrong. A path with no wrong answer stays
    ``none`` even if an intermediate was not in the manuscript.
    """
    seen_gap = False
    for step in steps:
        if step["correct"] is None:
            seen_gap = True
            continue
        if step["correct"] is False:
            return "incomplete" if seen_gap else step["position"]
    if steps and steps[-1]["correct"] is None:
        return "incomplete"
    return "none"


def trace_error(
    blank_id: str,
    observations: dict | None = None,
    factor: str | None = None,
) -> dict:
    """Walk each definition path forward and record whether each step is correct.

    ``observations`` maps a blank id to ``True``/``False`` or to a scored blank
    dict. Missing blanks stay unobserved. The path starts at the question where
    that definition enters and ends at ``blank_id``.
    """
    blank_id = str(blank_id)
    if blank_id not in NODE_BY_ID:
        raise KeyError(f"There is no ID{blank_id}.")
    observations = observations or {}
    traces = []
    for relation in definition_relations(blank_id):
        path = relation["path"]
        edge_types = relation["edge_types"]
        if path[-1] != blank_id:
            raise RuntimeError(f"The path of {relation['definition_id']} does not stop at ID{blank_id}.")
        steps = []
        for index, question_id in enumerate(path):
            observation = observations.get(question_id)
            if index == 0 and index == len(path) - 1:
                position = "direct"
            elif index == 0:
                position = "origin"
            elif index == len(path) - 1:
                position = "target"
            else:
                position = "intermediate"
            steps.append({
                "question_id": question_id,
                "role": NODE_BY_ID[question_id].role,
                "position": position,
                "correct": _observation_correct(observation),
                "answer": observation.get("answer") if isinstance(observation, dict) else None,
                "prediction": observation.get("prediction") if isinstance(observation, dict) else None,
                "edge_from_previous": None if index == 0 else edge_types[index - 1],
            })
        spec = _definition_spec(relation["definition_id"])
        first_wrong = next(
            (step["question_id"] for step in steps if step["correct"] is False),
            None,
        )
        intermediates = [step for step in steps if step["position"] == "intermediate"]
        traces.append({
            "definition_id": relation["definition_id"],
            "definition_key": spec["key"],
            "definition_name": spec["name"],
            "definition_standard": spec["standard"],
            "definition_perturbed": spec["perturbed"],
            "perturbed_in_this_run": factor == spec["key"] if factor else None,
            "relation_type": relation["relation_type"],
            "ground_truth_changes": relation["ground_truth_changes"],
            "error_candidate": relation["error_candidate"],
            "source_question": relation["source_question"],
            "steps": steps,
            "first_wrong": first_wrong,
            "where_error_appears": _where_error_appears(steps),
            "origin_correct": steps[0]["correct"],
            "intermediates_correct": (
                None if not intermediates
                else None if any(step["correct"] is None for step in intermediates)
                else all(step["correct"] for step in intermediates)
            ),
            "target_correct": steps[-1]["correct"],
        })
    traces.sort(key=lambda item: (
        item["perturbed_in_this_run"] is not True,
        not item["ground_truth_changes"],
        {"direct": 0, "origin": 0, "intermediate": 1, "target": 2, "incomplete": 3, "none": 4}[
            item["where_error_appears"]
        ],
        len(item["steps"]),
        item["definition_id"],
    ))
    target = NODE_BY_ID[blank_id]
    return {
        "question_id": blank_id,
        "role": target.role,
        "factor": factor,
        "target_correct": _observation_correct(observations.get(blank_id)),
        "traces": traces,
    }


def _step_mark(correct: bool | None) -> str:
    if correct is True:
        return "right"
    if correct is False:
        return "wrong"
    return "missing"


_ENTRY_OF: dict[str, tuple[str, ...]] | None = None


def _entry_of() -> dict[str, tuple[str, ...]]:
    """Questions at which a definition enters, keyed by blank id."""
    global _ENTRY_OF
    if _ENTRY_OF is not None:
        return _ENTRY_OF
    found: dict[str, set[str]] = {}
    for key, spec in effects()["by_definition"].items():
        for item in spec["questions"].values():
            found.setdefault(item["source_question"], set()).add(key)
    _ENTRY_OF = {
        blank_id: tuple(sorted(keys))
        for blank_id, keys in found.items()
    }
    return _ENTRY_OF


def _node_text(blank_id: str, correct: bool | None) -> str:
    text = f"ID{blank_id}({_step_mark(correct)})"
    entries = _entry_of().get(str(blank_id), ())
    if not entries:
        return text
    return f"{text}[entry:{'+'.join(entries)}]"


def _prefix_mode(condition: str) -> str:
    """Gold prefix has no model-error propagation. Model prefix does."""
    if condition in {"self-trunc", "self-keep", "session-model"}:
        return "model"
    return "gold"


def _ids(values) -> list[str]:
    return sorted({str(item) for item in values}, key=int)


def _answer_state(observations: dict, blank_id: str) -> bool | None:
    if blank_id not in observations:
        return None
    return _observation_correct(observations.get(blank_id))


def _wrong_carriers(blank_id: str, observations: dict, seen: set[str] | None = None) -> list[str]:
    """Wrong answers that reach this blank through direct edges, stopping at a correct parent."""
    seen = set() if seen is None else seen
    found: list[str] = []
    node = NODE_BY_ID.get(blank_id)
    if node is None:
        return found
    for parent in node.depends_on:
        if parent in seen or parent not in NODE_BY_ID:
            continue
        if _answer_state(observations, parent) is not False:
            continue
        seen.add(parent)
        found.append(parent)
        found.extend(_wrong_carriers(parent, observations, seen))
    return found


def _first_errors(observations: dict) -> dict[str, tuple[str, ...]]:
    """Earliest wrong answers on the direct-dependency recursion. Correct parents block a branch."""
    memo: dict[str, tuple[str, ...]] = {}

    def walk(blank_id: str, stack: set[str]) -> tuple[str, ...]:
        if blank_id in memo:
            return memo[blank_id]
        if blank_id in stack or blank_id not in NODE_BY_ID:
            return ()
        stack.add(blank_id)
        upstream: list[str] = []
        for parent in NODE_BY_ID[blank_id].depends_on:
            if _answer_state(observations, parent) is False:
                upstream.extend(walk(parent, stack))
        stack.remove(blank_id)
        if _answer_state(observations, blank_id) is False and not upstream:
            result = (blank_id,)
        else:
            result = tuple(_ids(upstream))
        memo[blank_id] = result
        return result

    for blank_id in observations:
        if blank_id in NODE_BY_ID:
            walk(blank_id, set())
    return memo


def _forward_path(source: str, target: str, observations: dict) -> list[str] | None:
    """Shortest direct-edge path. Intermediate steps must themselves be wrong."""
    previous = {source: None}
    queue = [source]
    while queue:
        current = queue.pop(0)
        if current == target:
            break
        node = NODE_BY_ID.get(current)
        if node is None:
            continue
        for child in node.direct_dependents:
            if child in previous or child not in observations:
                continue
            if child != target and _answer_state(observations, child) is not False:
                continue
            previous[child] = current
            queue.append(child)
    if target not in previous and source != target:
        return None
    path = []
    cursor = target
    while cursor is not None:
        path.append(cursor)
        cursor = previous.get(cursor)
    path.reverse()
    return path


def classify_errors(observations: dict, prefix: str) -> dict:
    """Node table, first-error table, and propagation table for one run.

    Gold prefix records every wrong answer as local_error. Model prefix stops
    attribution at the first wrong answer and still lists every later descendant.
    """
    present = [blank_id for blank_id in observations if blank_id in NODE_BY_ID]
    first = _first_errors(observations) if prefix == "model" else {}
    nodes = []
    first_rows = []
    propagation_rows = []
    for blank_id in sorted(present, key=int):
        state = _answer_state(observations, blank_id)
        entries = list(_entry_of().get(blank_id, ()))
        if prefix == "gold":
            sources = (blank_id,) if state is False else ()
            kind = "local_error" if state is False else "correct"
        else:
            sources = first.get(blank_id, ())
            if state is False and sources == (blank_id,):
                kind = "local_error"
            elif state is False:
                kind = "propagated_error"
            elif state is True and sources:
                kind = "recovered"
            else:
                kind = "correct"
                sources = ()
        paths = []
        if prefix == "model" and sources and kind != "correct":
            for source in sources:
                if source == blank_id:
                    paths.append([blank_id])
                    continue
                found = _forward_path(source, blank_id, observations)
                if found:
                    paths.append(found)
        depth = max((len(path) for path in paths), default=0)
        record = {
            "target_id": f"ID{blank_id}",
            "is_wrong": state is False,
            "first_error_nodes": [f"ID{item}" for item in sources],
            "error_type": kind,
            "direct_error": kind == "local_error",
            "definition_entries": entries,
            "logical_parents": [f"ID{item}" for item in NODE_BY_ID[blank_id].depends_on],
            "process": NODE_BY_ID[blank_id].process,
            "error_category": NODE_BY_ID[blank_id].error_type,
        }
        if kind == "propagated_error" or kind == "recovered":
            record["causal_ancestors"] = [
                f"ID{item}" for item in _ids(_wrong_carriers(blank_id, observations))
            ]
            record["propagation_depth"] = depth
            record["propagation_paths"] = [[f"ID{item}" for item in path] for path in paths]
        if kind == "local_error" and prefix == "model":
            affected = []
            recovered = []
            subprocesses = {NODE_BY_ID[blank_id].process}
            depths = []
            for other in present:
                if other == blank_id or blank_id not in first.get(other, ()):
                    continue
                other_state = _answer_state(observations, other)
                status = "propagated_error" if other_state is False else "recovered"
                path = _forward_path(blank_id, other, observations)
                if not path:
                    continue
                affected.append(other)
                if status == "recovered":
                    recovered.append(other)
                subprocesses.add(NODE_BY_ID[other].process)
                depths.append(len(path))
                propagation_rows.append({
                    "root_error_node": f"ID{blank_id}",
                    "root_entries": entries,
                    "descendant_id": f"ID{other}",
                    "descendant_entries": list(_entry_of().get(other, ())),
                    "path_length": len(path) - 1,
                    "descendant_status": status,
                    "affected_subprocess": NODE_BY_ID[other].process,
                })
            record["affected_descendants"] = [f"ID{item}" for item in _ids(affected)]
            record["recovered_descendants"] = [f"ID{item}" for item in _ids(recovered)]
            record["propagation_width"] = len(affected)
            record["propagation_depth"] = max(depths, default=0)
            record["affected_subprocesses"] = sorted(subprocesses)
        nodes.append(record)
        for source in sources:
            relation = "direct" if source == blank_id else "propagated"
            path = next((item for item in paths if item and item[0] == source), [source] if source == blank_id else None)
            if kind == "correct":
                continue
            first_rows.append({
                "path_id": f"ID{source}->ID{blank_id}",
                "target_id": f"ID{blank_id}",
                "first_error_node": f"ID{source}",
                "first_error_entries": list(_entry_of().get(source, ())),
                "error_category": NODE_BY_ID[source].error_type,
                "direct_or_propagated": relation,
                "target_status": kind,
                "propagation_path": [f"ID{item}" for item in path] if path else [],
            })
    counts = {kind: sum(item["error_type"] == kind for item in nodes) for kind in (
        "local_error", "propagated_error", "recovered", "correct",
    )}
    depths = [
        item["propagation_depth"] for item in nodes
        if item["error_type"] == "propagated_error"
    ]
    summary = {
        "prefix": prefix,
        **counts,
        "multi_source": sum(
            item["error_type"] == "propagated_error" and len(item["first_error_nodes"]) > 1
            for item in nodes
        ),
        "mean_propagation_depth": round(sum(depths) / len(depths), 2) if depths else 0,
        "roots": [
            {
                "root_error_node": item["target_id"],
                "definition_entries": item["definition_entries"],
                "affected_descendants": item.get("affected_descendants", []),
                "recovered_descendants": item.get("recovered_descendants", []),
                "propagation_width": item.get("propagation_width", 0),
                "propagation_depth": item.get("propagation_depth", 0),
                "affected_subprocesses": item.get("affected_subprocesses", []),
            }
            for item in nodes if item["error_type"] == "local_error" and prefix == "model"
        ],
    }
    return {
        "nodes": nodes,
        "first_errors": first_rows,
        "propagation": propagation_rows,
        "summary": summary,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _chain_lines(nodes: list[dict]) -> str:
    lines = []
    for item in nodes:
        if item["error_type"] == "correct":
            continue
        entry = ""
        if item["definition_entries"]:
            entry = "[entry:" + "+".join(item["definition_entries"]) + "] "
        sources = ",".join(item["first_error_nodes"])
        lines.append(f"{item['target_id']} {item['error_type']} {entry}first:{sources}".rstrip())
    return "\n".join(lines)


def write_error_chains(records: list[dict], out: Path) -> Path | None:
    """Write the node, first-error, and propagation tables for one finished run."""
    grouped: dict[tuple[str, str], dict] = {}
    for row in records:
        sample_id = str(row.get("sample_id") or "")
        condition = str(row.get("condition") or "")
        bucket = grouped.setdefault((sample_id, condition), {})
        for blank in row.get("blanks") or []:
            bucket[str(blank["blank_id"])] = blank
    usable = {
        key: observations for key, observations in grouped.items()
        if any(blank_id in NODE_BY_ID for blank_id in observations)
    }
    if not usable:
        return None
    out = Path(out)
    node_rows = []
    first_rows = []
    propagation_rows = []
    summaries = []
    texts = []
    for (sample_id, condition), observations in sorted(usable.items()):
        prefix = _prefix_mode(condition)
        classified = classify_errors(observations, prefix)
        factor = _FACTOR_OF_SAMPLE.get(sample_id.split("-", 1)[0])
        for row in classified["nodes"]:
            row["sample_id"] = sample_id
            row["condition"] = condition
            row["perturbed_definition"] = factor
        for row in classified["first_errors"]:
            row["sample_id"] = sample_id
            row["condition"] = condition
            row["perturbed_definition"] = factor
        for row in classified["propagation"]:
            row["sample_id"] = sample_id
            row["condition"] = condition
            row["perturbed_definition"] = factor
        node_rows.extend(classified["nodes"])
        first_rows.extend(classified["first_errors"])
        propagation_rows.extend(classified["propagation"])
        summary = classified["summary"]
        summary.update({
            "sample_id": sample_id,
            "condition": condition,
            "perturbed_definition": factor,
        })
        summaries.append(summary)
        if len(usable) > 1:
            texts.append(f"{sample_id} {condition}")
        texts.append(_chain_lines(classified["nodes"]))
    _write_jsonl(out / "error_nodes.jsonl", node_rows)
    _write_jsonl(out / "error_first.jsonl", first_rows)
    _write_jsonl(out / "error_propagation.jsonl", propagation_rows)
    (out / "error_summary.json").write_text(
        json.dumps(summaries[0] if len(summaries) == 1 else summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "error_chains.txt").write_text("\n".join(texts).rstrip() + "\n", encoding="utf-8")
    return out / "error_nodes.jsonl"


def observations_from_jsonl(
    path: Path,
    sample_id: str,
    condition: str | None = None,
    run: int | None = 0,
) -> dict[str, dict]:
    """Index scored blanks from one sample and one condition."""
    found: dict[str, dict] = {}
    seen_conditions: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_id") != sample_id:
            continue
        if run is not None and row.get("run") != run:
            continue
        row_condition = row.get("condition")
        if condition is not None and row_condition != condition:
            continue
        if row_condition:
            seen_conditions.add(row_condition)
        for blank in row.get("blanks") or []:
            found[str(blank["blank_id"])] = blank
    if condition is None and len(seen_conditions) > 1:
        names = ", ".join(sorted(seen_conditions))
        raise RuntimeError(f"{sample_id} has more than one condition: {names}. Name one of them.")
    if not found:
        raise RuntimeError(f"No score record was found for {sample_id}.")
    return found


def format_trace(report: dict) -> str:
    """One wrong blank, each definition path walked from its entry."""
    labels = {"origin": "entry", "intermediate": "intermediate", "target": "this question", "direct": "entry and this question"}
    arrival = {
        "direct": "This question reads the definition directly, and the error is in the entry question itself",
        "origin": "The error appears at the definition entry",
        "intermediate": "The error appears at an intermediate question",
        "target": "The entry and the intermediate questions are right, and the error appears in this question",
        "incomplete": "A question on the path is absent from this manuscript, so the entry of the error cannot be judged",
        "none": "No error was recorded on this path",
    }
    lines = [
        f"Wrong question ID{report['question_id']} {report['role']}",
    ]
    if report["target_correct"] is True:
        lines.append("This question is right in the current record. The definition paths that reach it are still listed below.")
    elif report["target_correct"] is False:
        lines.append("This question is wrong in the current record.")
    else:
        lines.append("The current record does not say whether this question is right or wrong.")
    if not report["traces"]:
        lines.append("No definition reaches this question.")
        return "\n".join(lines)
    for trace in report["traces"]:
        mark = ""
        if trace["perturbed_in_this_run"] is True:
            mark = ", the definition perturbed in this run"
        gold = "changes the correct answer" if trace["ground_truth_changes"] else "does not change the correct answer"
        route = " → ".join(f"ID{step['question_id']}" for step in trace["steps"])
        lines.append("")
        lines.append(
            f"{trace['definition_id']} {trace['definition_name']}{mark}. "
            f"{trace['relation_type']}, {gold}."
        )
        lines.append(f"Standard: {trace['definition_standard']}. Perturbed: {trace['definition_perturbed']}.")
        lines.append(route)
        for step in trace["steps"]:
            if step["correct"] is True:
                state = "right"
            elif step["correct"] is False:
                state = "wrong"
            else:
                state = "absent"
            extra = ""
            if step["question_id"] == trace["first_wrong"]:
                extra = (
                    ", first error among the recorded questions"
                    if trace["where_error_appears"] == "incomplete"
                    else ", first error on this path"
                )
            lines.append(
                f"  ID{step['question_id']} {step['role']}  "
                f"{labels[step['position']]}  {state}{extra}"
            )
        lines.append(arrival[trace["where_error_appears"]])
    return "\n".join(lines)


def annotation_for(blank_id: str) -> dict:
    from domains.wind_tunnel.contract_labels import annotation_for as _annotation

    return _annotation(blank_id)


def node(process: str, blank_id: str) -> BlankNode | None:
    return GRAPH.get(process, {}).get(str(blank_id))


def question_fields(process: str, blank_id: str, label: str) -> dict:
    """Fields stored on a generated question. Unknown blanks fail loudly."""
    found = node(process, blank_id)
    if found is None:
        raise RuntimeError(f"The dependency graph has no {process} ID{blank_id}.")
    if found.label != label:
        raise RuntimeError(
            f"{process} ID{blank_id} is labeled {label}, but the dependency graph says {found.label}."
        )
    return found.public_fields()


def dependency_fields(item: dict) -> dict:
    """Copy graph fields onto an evaluation blank without changing its label.

    A stored blank whose label disagrees with the graph is left untouched.
    The batch corpus is older than the current rule graph, and attaching the
    other blank's edges would mark the wrong parents.
    """
    if "depends_on" in item and "target_definition" in item:
        return {key: item[key] for key in PUBLIC_KEYS if key in item}
    found = node(str(item.get("process") or ""), str(item.get("id") or item.get("blank_id") or ""))
    label = item.get("label")
    if found is None or (label not in (None, "") and found.label != label):
        return {}
    return found.public_fields()


def annotate_question(question: dict) -> dict:
    found = node(str(question.get("process") or ""), str(question.get("id") or ""))
    if found is None:
        return question
    if question.get("label") != found.label:
        raise RuntimeError(
            f"{question.get('process')} ID{question.get('id')} is labeled "
            f"{question.get('label')}, but the dependency graph says {found.label}."
        )
    question.update(found.public_fields())
    return question


def annotate_sample(sample: dict) -> dict:
    for question in sample.get("questions") or []:
        annotate_question(question)
    return sample


def distance(process: str, source: str, target: str) -> int | None:
    """Shortest number of direct edges from source to target."""
    source, target = str(source), str(target)
    if node(process, source) is None or node(process, target) is None:
        return None
    if source == target:
        return 0
    seen = {source}
    layer = [source]
    steps = 0
    while layer:
        steps += 1
        nxt = []
        for blank_id in layer:
            for child in node(process, blank_id).direct_dependents:
                if child == target:
                    return steps
                if child not in seen:
                    seen.add(child)
                    nxt.append(child)
        layer = nxt
    return None


def _direct_path(process: str, blank_ids: tuple[str, ...]) -> None:
    for source, target in zip(blank_ids, blank_ids[1:]):
        found = node(process, target)
        if found is None or source not in found.depends_on:
            raise RuntimeError(f"{process} has no direct edge ID{source} -> ID{target}.")


def _check_known_paths() -> None:
    _direct_path(WIND_TO_BODY, ("23", "24", "25", "26"))
    _direct_path(WIND_TO_BODY, ("31", "32", "33", "34"))
    _direct_path(WIND_TO_BODY, ("40", "41", "42", "43"))
    _direct_path(WIND_TO_BODY, ("35", "42", "43", "55", "60", "80"))
    _direct_path(WIND_TO_BODY, ("60", "63", "70", "80"))
    _direct_path(WIND_TO_BODY, ("22", "68", "80"))
    _direct_path(WIND_TO_BODY, ("11", "12", "13", "56", "58", "60"))
    _direct_path(BALANCE_TO_BODY, ("83", "85", "87", "88", "104", "111"))
    _direct_path(BALANCE_TO_WIND, ("151", "152", "158", "164", "174", "176"))
    _direct_path(REFERENCE_POINT, ("182", "184", "189", "195"))
    alpha_sign = node(WIND_TO_BODY, "68")
    if alpha_sign is None or "60" in alpha_sign.depends_on or "60" in alpha_sign.ancestors:
        raise RuntimeError("The YTZ component cannot decide the outer sign of the theoretical angle of attack.")


def _check_generators() -> None:
    from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET as generated_offset
    from domains.wind_tunnel.dataset_generate import _output_format
    from domains.wind_tunnel.ref_point_generate import STANDARD_SPEC as ref_spec
    from domains.wind_tunnel.ref_point_generate import generate as generate_ref
    from domains.wind_tunnel.tp_fd_generate import STANDARD_SPEC as wind_spec
    from domains.wind_tunnel.tp_fd_generate import generate as generate_wind
    from domains.wind_tunnel.tp_tz_generate import STANDARD_SPEC as body_spec
    from domains.wind_tunnel.tp_tz_generate import generate as generate_body
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC as attitude_spec
    from domains.wind_tunnel.tz_fd_generate import generate as generate_attitude

    if generated_offset != PROCESS_OFFSET:
        raise RuntimeError(f"Dependency-graph offsets {PROCESS_OFFSET} differ from the generator offsets {generated_offset}.")
    produced = {
        WIND_TO_BODY: generate_attitude(attitude_spec)["items"],
        BALANCE_TO_BODY: generate_body(body_spec)["items"],
        BALANCE_TO_WIND: generate_wind(wind_spec)["items"],
        REFERENCE_POINT: generate_ref(ref_spec)["items"],
    }
    for process, items in produced.items():
        removed = len(REMOVED_SLOTS.get(process, ()))
        if len(items) != PROCESS_SIZE[process] - removed:
            raise RuntimeError(f"{process} generated {len(items)} blanks; the dependency graph has {PROCESS_SIZE[process]}.")
        for item in items:
            public_id = str(int(item["manuscript_id"]) + PROCESS_OFFSET[process])
            found = node(process, public_id)
            if found is None:
                raise RuntimeError(f"{process} generated ID{public_id}, which is not in the dependency graph.")
            if found.label != item["label"]:
                raise RuntimeError(
                    f"{process} ID{public_id} generator label is {item['label']}, "
                    f"but the dependency graph says {found.label}. Stem: {item['prompt']}"
                )
            output_format = _output_format(process, item)
            format_code = "choice" if output_format is None else _format_name(output_format)
            allowed = FORMAT_OF_ANSWER[found.answer_type]
            if format_code not in allowed:
                raise RuntimeError(
                    f"{process} ID{public_id} answer type {found.answer_type} "
                    f"does not match output format {format_code}. Stem: {item['prompt']}"
                )


def _format_name(output_format: str) -> str:
    from domains.wind_tunnel.dataset_generate import FORMAT_CODE

    return FORMAT_CODE[output_format]


def document() -> dict:
    processes = {}
    edge_count = 0
    future_count = 0
    max_depth = 0
    for process, nodes in GRAPH.items():
        blanks = {}
        for blank_id, item in nodes.items():
            edge_count += len(item.depends_on)
            future_count += len(item.late_parents())
            max_depth = max(max_depth, item.dependency_depth)
            blanks[blank_id] = {
                "id": blank_id,
                "process": process,
                "label": item.label,
                "role": item.role,
                "target_definition": item.target_definition,
                "answer_type": item.answer_type,
                "error_type": item.error_type,
                "depends_on": list(item.depends_on),
                "dependency_depth": item.dependency_depth,
                "min_dependency_depth": item.min_dependency_depth,
                "future_dependencies": list(item.late_parents()),
                "dependency_edge_types": [_hop_type(parent) for parent in item.depends_on],
                "definition_relations": definition_relations(item.blank_id),
                **rule_mention_fields(item.blank_id),
                "direct_dependents": list(item.direct_dependents),
                "ancestors": list(item.ancestors),
                "descendants": list(item.descendants),
            }
        processes[process] = {"blank_count": len(nodes), "blanks": blanks}
    payload = {
        "domain": "cross.wind_flight",
        "blank_count": sum(len(nodes) for nodes in GRAPH.values()),
        "edge_count": edge_count,
        "future_edge_count": future_count,
        "max_dependency_depth": max_depth,
        "edge_meaning": "u->v means the standard derivation of v uses the result of u. An edge is a logical dependency, not sentence order. future_dependencies keeps only edges whose parent blank appears in a later sentence; a prefix stops at the current sentence, so those parent answers cannot enter the prefix.",
        "depth_meaning": "dependency_depth is the longest number of direct edges from a declaration root to this blank. min_dependency_depth is the shortest. A single-point injection changes one upstream blank and keeps the rest of the prefix correct, so it follows direct edges; natural prefix propagation follows descendants.",
        "not_an_edge": "The YTZ component at ID60 does not decide the outer sign of the theoretical angle of attack at ID65. The outer sign reuses ID22. YTZ reaches the final formula by ID60->ID80 and by ID60->ID63->ID70->ID80.",
        "direct_paths": [list(path) for path in DIRECT_PATHS],
        "definition_edge_meaning": "D->Q is an edge from a definition into a question. direct_answer means the standard perturbation of this definition changes that question's correct answer; direct_method means it decides the algorithm or a factor position; direct_scope means the answer may stay the same, but the question must bind the definition to the right frame and must not attach it to another axis system. downstream is the descendant reached along answer-dependency edges. ground_truth_changes describes only the current standard perturbation and does not treat a scope question as one whose answer should change. error_candidate is a candidate error class during traceback, not a model error that has already been decided.",
        "semantic_codes": {
            "direct_answer": 1,
            "direct_method": 2,
            "direct_scope": 3,
            "downstream": 4,
        },
        "effect_codes": {
            "0": "this perturbation does not change the correct answer",
            "1": "this perturbation changes the correct answer directly",
            "3": "this perturbation changes the correct answer through an upstream answer",
            "4": "this perturbation changes the correct answer through a matrix or a formula",
        },
        "definitions": effects()["by_definition"],
        "rule_mention_meaning": "recall is a blank that asks the process to recall or confirm a rule; the no-rule-mention build drops it by rule_definition. method is a method hint in the stem, and the blank itself is kept. The declaration is not deleted because of these marks.",
        "rule_mentions": _rule_mention_catalog(),
        "processes": processes,
    }
    from domains.wind_tunnel.contract_graph import contract_layer

    payload["contract_layer"] = contract_layer()
    return payload


def _print_blank(process: str, blank_id: str) -> None:
    found = node(process, blank_id)
    if found is None:
        for candidate, nodes in GRAPH.items():
            if blank_id in nodes:
                found = nodes[blank_id]
                break
    if found is None:
        raise SystemExit(f"There is no ID{blank_id}.")
    print(
        f"ID{found.blank_id} [{found.label}] {found.role} "
        f"depth {found.dependency_depth} "
        f"parents {','.join(found.depends_on) or '-'} "
        f"children {','.join(found.direct_dependents) or '-'}"
    )
    for item in definition_relations(found.blank_id):
        route = " → ".join(f"ID{blank_id}" for blank_id in item["path"])
        print(
            f"  {item['definition_id']} {item['relation_type']} "
            f"gold_changes={item['ground_truth_changes']} {route}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the wind-tunnel error-dependency graph and check it against the generators.")
    parser.add_argument("--blank", action="append", default=[], help="Print the direct parents and children of these blanks.")
    parser.add_argument("--trace", help="Walk every definition path that reaches this wrong question.")
    parser.add_argument("--records", help="Score jsonl. Use it with --trace.")
    parser.add_argument("--sample", help="sample_id in the jsonl.")
    parser.add_argument("--condition", help="gold-trunc or self-trunc. Required when one file has more than one condition.")
    parser.add_argument("--run", type=int, default=0, help="Which run of the same sample, default 0.")
    parser.add_argument("--no-write", action="store_true", help="Check only; do not write JSON.")
    args = parser.parse_args()
    if args.trace:
        observations = {}
        factor = None
        if args.records:
            if not args.sample:
                raise SystemExit("--records also requires --sample.")
            observations = observations_from_jsonl(
                Path(args.records), args.sample, args.condition, args.run,
            )
            factor = _FACTOR_OF_SAMPLE.get(args.sample.split("-", 1)[0])
        print(format_trace(trace_error(args.trace, observations, factor)))
        return
    _check_known_paths()
    _check_generators()
    payload = document()
    if payload["future_edge_count"] != 0:
        raise RuntimeError(f"The problem text still has {payload['future_edge_count']} future dependencies.")
    if not args.no_write:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"{payload['blank_count']} blanks, {payload['edge_count']} edges, "
        f"{payload['future_edge_count']} future edges, max depth {payload['max_dependency_depth']}"
    )
    layer = payload["contract_layer"]
    print(
        f"contract atoms {layer['atom_count']} in graph, "
        f"latent {layer['latent_count']} excluded, "
        f"bindings {layer['binding_count']} not nodes, "
        f"direct effects {layer['direct_edge_count']}, "
        f"indirect paths {layer['indirect_path_count']}"
    )
    for spec in payload["definitions"].values():
        questions = spec["questions"]
        direct_changed = sum(
            1 for item in questions.values()
            if item["direct"] and item["ground_truth_changes"]
        )
        scope = sum(1 for item in questions.values() if item["relation_type"] == "direct_scope")
        method = sum(1 for item in questions.values() if item["relation_type"] == "direct_method")
        downstream_changed = sum(
            1 for item in questions.values()
            if item["relation_type"] == "downstream" and item["ground_truth_changes"]
        )
        print(
            f"{spec['id']} {spec['name']}: "
            f"direct_changed {direct_changed}, scope {scope}, method {method}, "
            f"downstream_changed {downstream_changed}"
        )
    print(payload["not_an_edge"])
    for path in DIRECT_PATHS:
        print(" → ".join(f"ID{blank_id}" for blank_id in path))
    for blank_id in args.blank:
        _print_blank("", blank_id)
    if not args.no_write:
        print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
