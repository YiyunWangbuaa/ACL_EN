# -*- coding: utf-8 -*-
"""Definition-perturbation model for body, wind, and balance frames.

Symbols:

    alpha       α     angle of attack
    beta        β     sideslip angle
    gamma       γ     roll angle
    alpha_m     α_M   mechanism angle of attack
    beta_m      β_M   mechanism sideslip angle
    gamma_m     γ_M   mechanism roll angle
    alpha_e     α_e   balance elastic angle of attack
    beta_e      β_e   balance elastic sideslip angle
    gamma_e     γ_e   balance elastic roll angle
    delta_alpha Δα    elastic angle-of-attack increment
    delta_beta  Δβ    elastic sideslip increment
    delta_gamma Δγ    elastic roll increment
    gamma_tp    γ_tp  balance installation angle
    gamma_model γ_model model installation angle
    alpha_v     α_V   flow-eccentricity angle of attack
    beta_v      β_V   flow-eccentricity sideslip angle

Aircraft, tunnel, and balance each have a fixed triad and six directions.
The owner prefix is body, tunnel, or balance. These directions do not change
when a definition is perturbed.

Body directions are body-relative. Placement is separate: upright keeps all
three axes parallel to the tunnel; inverted keeps the axial axis parallel and
reverses the normal and lateral axes. One fixed right-handed basis is used for
cross products and is not a perturbable definition:

    forward x right = down

Airspeed stays along the tunnel axial positive direction. The balance triad
coincides with the tunnel triad and has no extra placement. The balance
installation angle is the angle between the balance normal and the vertical
plane through the balance axial axis. The model installation angle uses the
same construction about the body axial axis. Each positive sense is declared
as the same as, or opposite to, the right-hand rule about the absolute axial
positive direction, which points forward, not about the coordinate axis on it.
A positive elastic
deformation moves the model the same way in the tunnel; placement then maps
that motion into body directions. The increment is positive when that motion
increases the corresponding theoretical angle.

Coordinate frames: body XTZ/YTZ/ZTZ, wind XFD/YFD/ZFD, balance XTP/YTP/ZYP.
Every frame uses a passive rotation. That convention is fixed and is not a
definition. Lift CN, side force CZ, and drag CA lie on the normal, lateral,
and axial axes. Pitch MZ, roll MA, and yaw MN lie on the lateral, axial, and
normal axes. These load associations are fixed and are not a definition.
Attitude changes of the aircraft are intrinsic: each angle turns
about an aircraft axis that already includes the previous rotations, so the
axes move with the aircraft. This wind tunnel has no fixed-axis rotation.
Wind and balance axes may be redefined. The wind-to-body pre-matrix depends on
placement and on where the two frames point.

Declare an example with::

    model = BodyFrameModel()
    model.declare({
        "alpha": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma": {"right_wing_toward": "body_down", "sign": "positive"},
        "alpha_m": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta_m": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma_m": {"right_wing_toward": "body_down", "sign": "positive"},
        "placement": "upright",
        "body_frame": {
            "chirality": "right",
            "XTZ": {"along": "body_axial", "points": "body_forward"},
            "YTZ": {"along": "body_normal", "points": "body_up"},
        },
        "wind_frame": {
            "chirality": "right",
            "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
            "YFD": {"along": "tunnel_normal", "points": "tunnel_up"},
        },
        "alpha_e": {"axial_bends": "balance_up", "sign": "positive"},
        "beta_e": {"axial_bends": "balance_left", "sign": "positive"},
        "gamma_e": {"lateral_moves": "balance_down", "sign": "positive"},
        "alpha_v": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta_v": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma_tp": {"sense": "same"},
        "gamma_model": {"sense": "same"},
        "balance_frame": {
            "chirality": "right",
            "XTP": {"along": "balance_axial", "points": "balance_forward"},
            "YTP": {"along": "balance_normal", "points": "balance_up"},
        },
    })
    print(model.describe())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sympy as sp


class DefinitionError(ValueError):
    """A definition contradicts the absolute datum or the declared chirality."""


_ROLE_AXES = ("axial", "normal", "lateral")
_ROLE_DIRECTIONS = ("forward", "rearward", "up", "down", "left", "right")
_AXIS_OF_ROLE = {
    "forward": "axial",
    "rearward": "axial",
    "up": "normal",
    "down": "normal",
    "left": "lateral",
    "right": "lateral",
}
_OPPOSITE_ROLE = {
    "forward": "rearward",
    "rearward": "forward",
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
# Reference pose: forward x right = down. Body, tunnel, and balance share these vectors.
_ROLE_VECTOR = {
    "forward": (1, 0, 0),
    "rearward": (-1, 0, 0),
    "right": (0, 1, 0),
    "left": (0, -1, 0),
    "down": (0, 0, 1),
    "up": (0, 0, -1),
}


@dataclass(frozen=True)
class DirectionVocabulary:
    """Three axes and six directions of one physical entity."""

    owner: str
    axes: tuple[str, str, str]
    directions: tuple[str, ...]
    axis_of_direction: dict[str, str]
    opposite: dict[str, str]
    name_by_vector: dict[tuple[int, int, int], str]
    coordinate_axes: tuple[str, str, str]


def _make_vocabulary(owner: str, coordinate_axes: tuple[str, str, str]) -> DirectionVocabulary:
    axes = tuple(f"{owner}_{role}" for role in _ROLE_AXES)
    directions = tuple(f"{owner}_{role}" for role in _ROLE_DIRECTIONS)
    return DirectionVocabulary(
        owner=owner,
        axes=axes,  # type: ignore[arg-type]
        directions=directions,
        axis_of_direction={
            f"{owner}_{role}": f"{owner}_{_AXIS_OF_ROLE[role]}" for role in _ROLE_DIRECTIONS
        },
        opposite={f"{owner}_{role}": f"{owner}_{_OPPOSITE_ROLE[role]}" for role in _ROLE_DIRECTIONS},
        name_by_vector={_ROLE_VECTOR[role]: f"{owner}_{role}" for role in _ROLE_DIRECTIONS},
        coordinate_axes=coordinate_axes,
    )


BODY_VOCAB = _make_vocabulary("body", ("XTZ", "YTZ", "ZTZ"))
WIND_VOCAB = _make_vocabulary("tunnel", ("XFD", "YFD", "ZFD"))
BALANCE_VOCAB = _make_vocabulary("balance", ("XTP", "YTP", "ZYP"))

AXIAL, NORMAL, LATERAL = BODY_VOCAB.axes
ABSOLUTE_AXES = BODY_VOCAB.axes
FRONT, REAR, UP, DOWN, LEFT, RIGHT = BODY_VOCAB.directions
DIRECTIONS = BODY_VOCAB.directions
AXIS_OF_DIRECTION = BODY_VOCAB.axis_of_direction
OPPOSITE = BODY_VOCAB.opposite
WIND_AXES = WIND_VOCAB.axes
WIND_DIRECTIONS = WIND_VOCAB.directions
TUNNEL_AXIAL, TUNNEL_NORMAL, TUNNEL_LATERAL = WIND_VOCAB.axes
TUNNEL_FRONT, TUNNEL_REAR, TUNNEL_UP, TUNNEL_DOWN, TUNNEL_LEFT, TUNNEL_RIGHT = WIND_VOCAB.directions
PLACE_UPRIGHT = "upright"
PLACE_INVERTED = "inverted"
_REVERSED_ROLE = {
    "forward": "forward",
    "rearward": "rearward",
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
BALANCE_AXES = BALANCE_VOCAB.axes
BALANCE_DIRECTIONS = BALANCE_VOCAB.directions
BALANCE_AXIAL, BALANCE_NORMAL, BALANCE_LATERAL = BALANCE_VOCAB.axes
BALANCE_FRONT, BALANCE_REAR, BALANCE_UP, BALANCE_DOWN, BALANCE_LEFT, BALANCE_RIGHT = BALANCE_VOCAB.directions


@dataclass(frozen=True)
class LoadCoefficient:
    """One force or moment coefficient tied to an absolute axis. This is not a definition."""

    symbol: str
    kind: str
    role: str
    name: str

    def axis(self, owner: str) -> str:
        return f"{owner}_{self.role}"


LOAD_COEFFICIENTS = (
    LoadCoefficient("CN", "force", "normal", "lift"),
    LoadCoefficient("CZ", "force", "lateral", "side_force"),
    LoadCoefficient("CA", "force", "axial", "drag"),
    LoadCoefficient("MZ", "moment", "lateral", "pitch"),
    LoadCoefficient("MA", "moment", "axial", "roll"),
    LoadCoefficient("MN", "moment", "normal", "yaw"),
)
LOAD_BY_SYMBOL = {item.symbol: item for item in LOAD_COEFFICIENTS}

_VECTOR = {
    name: _ROLE_VECTOR[name.removeprefix(f"{owner}_")]
    for owner, vocab in (("body", BODY_VOCAB), ("tunnel", WIND_VOCAB), ("balance", BALANCE_VOCAB))
    for name in vocab.directions
}

ALPHA = "alpha"
BETA = "beta"
GAMMA = "gamma"
ANGLES = (ALPHA, BETA, GAMMA)
# Spherical wrist: yaw, pitch, roll. Yaw is sideslip, pitch is angle of attack.
WRIST_ORDER = (BETA, ALPHA, GAMMA)
ALPHA_M = "alpha_m"
BETA_M = "beta_m"
GAMMA_M = "gamma_m"
MECHANISM_ANGLES = (ALPHA_M, BETA_M, GAMMA_M)
ALPHA_E = "alpha_e"
BETA_E = "beta_e"
GAMMA_E = "gamma_e"
ELASTIC_ANGLES = (ALPHA_E, BETA_E, GAMMA_E)
DELTA_ALPHA = "delta_alpha"
DELTA_BETA = "delta_beta"
DELTA_GAMMA = "delta_gamma"
GAMMA_TP = "gamma_tp"
GAMMA_MODEL = "gamma_model"
ALPHA_V = "alpha_v"
BETA_V = "beta_v"
FLOW_ANGLES = (ALPHA_V, BETA_V)
# Allowed airspeed sides, then the body axis the positive angle turns about.
FLOW_SPEC = {
    ALPHA_V: ((UP, DOWN), LATERAL),
    BETA_V: ((LEFT, RIGHT), NORMAL),
}
SENSE_SAME = "same"
SENSE_OPPOSITE = "opposite"
SYMBOL = {
    ALPHA: "α",
    BETA: "β",
    GAMMA: "γ",
    ALPHA_M: "α_M",
    BETA_M: "β_M",
    GAMMA_M: "γ_M",
    ALPHA_E: "α_e",
    BETA_E: "β_e",
    GAMMA_E: "γ_e",
    DELTA_ALPHA: "Δα",
    DELTA_BETA: "Δβ",
    DELTA_GAMMA: "Δγ",
    GAMMA_TP: "γ_tp",
    GAMMA_MODEL: "γ_model",
    ALPHA_V: "α_V",
    BETA_V: "β_V",
}
# Mechanism alpha and beta deflect airspeed about these absolute axes.
# Mechanism roll uses the same right-wing declaration as gamma.
MECHANISM_OFFSET_AXIS = {
    ALPHA_M: NORMAL,
    BETA_M: LATERAL,
}
# Which absolute axis changes which theoretical angle. Mechanism angles share those axes.
ANGLE_ABOUT_ABSOLUTE_AXIS = {
    LATERAL: ALPHA,
    NORMAL: BETA,
    AXIAL: GAMMA,
}
ABSOLUTE_AXIS_OF_MECHANISM = {
    ALPHA_M: LATERAL,
    BETA_M: NORMAL,
    GAMMA_M: AXIAL,
}
MECHANISM_ABOUT_ABSOLUTE_AXIS = {axis: name for name, axis in ABSOLUTE_AXIS_OF_MECHANISM.items()}
ELASTIC_SPEC = {
    ALPHA_E: ("axial_bends", (BALANCE_UP, BALANCE_DOWN), BALANCE_LATERAL, BALANCE_FRONT),
    BETA_E: ("axial_bends", (BALANCE_LEFT, BALANCE_RIGHT), BALANCE_NORMAL, BALANCE_FRONT),
    GAMMA_E: ("lateral_moves", (BALANCE_UP, BALANCE_DOWN), BALANCE_AXIAL, BALANCE_RIGHT),
}
# Elastic angle, the increment it causes, the scheduled theoretical angle, and the effect kind.
INCREMENT_SPEC = {
    ALPHA_E: (DELTA_ALPHA, ALPHA, "airspeed"),
    BETA_E: (DELTA_BETA, BETA, "airspeed"),
    GAMMA_E: (DELTA_GAMMA, GAMMA, "right_wing"),
}

BODY_AXES = ("XTZ", "YTZ", "ZTZ")
POSITIVE = "positive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
POSITIVE_ROTATION = "positive_rotation"
NEGATIVE_ROTATION = "negative_rotation"
UNDEFINED_ROTATION = "undefined"
RIGHT_HANDED = "right"
LEFT_HANDED = "left"
INTRINSIC = "intrinsic"
DECLARED = "declared"
INFERRED = "inferred"
ROTATION_ANGLE = sp.symbols("t")


def _label(name: str) -> str:
    symbol = SYMBOL.get(name)
    if symbol:
        return f"{symbol} ({name})"
    return name


def _cross(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _add(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul(vector: tuple[int, int, int], scale: int) -> tuple[int, int, int]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _body_direction_in_tunnel(direction: str, placement: str) -> tuple[int, int, int]:
    """Place a body-relative direction into the tunnel. Inverted swaps normal and lateral."""
    role = direction.removeprefix("body_")
    if placement == PLACE_INVERTED:
        role = _REVERSED_ROLE[role]
    return _ROLE_VECTOR[role]


def _body_along_balance(balance_direction: str, placement: str) -> str:
    """Body direction parallel to this balance direction under the current placement."""
    target = _VECTOR[balance_direction]
    for body_direction in DIRECTIONS:
        if _body_direction_in_tunnel(body_direction, placement) == target:
            return body_direction
    raise DefinitionError(
        f"Placement {placement} has no body direction parallel to {balance_direction}."
    )


def roll_about_nose_sense(right_wing_toward: str) -> str:
    """Whether a positive roll about the nose matches the right-hand rule.

    The nose is the model's axial positive direction. Inversion does not move it,
    so this sense does not change between upright and inverted installation.
    A matching rotation carries the right wing toward body down.
    """
    if not right_wing_toward:
        raise DefinitionError("Roll has no right-wing direction.")
    rh_wing = _direction_name(_cross(_VECTOR[FRONT], _VECTOR[RIGHT]), BODY_VOCAB)
    if right_wing_toward == rh_wing:
        return SENSE_SAME
    if right_wing_toward == OPPOSITE[rh_wing]:
        return SENSE_OPPOSITE
    raise DefinitionError("Roll is not about the model's axial positive direction.")


def _direction_name(vector: tuple[int, int, int], vocab: DirectionVocabulary) -> str:
    try:
        return vocab.name_by_vector[vector]
    except KeyError as exc:
        raise DefinitionError(
            f"Inferred direction {vector} is not one of the six {vocab.owner} directions."
        ) from exc


@dataclass
class AxisRecord:
    """Spatial relations stored on one coordinate axis."""

    name: str
    absolute_axis: str = ""
    points_to: str = ""
    source: str = ""
    associated_angle: str = ""
    rotation_when_positive: str = UNDEFINED_ROTATION
    associated_mechanism_angle: str = ""
    mechanism_rotation_when_positive: str = UNDEFINED_ROTATION
    installation_angle: str = ""
    installation_rotation_when_positive: str = ""
    rotation_matrix: sp.Matrix | None = None

    @property
    def vector(self) -> tuple[int, int, int]:
        return _VECTOR[self.points_to]


@dataclass
class AngleRecord:
    """Spatial relations stored on one angle.

    The angle is bound to an absolute direction axis first (``absolute_axis``);
    ``associated_axis`` is the coordinate axis that a declaration puts on that
    direction, inferred with :meth:`BodyFrameModel._bind_to_absolute`.
    """

    name: str
    symbol: str = ""
    definition: str = ""
    positive_airspeed_toward: str = ""
    airspeed_signs: dict[str, str] = field(default_factory=dict)
    right_wing_toward: str = ""
    associated_axis: str = ""
    rotation_when_positive: str = UNDEFINED_ROTATION
    motion: str = ""
    positive_toward: str = ""
    sense: str = ""
    component_signs: dict[str, str] = field(default_factory=dict)
    absolute_axis: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            self.symbol = SYMBOL.get(self.name, "")


@dataclass
class IncrementRecord:
    """Increment applied to a theoretical angle by a positive elastic deformation."""

    name: str
    elastic_angle: str
    theoretical_angle: str
    kind: str
    symbol: str = ""
    definition: str = ""
    deformation: str = ""
    effect_toward: str = ""
    theory_toward: str = ""
    increment_sign: str = ""

    def __post_init__(self) -> None:
        if not self.symbol:
            self.symbol = SYMBOL[self.name]


@dataclass
class Aircraft:
    """Aircraft. Holds the body triad and six directions; declarations do not move them.

    Attitude rotations are intrinsic. Each angle turns a moving aircraft axis.
    """

    axes: tuple[str, ...] = ABSOLUTE_AXES
    directions: tuple[str, ...] = DIRECTIONS
    rotation_type: str = INTRINSIC


@dataclass
class WindTunnel:
    """Wind tunnel. Holds the tunnel triad and six directions."""

    axes: tuple[str, ...] = WIND_AXES
    directions: tuple[str, ...] = WIND_DIRECTIONS


@dataclass
class Balance:
    """Balance. Absolute directions coincide with the tunnel."""

    axes: tuple[str, ...] = BALANCE_AXES
    directions: tuple[str, ...] = BALANCE_DIRECTIONS


@dataclass
class Airspeed:
    """Airspeed. At zero attitude it lies along the tunnel axial positive direction."""

    along_tunnel_axis: str = TUNNEL_FRONT
    positive_alpha_toward: str = ""
    positive_beta_toward: str = ""
    positive_mechanism_alpha_toward: str = ""
    positive_mechanism_beta_toward: str = ""
    positive_alpha_v_toward: str = ""
    positive_beta_v_toward: str = ""


class BodyFrameModel:
    """Read structured definitions and refresh the stored spatial relations."""

    def __init__(self) -> None:
        self.aircraft = Aircraft()
        self.wind_tunnel = WindTunnel()
        self.balance = Balance()
        self.airspeed = Airspeed()
        self.angles = {name: AngleRecord(name) for name in ANGLES}
        self.mechanism_angles = {name: AngleRecord(name) for name in MECHANISM_ANGLES}
        self.flow_angles = {name: AngleRecord(name) for name in FLOW_ANGLES}
        self.elastic_angles = {name: AngleRecord(name) for name in ELASTIC_ANGLES}
        self.increments = {
            delta_name: IncrementRecord(delta_name, elastic_name, theory_name, kind)
            for elastic_name, (delta_name, theory_name, kind) in INCREMENT_SPEC.items()
        }
        self.balance_body: dict[str, str] = {}
        self.installation = AngleRecord(GAMMA_TP)
        self.model_installation = AngleRecord(GAMMA_MODEL)
        self.body_frame = {name: AxisRecord(name) for name in BODY_AXES}
        self.wind_frame = {name: AxisRecord(name) for name in WIND_VOCAB.coordinate_axes}
        self.balance_frame = {name: AxisRecord(name) for name in BALANCE_VOCAB.coordinate_axes}
        self.chirality = ""
        self.wind_chirality = ""
        self.balance_chirality = ""
        self._balance_frame_spec: dict | None = None
        self.placement = ""
        self.wind_to_body: sp.Matrix | None = None
        self.euler_order = ANGLES
        self._frame_spec: dict | None = None
        self._wind_frame_spec: dict | None = None

    def declare(self, spec: dict) -> BodyFrameModel:
        """Store or perturb definitions, then recompute every stored relation."""
        if "rotation_type" in spec:
            raise DefinitionError(
                "Aircraft rotation is intrinsic, about moving axes. It is fixed by the aircraft and is not a definition."
            )
        if "load_coefficients" in spec or any(symbol in spec for symbol in LOAD_BY_SYMBOL):
            raise DefinitionError(
                "Force and moment coefficients are fixed to the absolute axes and are not a definition."
            )
        if ALPHA in spec:
            self._set_angle(ALPHA, spec[ALPHA], NORMAL)
        if BETA in spec:
            self._set_angle(BETA, spec[BETA], LATERAL)
        if GAMMA in spec:
            self._set_roll(GAMMA, spec[GAMMA])
        if GAMMA_M in spec:
            self._set_roll(GAMMA_M, spec[GAMMA_M])
        for name, allowed_axis in MECHANISM_OFFSET_AXIS.items():
            if name in spec:
                self._set_angle(name, spec[name], allowed_axis)
        for name in FLOW_ANGLES:
            if name in spec:
                self._set_flow_angle(name, spec[name])
        if "body_frame" in spec:
            self._frame_spec = spec["body_frame"]
            order = self._frame_spec.get("euler_order")
            if order is not None:
                self._set_euler_order(order)
        if "wind_frame" in spec:
            self._wind_frame_spec = spec["wind_frame"]
        if "placement" in spec:
            self._set_placement(spec["placement"])
        for name in ELASTIC_ANGLES:
            if name in spec:
                self._set_elastic(name, spec[name])
        if GAMMA_TP in spec:
            self._set_installation(self.installation, spec[GAMMA_TP])
        if GAMMA_MODEL in spec:
            self._set_installation(self.model_installation, spec[GAMMA_MODEL])
        if "balance_frame" in spec:
            self._balance_frame_spec = spec["balance_frame"]
        self._refresh()
        return self

    def describe(self) -> str:
        lines = [
            "Right-handed embedding: forward x right = down. Body directions are body-relative.",
            f"Aircraft axes: {', '.join(self.aircraft.axes)}.",
            "Aircraft rotation: intrinsic, about the aircraft's moving axes. This is the only type.",
            f"Tunnel axes: {', '.join(self.wind_tunnel.axes)}.",
            f"Balance axes: {', '.join(self.balance.axes)}. Directions coincide with the tunnel.",
            f"Placement: {self.placement or 'undeclared'}. {_placement_text(self.placement)}",
            (
                "Euler order is the spherical wrist, not the angle-rod mechanism: "
                "beta, then alpha, then gamma. Yaw, then pitch, then roll."
                if tuple(self.euler_order) == WRIST_ORDER
                else
                "Euler order is fixed by the angle mechanism, not a definition: "
                "alpha, then beta, then gamma. Alpha rotates first."
            ),
            "Rotation: passive for every frame. This convention is fixed.",
            "Each axis is first assigned an absolute-direction axis by the declaration. The angle is then bound to that absolute-direction axis, and the concrete coordinate axis is inferred from how that frame's axes correspond to the absolute directions.",
            "Angle records keep both absolute_axis and the inferred associated_axis.",
            "Load coefficients are fixed to absolute axes. The same roles apply to body and balance.",
            *[
                f"  {item.symbol} {item.name} {item.kind}: {item.role}"
                for item in LOAD_COEFFICIENTS
            ],
            f"Body frame chirality: {self.chirality or 'undeclared'}.",
            f"Wind frame chirality: {self.wind_chirality or 'undeclared'}.",
            f"Balance frame chirality: {self.balance_chirality or 'undeclared'}.",
            "",
            "Airspeed",
            f"  along: {self.airspeed.along_tunnel_axis}",
            f"  positive {SYMBOL[ALPHA]} deflects airspeed toward: {self.airspeed.positive_alpha_toward or 'undeclared'}",
            f"  positive {SYMBOL[BETA]} deflects airspeed toward: {self.airspeed.positive_beta_toward or 'undeclared'}",
            f"  positive {SYMBOL[ALPHA_M]} deflects airspeed toward: {self.airspeed.positive_mechanism_alpha_toward or 'undeclared'}",
            f"  positive {SYMBOL[BETA_M]} deflects airspeed toward: {self.airspeed.positive_mechanism_beta_toward or 'undeclared'}",
            f"  positive {SYMBOL[ALPHA_V]} deflects airspeed toward: {self.airspeed.positive_alpha_v_toward or 'undeclared'}",
            f"  positive {SYMBOL[BETA_V]} deflects airspeed toward: {self.airspeed.positive_beta_v_toward or 'undeclared'}",
            "",
            "Theoretical angles",
        ]
        for name in ANGLES:
            _append_angle(lines, self.angles[name])
        lines.append("")
        lines.append("Mechanism angles")
        for name in MECHANISM_ANGLES:
            _append_angle(lines, self.mechanism_angles[name])
        lines.append("")
        lines.append("Flow eccentricity angles, stated for an upright level model")
        for name in FLOW_ANGLES:
            _append_angle(lines, self.flow_angles[name])
        lines.append("")
        lines.append("Elastic angles")
        for name in ELASTIC_ANGLES:
            _append_elastic(lines, self.elastic_angles[name])
        lines.append("")
        lines.append("Balance-body alignment")
        _append_balance_body(lines, self.placement, self.balance_body)
        lines.append("")
        lines.append("Elastic increments")
        for _elastic_name, (delta_name, _theory_name, _kind) in INCREMENT_SPEC.items():
            _append_increment(lines, self.increments[delta_name])
        lines.append("")
        lines.append("Balance installation angle")
        _append_elastic(lines, self.installation)
        lines.append("")
        lines.append("Model installation angle")
        _append_elastic(lines, self.model_installation)
        lines.append("")
        _append_frame(lines, "Body frame", BODY_AXES, self.body_frame)
        lines.append("")
        _append_frame(lines, "Wind frame", WIND_VOCAB.coordinate_axes, self.wind_frame)
        if self.wind_to_body is not None:
            lines.append(
                "  Pre-matrix, wind coordinates to body coordinates at the current placement"
                " (rows XTZ/YTZ/ZTZ, columns XFD/YFD/ZFD):"
            )
            lines.append(_format_matrix(self.wind_to_body))
        lines.append("")
        _append_frame(lines, "Balance frame", BALANCE_VOCAB.coordinate_axes, self.balance_frame)
        return "\n".join(lines)

    def _set_angle(self, name: str, spec: dict, allowed_axis: str) -> None:
        located = spec.get("airspeed_toward")
        sign = spec.get("sign", POSITIVE)
        if located not in DIRECTIONS:
            raise DefinitionError(f"{_label(name)} airspeed_toward must be one of the six body directions.")
        if AXIS_OF_DIRECTION[located] != allowed_axis:
            raise DefinitionError(
                f"{_label(name)} airspeed must lie on {allowed_axis}, got {located}."
            )
        if sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{_label(name)} sign must be positive or negative.")
        toward = located if sign == POSITIVE else OPPOSITE[located]
        record = self._angle_record(name)
        record.positive_airspeed_toward = toward
        record.definition = f"airspeed toward {located} is {sign}"

    def _set_flow_angle(self, name: str, spec: dict) -> None:
        located = spec.get("airspeed_toward")
        sign = spec.get("sign", POSITIVE)
        allowed, _rotation_axis = FLOW_SPEC[name]
        if located not in allowed:
            raise DefinitionError(f"{_label(name)} airspeed_toward must be {' or '.join(allowed)}.")
        if sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{_label(name)} sign must be positive or negative.")
        toward = located if sign == POSITIVE else OPPOSITE[located]
        record = self.flow_angles[name]
        record.positive_airspeed_toward = toward
        record.definition = (
            "When the model is upright and level, tunnel flow eccentricity is positive "
            f"when airspeed toward {located} is {sign}."
        )

    def _set_roll(self, name: str, spec: dict) -> None:
        toward = spec.get("right_wing_toward")
        sign = spec.get("sign", POSITIVE)
        if toward not in (UP, DOWN):
            raise DefinitionError(f"{_label(name)} right_wing_toward must be body_up or body_down.")
        if sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{_label(name)} sign must be positive or negative.")
        record = self._angle_record(name)
        record.right_wing_toward = toward if sign == POSITIVE else OPPOSITE[toward]
        record.positive_airspeed_toward = ""
        record.airspeed_signs = {}
        record.definition = f"right wing toward {toward} is {sign}"

    def _set_euler_order(self, order) -> None:
        """Accept the rod order or the spherical-wrist order, and nothing else.

        The angle-of-attack rod moves first, the sideslip rod is mounted on it,
        and the body then rolls about that rod. That order is alpha, beta, gamma.
        The spherical wrist is a separate mechanism: yaw, pitch, then roll,
        which is beta, alpha, gamma. Any other permutation is rejected.
        """
        order = tuple(order)
        if order == ANGLES:
            self.euler_order = ANGLES
            return
        if order == WRIST_ORDER:
            self.euler_order = WRIST_ORDER
            return
        raise DefinitionError(
            "Only two intrinsic orders are accepted: "
            "alpha, beta, gamma for the angle-rod mechanism, and "
            "beta, alpha, gamma for the spherical wrist (yaw, pitch, roll)."
        )

    def _set_placement(self, placement: str) -> None:
        if placement not in (PLACE_UPRIGHT, PLACE_INVERTED):
            raise DefinitionError("placement must be upright or inverted.")
        self.placement = placement

    def _set_elastic(self, name: str, spec: dict) -> None:
        key, allowed, _absolute_axis, _base = ELASTIC_SPEC[name]
        located = spec.get(key)
        sign = spec.get("sign", POSITIVE)
        if located not in allowed:
            raise DefinitionError(f"{_label(name)} {key} must be {' or '.join(allowed)}.")
        if sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{_label(name)} sign must be positive or negative.")
        record = self.elastic_angles[name]
        record.motion = key
        record.positive_toward = located if sign == POSITIVE else BALANCE_VOCAB.opposite[located]
        record.definition = f"{key} {located} is {sign}"

    def _set_installation(self, record: AngleRecord, spec: dict) -> None:
        if "normal_lies" in spec or "sign" in spec:
            raise DefinitionError(
                f"{_label(record.name)} is the angle between the normal and the vertical plane "
                "through the axial axis. Declare sense as same or opposite to the right-hand rule "
                "about that axis."
            )
        sense = spec.get("sense")
        if sense not in (SENSE_SAME, SENSE_OPPOSITE):
            raise DefinitionError(f"{_label(record.name)} sense must be same or opposite.")
        record.sense = sense
        record.motion = "normal_from_vertical_plane"
        record.definition = (
            "angle between the normal and the vertical plane through the axial axis; "
            f"positive sense is {sense} as the right-hand rule about the axial axis"
        )

    def _refresh(self) -> None:
        alpha = self.angles[ALPHA].positive_airspeed_toward
        beta = self.angles[BETA].positive_airspeed_toward
        self.airspeed.positive_alpha_toward = alpha
        self.airspeed.positive_beta_toward = beta
        self.airspeed.positive_mechanism_alpha_toward = self.mechanism_angles[ALPHA_M].positive_airspeed_toward
        self.airspeed.positive_mechanism_beta_toward = self.mechanism_angles[BETA_M].positive_airspeed_toward
        self.airspeed.positive_alpha_v_toward = self.flow_angles[ALPHA_V].positive_airspeed_toward
        self.airspeed.positive_beta_v_toward = self.flow_angles[BETA_V].positive_airspeed_toward
        if self._frame_spec is not None:
            self._build_frame(self._frame_spec, BODY_VOCAB, self.body_frame, bind_angles=True)
            self._project_airspeed(ALPHA, alpha)
            self._project_airspeed(BETA, beta)
            self.angles[GAMMA].airspeed_signs = {}
            for name in MECHANISM_OFFSET_AXIS:
                self._project_airspeed(name, self.mechanism_angles[name].positive_airspeed_toward)
            self.mechanism_angles[GAMMA_M].airspeed_signs = {}
            for name in FLOW_ANGLES:
                self._project_airspeed(name, self.flow_angles[name].positive_airspeed_toward)
            self._bind_rotations()
            self._bind_flow_rotations()
        if self._wind_frame_spec is not None:
            self._build_frame(self._wind_frame_spec, WIND_VOCAB, self.wind_frame, bind_angles=False)
        if self._balance_frame_spec is not None:
            self._build_frame(self._balance_frame_spec, BALANCE_VOCAB, self.balance_frame, bind_angles=False)
        self._bind_elastic()
        self._bind_installation(
            self.installation, self.balance_frame, BALANCE_VOCAB, BALANCE_AXIAL, BALANCE_UP
        )
        self._bind_installation(self.model_installation, self.body_frame, BODY_VOCAB, AXIAL, UP)
        self._bind_balance_body()
        self.wind_to_body = self._wind_to_body_matrix()

    def _build_frame(
        self,
        spec: dict,
        vocab: DirectionVocabulary,
        store: dict[str, AxisRecord],
        bind_angles: bool,
    ) -> None:
        x_name, y_name, z_name = vocab.coordinate_axes
        chirality = spec.get("chirality")
        if chirality not in (RIGHT_HANDED, LEFT_HANDED):
            raise DefinitionError(f"{vocab.owner} frame chirality must be right or left.")
        declared = {}
        for name in vocab.coordinate_axes:
            if name in spec:
                declared[name] = self._parse_axis(name, spec[name], vocab)
        if x_name not in declared:
            raise DefinitionError(f"{x_name} must be declared.")
        if y_name not in declared and z_name not in declared:
            raise DefinitionError(f"Declare at least one of {y_name} and {z_name}; chirality fixes the other.")
        used = [item[0] for item in declared.values()]
        if len(used) != len(set(used)):
            raise DefinitionError(f"Two {vocab.owner} axes were declared along the same absolute axis.")

        x_vector = _VECTOR[declared[x_name][1]]
        if y_name in declared and z_name in declared:
            y_vector = _VECTOR[declared[y_name][1]]
            z_vector = _VECTOR[declared[z_name][1]]
            expected = _cross(x_vector, y_vector) if chirality == RIGHT_HANDED else _cross(y_vector, x_vector)
            if expected != z_vector:
                raise DefinitionError(
                    f"Directions of {x_name}, {y_name}, and {z_name} contradict a {chirality}-handed frame. "
                    f"Chirality requires {z_name} to point {_direction_name(expected, vocab)}."
                )
            sources = {name: DECLARED for name in declared}
        elif y_name in declared:
            y_vector = _VECTOR[declared[y_name][1]]
            z_vector = _cross(x_vector, y_vector) if chirality == RIGHT_HANDED else _cross(y_vector, x_vector)
            pointed = _direction_name(z_vector, vocab)
            declared[z_name] = (vocab.axis_of_direction[pointed], pointed)
            sources = {x_name: DECLARED, y_name: DECLARED, z_name: INFERRED}
        else:
            z_vector = _VECTOR[declared[z_name][1]]
            y_vector = _cross(z_vector, x_vector) if chirality == RIGHT_HANDED else _cross(x_vector, z_vector)
            pointed = _direction_name(y_vector, vocab)
            declared[y_name] = (vocab.axis_of_direction[pointed], pointed)
            sources = {x_name: DECLARED, y_name: INFERRED, z_name: DECLARED}

        if "rotation" in spec:
            raise DefinitionError(
                f"The {vocab.owner} frame rotation is fixed as passive and is not a definition."
            )
        if "rotation_type" in spec:
            raise DefinitionError(
                "Aircraft rotation is intrinsic, about moving axes. It is fixed by the aircraft and is not a definition."
            )
        if vocab.owner == "body":
            self.chirality = chirality
        elif vocab.owner == "tunnel":
            self.wind_chirality = chirality
        else:
            self.balance_chirality = chirality
        for name in vocab.coordinate_axes:
            absolute_axis, points_to = declared[name]
            axis = store[name]
            axis.absolute_axis = absolute_axis
            axis.points_to = points_to
            axis.source = sources[name]
            axis.associated_angle = ANGLE_ABOUT_ABSOLUTE_AXIS[absolute_axis] if bind_angles else ""
            axis.rotation_when_positive = UNDEFINED_ROTATION
            axis.associated_mechanism_angle = MECHANISM_ABOUT_ABSOLUTE_AXIS[absolute_axis] if bind_angles else ""
            axis.mechanism_rotation_when_positive = UNDEFINED_ROTATION
            axis.rotation_matrix = _axis_rotation_matrix(name, chirality)
            if bind_angles and axis.associated_angle:
                record = self.angles[axis.associated_angle]
                record.absolute_axis = absolute_axis
                record.associated_axis = name

    def _parse_axis(self, name: str, spec: dict, vocab: DirectionVocabulary) -> tuple[str, str]:
        absolute_axis = spec.get("along")
        points_to = spec.get("points")
        if absolute_axis not in vocab.axes:
            raise DefinitionError(f"{name} along must be one of {', '.join(vocab.axes)}.")
        if points_to not in vocab.directions:
            raise DefinitionError(f"{name} points must be one of the six {vocab.owner} directions.")
        if vocab.axis_of_direction[points_to] != absolute_axis:
            raise DefinitionError(
                f"{name} is along {absolute_axis} but points {points_to}; the two do not match."
            )
        return absolute_axis, points_to

    def _wind_to_body_matrix(self) -> sp.Matrix | None:
        """Map wind-frame coordinates into body-frame coordinates at the current placement.

        Column j is the positive direction of wind axis j in XTZ, YTZ, ZTZ components.
        Upright frames with matching axes give the identity. Inverted placement negates
        the normal and lateral axes.
        """
        if any(not self.body_frame[name].points_to for name in BODY_AXES):
            return None
        if any(not self.wind_frame[name].points_to for name in WIND_VOCAB.coordinate_axes):
            return None
        if not self.placement:
            raise DefinitionError("placement must be upright or inverted once both frames exist.")
        return sp.Matrix([
            [
                _dot(
                    _body_direction_in_tunnel(self.body_frame[body_name].points_to, self.placement),
                    self.wind_frame[wind_name].vector,
                )
                for wind_name in WIND_VOCAB.coordinate_axes
            ]
            for body_name in BODY_AXES
        ])

    def _bind_elastic(self) -> None:
        ready = all(self.balance_frame[name].points_to for name in BALANCE_VOCAB.coordinate_axes)
        for axis in self.balance_frame.values():
            axis.associated_angle = ""
            axis.rotation_when_positive = UNDEFINED_ROTATION
        for name, (motion, _allowed, absolute_axis, base) in ELASTIC_SPEC.items():
            record = self.elastic_angles[name]
            record.motion = record.motion or motion
            if not ready or not record.positive_toward:
                record.absolute_axis = absolute_axis
                record.associated_axis = ""
                record.component_signs = {}
                record.rotation_when_positive = UNDEFINED_ROTATION
                continue
            axis = self._bind_to_absolute(record, self.balance_frame, absolute_axis)
            if axis is None:
                record.component_signs = {}
                record.rotation_when_positive = UNDEFINED_ROTATION
                continue
            record.component_signs = self._component_signs(base, record.positive_toward)
            if name == GAMMA_E:
                record.rotation_when_positive = self._roll_sense(axis.vector, record.positive_toward)
            else:
                record.rotation_when_positive = self._airspeed_rotation_sense(axis.vector, record.positive_toward)
            axis.associated_angle = name
            axis.rotation_when_positive = record.rotation_when_positive

    def _bind_installation(
        self,
        record: AngleRecord,
        frame: dict[str, AxisRecord],
        vocab: DirectionVocabulary,
        axial_absolute: str,
        upward: str,
    ) -> None:
        """Installation angle: normal against the vertical plane through the axial axis.

        The declared sense is the right-hand rule about the absolute axial positive
        direction, which points forward. rotation_when_positive stores that absolute
        sense. A coordinate axis pointing rearward reverses the passive rotation
        about its own positive direction, and that reversed sense is stored on the axis.
        """
        for axis in frame.values():
            if axis.installation_angle == record.name:
                axis.installation_angle = ""
                axis.installation_rotation_when_positive = ""
        ready = all(frame[name].points_to for name in vocab.coordinate_axes)
        record.absolute_axis = axial_absolute
        if not record.sense or not ready:
            record.associated_axis = ""
            record.positive_toward = ""
            record.rotation_when_positive = UNDEFINED_ROTATION
            return
        axial = self._bind_to_absolute(record, frame, axial_absolute)
        if axial is None:
            record.positive_toward = ""
            record.rotation_when_positive = UNDEFINED_ROTATION
            return
        forward = vocab.name_by_vector[(1, 0, 0)]
        pointing = _dot(axial.vector, _VECTOR[forward])
        if pointing not in (1, -1):
            raise DefinitionError(f"{axial.name} does not lie on the absolute axial direction.")
        rh_side = _direction_name(_cross(_VECTOR[forward], _VECTOR[upward]), vocab)
        record.positive_toward = rh_side if record.sense == SENSE_SAME else vocab.opposite[rh_side]
        record.rotation_when_positive = (
            POSITIVE_ROTATION if record.sense == SENSE_SAME else NEGATIVE_ROTATION
        )
        coordinate_positive = record.sense == SENSE_SAME if pointing == 1 else record.sense != SENSE_SAME
        axial.installation_angle = record.name
        axial.installation_rotation_when_positive = (
            POSITIVE_ROTATION if coordinate_positive else NEGATIVE_ROTATION
        )

    def _bind_balance_body(self) -> None:
        """Map a positive balance deformation through placement onto the scheduled angle."""
        for record in self.increments.values():
            record.definition = ""
            record.deformation = ""
            record.effect_toward = ""
            record.theory_toward = ""
            record.increment_sign = ""
        if not self.placement:
            self.balance_body = {}
            return
        self.balance_body = {
            direction: _body_along_balance(direction, self.placement) for direction in BALANCE_DIRECTIONS
        }
        for elastic_name, (delta_name, theory_name, _kind) in INCREMENT_SPEC.items():
            self._bind_increment(elastic_name, self.increments[delta_name], theory_name)

    def _bind_increment(self, elastic_name: str, record: IncrementRecord, theory_name: str) -> None:
        elastic = self.elastic_angles[elastic_name]
        if not elastic.positive_toward:
            return
        record.deformation = f"{elastic.motion} {elastic.positive_toward}"
        if record.kind == "airspeed":
            # The nose bends toward the declared side, so relative airspeed deflects the other way.
            record.effect_toward = OPPOSITE[self.balance_body[elastic.positive_toward]]
            theory = self.angles[theory_name].positive_airspeed_toward
            effect_text = f"airspeed deflects toward {record.effect_toward}"
        else:
            # The balance right side moves as declared. Inversion puts the right wing on the
            # balance left and also swaps up with down, so the two reversals cancel.
            same_side = self.balance_body[BALANCE_RIGHT] == RIGHT
            motion = elastic.positive_toward if same_side else BALANCE_VOCAB.opposite[elastic.positive_toward]
            record.effect_toward = self.balance_body[motion]
            theory = self.angles[theory_name].right_wing_toward
            effect_text = f"right wing moves toward {record.effect_toward}"
        elastic_label = _label(elastic_name)
        if not theory:
            record.definition = (
                f"When {elastic_label} is positive ({record.deformation}), {effect_text}. "
                f"{_label(theory_name)} is undeclared, so the scheduled increment sign is unset."
            )
            return
        record.theory_toward = theory
        record.increment_sign = POSITIVE if record.effect_toward == theory else NEGATIVE
        relation = "matches" if record.increment_sign == POSITIVE else "opposes"
        record.definition = (
            f"When {elastic_label} is positive ({record.deformation}), {effect_text}; "
            f"this {relation} the positive sense of {_label(theory_name)}, "
            f"so {record.symbol} is {record.increment_sign}."
        )

    def _component_signs(self, base: str, offset: str) -> dict[str, str]:
        direction = _add(_VECTOR[base], _VECTOR[offset])
        signs = {}
        for name in BALANCE_VOCAB.coordinate_axes:
            component = _dot(direction, self.balance_frame[name].vector)
            if component == 0:
                signs[name] = UNKNOWN
            elif component > 0:
                signs[name] = POSITIVE
            else:
                signs[name] = NEGATIVE
        return signs

    def _angle_record(self, name: str) -> AngleRecord:
        if name in self.mechanism_angles:
            return self.mechanism_angles[name]
        if name in self.flow_angles:
            return self.flow_angles[name]
        return self.angles[name]

    def _project_airspeed(self, angle_name: str, offset: str) -> None:
        record = self._angle_record(angle_name)
        if not offset:
            record.airspeed_signs = {}
            return
        velocity = _add(_VECTOR[FRONT], _VECTOR[offset])
        signs = {}
        for name in BODY_AXES:
            component = _dot(velocity, self.body_frame[name].vector)
            if component == 0:
                signs[name] = UNKNOWN
            elif component > 0:
                signs[name] = POSITIVE
            else:
                signs[name] = NEGATIVE
        record.airspeed_signs = signs

    def _axis_on_absolute(self, frame: dict[str, AxisRecord], absolute_axis: str) -> AxisRecord | None:
        """Coordinate axis of a frame that lies on an absolute direction axis."""
        return next(
            (item for item in frame.values() if item.absolute_axis == absolute_axis),
            None,
        )

    def _bind_to_absolute(
        self,
        record: AngleRecord,
        frame: dict[str, AxisRecord],
        absolute_axis: str,
    ) -> AxisRecord | None:
        """Bind an angle to an absolute direction first, then infer the coordinate axis."""
        record.absolute_axis = absolute_axis
        axis = self._axis_on_absolute(frame, absolute_axis)
        if axis is None:
            record.associated_axis = ""
            return None
        record.associated_axis = axis.name
        return axis

    def _bind_rotations(self) -> None:
        offsets = {
            ALPHA: self.angles[ALPHA].positive_airspeed_toward,
            BETA: self.angles[BETA].positive_airspeed_toward,
        }
        axial = next((axis for axis in self.body_frame.values() if axis.absolute_axis == AXIAL), None)
        axial_vector = axial.vector if axial is not None else None
        roll = self.angles[GAMMA]
        roll.rotation_when_positive = self._roll_sense(axial_vector, roll.right_wing_toward)
        for axis in self.body_frame.values():
            angle_name = axis.associated_angle
            offset = offsets.get(angle_name, "")
            if angle_name == GAMMA:
                axis.rotation_when_positive = roll.rotation_when_positive
                continue
            axis.rotation_when_positive = self._airspeed_rotation_sense(axis.vector, offset)
        for name, absolute_axis in ABSOLUTE_AXIS_OF_MECHANISM.items():
            record = self.mechanism_angles[name]
            axis = self._bind_to_absolute(record, self.body_frame, absolute_axis)
            if axis is None:
                record.rotation_when_positive = UNDEFINED_ROTATION
                continue
            if name == GAMMA_M:
                record.rotation_when_positive = self._roll_sense(axis.vector, record.right_wing_toward)
            else:
                record.rotation_when_positive = self._airspeed_rotation_sense(
                    axis.vector,
                    record.positive_airspeed_toward,
                )
            axis.mechanism_rotation_when_positive = record.rotation_when_positive

    def _bind_flow_rotations(self) -> None:
        """Store the flow-angle sense on the angle only. Body axes keep the theoretical sense."""
        for name, (_allowed, absolute_axis) in FLOW_SPEC.items():
            record = self.flow_angles[name]
            axis = self._bind_to_absolute(record, self.body_frame, absolute_axis)
            if axis is None or not record.positive_airspeed_toward:
                record.associated_axis = ""
                record.rotation_when_positive = UNDEFINED_ROTATION
                continue
            record.rotation_when_positive = self._airspeed_rotation_sense(
                axis.vector,
                record.positive_airspeed_toward,
            )

    def _airspeed_rotation_sense(self, axis_vector: tuple[int, int, int], offset: str) -> str:
        if not offset:
            return UNDEFINED_ROTATION
        # Passive rotation: after a positive right-handed turn about the axis,
        # the fixed airspeed vector gains -(axis x forward).
        passive_shift = _mul(_cross(axis_vector, _VECTOR[FRONT]), -1)
        return self._sense_from_alignment(_dot(passive_shift, _VECTOR[offset]))

    def _roll_sense(self, axial_vector: tuple[int, int, int] | None, target: str) -> str:
        if axial_vector is None or not target:
            return UNDEFINED_ROTATION
        # A positive rotation about the axial axis carries the right wing toward n x right.
        return self._sense_from_alignment(_dot(_cross(axial_vector, _VECTOR[RIGHT]), _VECTOR[target]))

    def _sense_from_alignment(self, alignment: int) -> str:
        if alignment > 0:
            return POSITIVE_ROTATION
        if alignment < 0:
            return NEGATIVE_ROTATION
        return UNDEFINED_ROTATION


def _axis_rotation_matrix(axis_name: str, chirality: str) -> sp.Matrix:
    """Passive positive-rotation matrix bound to the X, Y, or Z slot. t is that angle.

    Right-handed XTZ is
    [[1, 0, 0], [0, cos(t), sin(t)], [0, -sin(t), cos(t)]].
    A left-handed frame flips the sign of t. The passive convention itself is fixed.
    """
    cosine = sp.cos(ROTATION_ANGLE)
    sine = sp.sin(ROTATION_ANGLE)
    if chirality == LEFT_HANDED:
        sine = -sine
    slot = {
        "XTZ": "X", "XFD": "X", "XTP": "X",
        "YTZ": "Y", "YFD": "Y", "YTP": "Y",
        "ZTZ": "Z", "ZFD": "Z", "ZYP": "Z",
    }.get(axis_name)
    if slot == "X":
        return sp.Matrix([
            [1, 0, 0],
            [0, cosine, sine],
            [0, -sine, cosine],
        ])
    if slot == "Y":
        return sp.Matrix([
            [cosine, 0, -sine],
            [0, 1, 0],
            [sine, 0, cosine],
        ])
    if slot == "Z":
        return sp.Matrix([
            [cosine, sine, 0],
            [-sine, cosine, 0],
            [0, 0, 1],
        ])
    raise DefinitionError(f"No coordinate axis named {axis_name}.")


def _placement_text(placement: str) -> str:
    if placement == PLACE_UPRIGHT:
        return "Axial, normal, and lateral axes point with the tunnel."
    if placement == PLACE_INVERTED:
        return "The axial axis points with the tunnel; normal and lateral axes point opposite the tunnel."
    return ""


def _append_frame(lines: list[str], title: str, names: tuple[str, ...], store: dict[str, AxisRecord]) -> None:
    lines.append(title)
    for name in names:
        axis = store[name]
        if not axis.points_to:
            lines.append(f"  {name}: not built")
            continue
        lines.append(f"  {name}: along {axis.absolute_axis}, points {axis.points_to} ({axis.source})")
        if axis.associated_angle:
            lines.append(f"    associated angle: {_label(axis.associated_angle)}")
            lines.append(f"    {_label(axis.associated_angle)} > 0: {axis.rotation_when_positive}")
        if axis.associated_mechanism_angle:
            lines.append(f"    associated mechanism angle: {_label(axis.associated_mechanism_angle)}")
            lines.append(
                f"    {_label(axis.associated_mechanism_angle)} > 0: {axis.mechanism_rotation_when_positive}"
            )
        if axis.installation_rotation_when_positive:
            lines.append(
                f"    {_label(axis.installation_angle)} > 0: {axis.installation_rotation_when_positive}"
            )
        if axis.rotation_matrix is not None:
            lines.append("    passive positive rotation R(t):")
            lines.append(_format_matrix(axis.rotation_matrix))


def _append_balance_body(lines: list[str], placement: str, alignment: dict[str, str]) -> None:
    if not placement:
        lines.append("  Placement is undeclared, so balance directions are not yet tied to the body.")
        return
    lines.append(f"  {placement}: a balance deformation moves the model the same way in the tunnel.")
    for direction in BALANCE_DIRECTIONS:
        lines.append(f"  {direction} -> {alignment[direction]}")


def _append_increment(lines: list[str], record: IncrementRecord) -> None:
    lines.append(f"  {_label(record.name)}: {record.definition or 'unset'}")
    if record.deformation:
        lines.append(f"    positive elastic deformation: {record.deformation}")
    if record.effect_toward:
        verb = "airspeed deflects toward" if record.kind == "airspeed" else "right wing moves toward"
        lines.append(f"    on the body: {verb} {record.effect_toward}")
    if record.theory_toward:
        verb = "airspeed toward" if record.kind == "airspeed" else "right wing toward"
        lines.append(f"    positive {_label(record.theoretical_angle)}: {verb} {record.theory_toward}")
    if record.increment_sign:
        lines.append(f"    scheduled increment sign at positive elastic deformation: {record.increment_sign}")


def _append_elastic(lines: list[str], record: AngleRecord) -> None:
    lines.append(f"  {_label(record.name)}: {record.definition or 'undeclared'}")
    if record.sense and record.positive_toward:
        lines.append(f"    positive angle moves the normal toward: {record.positive_toward}")
    elif record.positive_toward:
        lines.append(f"    positive {record.motion}: {record.positive_toward}")
    if record.associated_axis:
        source = f"absolute {record.absolute_axis} -> {record.associated_axis}" if record.absolute_axis else record.associated_axis
        lines.append(
            f"    {source}: positive angle is {record.rotation_when_positive}"
        )
    if record.component_signs:
        signs = ", ".join(
            f"{axis} {record.component_signs[axis]}" for axis in BALANCE_VOCAB.coordinate_axes
        )
        lines.append(f"    direction signs: {signs}")


def _append_angle(lines: list[str], record: AngleRecord) -> None:
    lines.append(f"  {_label(record.name)}: {record.definition or 'undeclared'}")
    if record.right_wing_toward:
        lines.append(f"    positive angle moves the right wing toward: {record.right_wing_toward}")
        if not record.associated_axis and record.rotation_when_positive != UNDEFINED_ROTATION:
            lines.append(f"    positive angle about the axial axis: {record.rotation_when_positive}")
    elif record.positive_airspeed_toward:
        lines.append(f"    positive angle deflects airspeed toward: {record.positive_airspeed_toward}")
    if record.associated_axis:
        source = f"absolute {record.absolute_axis} -> {record.associated_axis}" if record.absolute_axis else record.associated_axis
        lines.append(
            f"    {source}: positive angle is {record.rotation_when_positive}"
        )
    if record.airspeed_signs:
        signs = ", ".join(f"{axis} {record.airspeed_signs[axis]}" for axis in BODY_AXES)
        lines.append(f"    airspeed signs: {signs}")


def _format_matrix(matrix: sp.Matrix) -> str:
    rows = []
    for row in range(matrix.rows):
        cells = ", ".join(str(matrix[row, column]) for column in range(matrix.cols))
        rows.append(f"      [{cells}]")
    return "\n".join(rows)


def _check_elastic_increments() -> None:
    def declared(placement: str, alpha_toward: str, roll_toward: str) -> BodyFrameModel:
        model = BodyFrameModel()
        model.declare({
            ALPHA: {"airspeed_toward": alpha_toward, "sign": POSITIVE},
            BETA: {"airspeed_toward": RIGHT, "sign": POSITIVE},
            GAMMA: {"right_wing_toward": roll_toward, "sign": POSITIVE},
            "placement": placement,
            ALPHA_E: {"axial_bends": BALANCE_UP, "sign": POSITIVE},
            BETA_E: {"axial_bends": BALANCE_LEFT, "sign": POSITIVE},
            GAMMA_E: {"lateral_moves": BALANCE_DOWN, "sign": POSITIVE},
        })
        return model

    upright = declared(PLACE_UPRIGHT, DOWN, DOWN)
    assert upright.balance_body[BALANCE_UP] == UP
    assert upright.balance_body[BALANCE_RIGHT] == RIGHT
    alpha = upright.increments[DELTA_ALPHA]
    assert alpha.symbol == "Δα"
    assert alpha.effect_toward == DOWN
    assert alpha.increment_sign == POSITIVE
    beta = upright.increments[DELTA_BETA]
    assert beta.effect_toward == RIGHT
    assert beta.increment_sign == POSITIVE
    roll = upright.increments[DELTA_GAMMA]
    assert roll.effect_toward == DOWN
    assert roll.increment_sign == POSITIVE

    inverted = declared(PLACE_INVERTED, DOWN, DOWN)
    assert inverted.balance_body[BALANCE_UP] == DOWN
    assert inverted.balance_body[BALANCE_RIGHT] == LEFT
    assert inverted.increments[DELTA_ALPHA].effect_toward == UP
    assert inverted.increments[DELTA_ALPHA].increment_sign == NEGATIVE
    assert inverted.increments[DELTA_BETA].effect_toward == LEFT
    assert inverted.increments[DELTA_BETA].increment_sign == NEGATIVE
    assert inverted.increments[DELTA_GAMMA].effect_toward == DOWN
    assert inverted.increments[DELTA_GAMMA].increment_sign == POSITIVE

    nose_up = declared(PLACE_UPRIGHT, UP, UP)
    assert nose_up.increments[DELTA_ALPHA].increment_sign == NEGATIVE
    assert nose_up.increments[DELTA_GAMMA].increment_sign == NEGATIVE
    missing = BodyFrameModel()
    missing.declare({
        ALPHA_E: {"axial_bends": BALANCE_UP, "sign": POSITIVE},
        "placement": PLACE_UPRIGHT,
    })
    assert missing.increments[DELTA_ALPHA].effect_toward == DOWN
    assert missing.increments[DELTA_ALPHA].increment_sign == ""


def _check_flow_angles() -> None:
    model = BodyFrameModel()
    model.declare({
        ALPHA_V: {"airspeed_toward": DOWN, "sign": POSITIVE},
        BETA_V: {"airspeed_toward": RIGHT, "sign": POSITIVE},
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": NORMAL, "points": UP},
        },
    })
    alpha_v = model.flow_angles[ALPHA_V]
    assert alpha_v.symbol == "α_V"
    assert alpha_v.positive_airspeed_toward == DOWN
    assert alpha_v.airspeed_signs == {"XTZ": POSITIVE, "YTZ": NEGATIVE, "ZTZ": UNKNOWN}
    assert alpha_v.associated_axis == "ZTZ"
    assert alpha_v.rotation_when_positive == POSITIVE_ROTATION
    beta_v = model.flow_angles[BETA_V]
    assert beta_v.symbol == "β_V"
    assert beta_v.positive_airspeed_toward == RIGHT
    assert beta_v.airspeed_signs == {"XTZ": POSITIVE, "YTZ": UNKNOWN, "ZTZ": POSITIVE}
    assert beta_v.associated_axis == "YTZ"
    assert beta_v.rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["ZTZ"].rotation_when_positive == UNDEFINED_ROTATION
    model.declare({
        ALPHA_V: {"airspeed_toward": UP, "sign": POSITIVE},
        BETA_V: {"airspeed_toward": LEFT, "sign": POSITIVE},
        "placement": PLACE_INVERTED,
    })
    assert model.flow_angles[ALPHA_V].positive_airspeed_toward == UP
    assert model.flow_angles[ALPHA_V].airspeed_signs["YTZ"] == POSITIVE
    assert model.flow_angles[ALPHA_V].rotation_when_positive == NEGATIVE_ROTATION
    assert model.flow_angles[BETA_V].positive_airspeed_toward == LEFT
    assert model.flow_angles[BETA_V].airspeed_signs["ZTZ"] == NEGATIVE
    try:
        BodyFrameModel().declare({ALPHA_V: {"airspeed_toward": RIGHT, "sign": POSITIVE}})
    except DefinitionError:
        return
    raise AssertionError("alpha_v airspeed must lie above or below the fuselage.")


def _check_balance() -> None:
    model = BodyFrameModel()
    model.declare({
        ALPHA_E: {"axial_bends": BALANCE_UP, "sign": POSITIVE},
        BETA_E: {"axial_bends": BALANCE_LEFT, "sign": POSITIVE},
        GAMMA_E: {"lateral_moves": BALANCE_DOWN, "sign": POSITIVE},
        GAMMA_TP: {"sense": SENSE_SAME},
        "balance_frame": {
            "chirality": RIGHT_HANDED,
            "XTP": {"along": BALANCE_AXIAL, "points": BALANCE_FRONT},
            "YTP": {"along": BALANCE_LATERAL, "points": BALANCE_RIGHT},
        },
    })
    assert model.balance.directions == BALANCE_DIRECTIONS
    assert _VECTOR[BALANCE_FRONT] == _VECTOR[TUNNEL_FRONT]
    assert model.balance_frame["ZYP"].points_to == BALANCE_DOWN
    assert model.elastic_angles[ALPHA_E].symbol == "α_e"
    assert model.elastic_angles[ALPHA_E].component_signs == {
        "XTP": POSITIVE,
        "YTP": UNKNOWN,
        "ZYP": NEGATIVE,
    }
    assert model.elastic_angles[ALPHA_E].associated_axis == "YTP"
    assert model.elastic_angles[ALPHA_E].rotation_when_positive == NEGATIVE_ROTATION
    assert model.balance_frame["YTP"].rotation_when_positive == NEGATIVE_ROTATION
    assert model.elastic_angles[BETA_E].component_signs == {
        "XTP": POSITIVE,
        "YTP": NEGATIVE,
        "ZYP": UNKNOWN,
    }
    assert model.elastic_angles[BETA_E].rotation_when_positive == POSITIVE_ROTATION
    assert model.elastic_angles[GAMMA_E].rotation_when_positive == POSITIVE_ROTATION
    assert model.balance_frame["XTP"].rotation_when_positive == POSITIVE_ROTATION
    assert model.installation.symbol == "γ_tp"
    assert model.installation.sense == SENSE_SAME
    assert model.installation.positive_toward == BALANCE_RIGHT
    assert model.installation.rotation_when_positive == POSITIVE_ROTATION
    assert model.balance_frame["XTP"].installation_angle == GAMMA_TP
    assert model.balance_frame["XTP"].installation_rotation_when_positive == POSITIVE_ROTATION
    model.declare({GAMMA_TP: {"sense": SENSE_OPPOSITE}})
    assert model.installation.positive_toward == BALANCE_LEFT
    assert model.installation.rotation_when_positive == NEGATIVE_ROTATION
    assert model.elastic_angles[ALPHA_E].rotation_when_positive == NEGATIVE_ROTATION
    model.declare({
        GAMMA_TP: {"sense": SENSE_SAME},
        "balance_frame": {
            "chirality": RIGHT_HANDED,
            "XTP": {"along": BALANCE_AXIAL, "points": BALANCE_REAR},
            "YTP": {"along": BALANCE_NORMAL, "points": BALANCE_UP},
        },
    })
    assert model.installation.positive_toward == BALANCE_RIGHT
    assert model.installation.rotation_when_positive == POSITIVE_ROTATION
    assert model.balance_frame["XTP"].installation_rotation_when_positive == NEGATIVE_ROTATION
    model.declare({
        GAMMA_MODEL: {"sense": SENSE_SAME},
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": NORMAL, "points": UP},
        },
    })
    assert model.model_installation.symbol == "γ_model"
    assert model.model_installation.positive_toward == RIGHT
    assert model.model_installation.rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["XTZ"].installation_angle == GAMMA_MODEL
    assert model.body_frame["XTZ"].installation_rotation_when_positive == POSITIVE_ROTATION
    model.declare({"placement": PLACE_INVERTED, GAMMA_MODEL: {"sense": SENSE_OPPOSITE}})
    assert model.model_installation.positive_toward == LEFT
    assert model.model_installation.rotation_when_positive == NEGATIVE_ROTATION
    try:
        BodyFrameModel().declare({ALPHA_E: {"axial_bends": BALANCE_LEFT, "sign": POSITIVE}})
    except DefinitionError:
        return
    raise AssertionError("alpha_e must not bend along the lateral axis.")


def _check_alignment_matrix() -> None:
    mirror = BodyFrameModel()
    mirror.declare({
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": LATERAL, "points": RIGHT},
        },
        "wind_frame": {
            "chirality": LEFT_HANDED,
            "XFD": {"along": TUNNEL_AXIAL, "points": TUNNEL_FRONT},
            "YFD": {"along": TUNNEL_LATERAL, "points": TUNNEL_RIGHT},
        },
    })
    assert mirror.wind_frame["ZFD"].points_to == TUNNEL_UP
    assert mirror.wind_to_body.equals(sp.Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]))
    assert mirror.wind_to_body.det() == -1

    rearward = BodyFrameModel()
    rearward.declare({
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": REAR},
            "YTZ": {"along": LATERAL, "points": RIGHT},
        },
        "wind_frame": {
            "chirality": RIGHT_HANDED,
            "XFD": {"along": TUNNEL_AXIAL, "points": TUNNEL_FRONT},
            "YFD": {"along": TUNNEL_LATERAL, "points": TUNNEL_RIGHT},
        },
    })
    assert rearward.wind_to_body.equals(sp.Matrix([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1],
    ]))
    _check_placement()


def _check_placement() -> None:
    model = BodyFrameModel()
    model.declare({
        ALPHA: {"airspeed_toward": DOWN, "sign": POSITIVE},
        "placement": PLACE_INVERTED,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": LATERAL, "points": RIGHT},
        },
        "wind_frame": {
            "chirality": RIGHT_HANDED,
            "XFD": {"along": TUNNEL_AXIAL, "points": TUNNEL_FRONT},
            "YFD": {"along": TUNNEL_LATERAL, "points": TUNNEL_RIGHT},
        },
    })
    assert model.wind_to_body.equals(sp.Matrix([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1],
    ]))
    assert model.angles[ALPHA].airspeed_signs["ZTZ"] == POSITIVE
    assert model.airspeed.along_tunnel_axis == TUNNEL_FRONT

    free = BodyFrameModel()
    free.declare({
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": LATERAL, "points": RIGHT},
        },
        "wind_frame": {
            "chirality": RIGHT_HANDED,
            "XFD": {"along": TUNNEL_NORMAL, "points": TUNNEL_UP},
            "YFD": {"along": TUNNEL_LATERAL, "points": TUNNEL_RIGHT},
        },
    })
    assert free.wind_frame["XFD"].points_to == TUNNEL_UP
    assert free.wind_frame["ZFD"].points_to == TUNNEL_FRONT
    assert free.airspeed.along_tunnel_axis == TUNNEL_FRONT
    assert free.wind_to_body.equals(sp.Matrix([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ]))


def _check_standard_case() -> None:
    model = BodyFrameModel()
    model.declare({
        ALPHA: {"airspeed_toward": DOWN, "sign": POSITIVE},
        BETA: {"airspeed_toward": RIGHT, "sign": POSITIVE},
        GAMMA: {"right_wing_toward": DOWN, "sign": POSITIVE},
        "placement": PLACE_UPRIGHT,
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": NORMAL, "points": UP},
        },
        "wind_frame": {
            "chirality": RIGHT_HANDED,
            "XFD": {"along": TUNNEL_AXIAL, "points": TUNNEL_FRONT},
            "YFD": {"along": TUNNEL_NORMAL, "points": TUNNEL_UP},
        },
    })
    assert model.body_frame["ZTZ"].points_to == RIGHT
    assert model.wind_frame["ZFD"].points_to == TUNNEL_RIGHT
    assert model.wind_frame["XFD"].absolute_axis == TUNNEL_AXIAL
    assert model.wind_frame["XFD"].rotation_matrix.equals(model.body_frame["XTZ"].rotation_matrix)
    assert model.aircraft.rotation_type == INTRINSIC
    assert LOAD_BY_SYMBOL["CN"].axis("body") == NORMAL
    assert LOAD_BY_SYMBOL["CZ"].axis("body") == LATERAL
    assert LOAD_BY_SYMBOL["CA"].axis("body") == AXIAL
    assert LOAD_BY_SYMBOL["MZ"].axis("balance") == BALANCE_LATERAL
    assert LOAD_BY_SYMBOL["MA"].axis("balance") == BALANCE_AXIAL
    assert LOAD_BY_SYMBOL["MN"].axis("body") == NORMAL
    assert model.wind_tunnel.axes == WIND_AXES
    assert model.placement == PLACE_UPRIGHT
    assert model.airspeed.along_tunnel_axis == TUNNEL_FRONT
    assert model.wind_to_body.equals(sp.eye(3))
    assert model.angles[ALPHA].symbol == "α"
    assert model.angles[ALPHA].airspeed_signs == {
        "XTZ": POSITIVE,
        "YTZ": NEGATIVE,
        "ZTZ": UNKNOWN,
    }
    assert model.angles[BETA].airspeed_signs == {
        "XTZ": POSITIVE,
        "YTZ": UNKNOWN,
        "ZTZ": POSITIVE,
    }
    assert model.body_frame["YTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["ZTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert model.angles[GAMMA].right_wing_toward == DOWN
    assert model.angles[GAMMA].rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["XTZ"].rotation_when_positive == POSITIVE_ROTATION
    model.declare({
        ALPHA_M: {"airspeed_toward": UP, "sign": POSITIVE},
        BETA_M: {"airspeed_toward": LEFT, "sign": POSITIVE},
        GAMMA_M: {"right_wing_toward": UP, "sign": POSITIVE},
    })
    assert model.angles[ALPHA].airspeed_signs["YTZ"] == NEGATIVE
    assert model.angles[BETA].airspeed_signs["ZTZ"] == POSITIVE
    assert model.mechanism_angles[ALPHA_M].symbol == "α_M"
    assert model.mechanism_angles[ALPHA_M].associated_axis == "ZTZ"
    assert model.mechanism_angles[ALPHA_M].rotation_when_positive == NEGATIVE_ROTATION
    assert model.mechanism_angles[BETA_M].associated_axis == "YTZ"
    assert model.mechanism_angles[BETA_M].rotation_when_positive == NEGATIVE_ROTATION
    assert model.mechanism_angles[ALPHA_M].positive_airspeed_toward == UP
    assert model.mechanism_angles[ALPHA_M].airspeed_signs == {
        "XTZ": POSITIVE,
        "YTZ": POSITIVE,
        "ZTZ": UNKNOWN,
    }
    assert model.mechanism_angles[BETA_M].airspeed_signs == {
        "XTZ": POSITIVE,
        "YTZ": UNKNOWN,
        "ZTZ": NEGATIVE,
    }
    assert model.mechanism_angles[GAMMA_M].right_wing_toward == UP
    assert model.mechanism_angles[GAMMA_M].airspeed_signs == {}
    assert model.mechanism_angles[GAMMA_M].associated_axis == "XTZ"
    assert model.mechanism_angles[GAMMA_M].rotation_when_positive == NEGATIVE_ROTATION
    assert model.airspeed.positive_mechanism_beta_toward == LEFT
    assert model.body_frame["YTZ"].mechanism_rotation_when_positive == NEGATIVE_ROTATION
    assert model.body_frame["ZTZ"].mechanism_rotation_when_positive == NEGATIVE_ROTATION
    assert model.body_frame["XTZ"].mechanism_rotation_when_positive == NEGATIVE_ROTATION
    assert model.body_frame["YTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["XTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["ZTZ"].rotation_when_positive == POSITIVE_ROTATION
    cosine = sp.cos(ROTATION_ANGLE)
    sine = sp.sin(ROTATION_ANGLE)
    assert model.body_frame["XTZ"].rotation_matrix.equals(sp.Matrix([
        [1, 0, 0],
        [0, cosine, sine],
        [0, -sine, cosine],
    ]))
    assert model.body_frame["YTZ"].rotation_matrix.equals(sp.Matrix([
        [cosine, 0, -sine],
        [0, 1, 0],
        [sine, 0, cosine],
    ]))
    assert model.body_frame["ZTZ"].rotation_matrix.equals(sp.Matrix([
        [cosine, sine, 0],
        [-sine, cosine, 0],
        [0, 0, 1],
    ]))
    model.declare({BETA: {"airspeed_toward": LEFT, "sign": POSITIVE}})
    assert model.wind_to_body.equals(sp.eye(3))
    assert model.airspeed.positive_alpha_toward == DOWN
    assert model.angles[BETA].airspeed_signs["ZTZ"] == NEGATIVE
    assert model.body_frame["ZTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert model.body_frame["ZTZ"].mechanism_rotation_when_positive == NEGATIVE_ROTATION
    assert model.body_frame["YTZ"].rotation_when_positive == NEGATIVE_ROTATION

    left = BodyFrameModel()
    left.declare({
        ALPHA: {"airspeed_toward": DOWN, "sign": POSITIVE},
        BETA: {"airspeed_toward": RIGHT, "sign": POSITIVE},
        "body_frame": {
            "chirality": LEFT_HANDED,
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": LATERAL, "points": RIGHT},
        },
    })
    assert left.body_frame["ZTZ"].points_to == UP
    assert left.body_frame["ZTZ"].source == INFERRED
    assert left.angles[ALPHA].airspeed_signs["ZTZ"] == NEGATIVE
    assert left.body_frame["ZTZ"].rotation_when_positive == POSITIVE_ROTATION
    assert left.body_frame["XTZ"].rotation_matrix.equals(sp.Matrix([
        [1, 0, 0],
        [0, cosine, -sine],
        [0, sine, cosine],
    ]))
    _check_balance()
    _check_elastic_increments()
    _check_alignment_matrix()

    errors = [
        {
            "body_frame": {
                "chirality": RIGHT_HANDED,
                "rotation": "active",
                "XTZ": {"along": AXIAL, "points": FRONT},
                "YTZ": {"along": LATERAL, "points": RIGHT},
            },
        },
        {
            "body_frame": {
                "chirality": RIGHT_HANDED,
                "XTZ": {"along": AXIAL, "points": DOWN},
                "YTZ": {"along": LATERAL, "points": RIGHT},
            },
        },
        {
            "body_frame": {
                "chirality": RIGHT_HANDED,
                "XTZ": {"along": AXIAL, "points": FRONT},
                "YTZ": {"along": AXIAL, "points": REAR},
            },
        },
        {
            "body_frame": {
                "chirality": RIGHT_HANDED,
                "XTZ": {"along": AXIAL, "points": FRONT},
                "YTZ": {"along": LATERAL, "points": RIGHT},
                "ZTZ": {"along": NORMAL, "points": UP},
            },
        },
        {
            "body_frame": {
                "chirality": RIGHT_HANDED,
                "euler_order": ["gamma", "beta", "alpha"],
                "XTZ": {"along": AXIAL, "points": FRONT},
                "YTZ": {"along": NORMAL, "points": UP},
            },
        },
        {"placement": "sideways"},
        {"rotation_type": "extrinsic"},
        {"CN": "axial"},
        {"load_coefficients": {"CN": "lateral"}},
    ]
    wrist = BodyFrameModel().declare({
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "euler_order": ["beta", "alpha", "gamma"],
            "XTZ": {"along": AXIAL, "points": FRONT},
            "YTZ": {"along": NORMAL, "points": UP},
        },
    })
    if tuple(wrist.euler_order) != WRIST_ORDER:
        raise AssertionError(tuple(wrist.euler_order))
    for spec in errors:
        try:
            BodyFrameModel().declare(spec)
        except DefinitionError:
            continue
        raise AssertionError(f"contradictory definition should fail: {spec}")


if __name__ == "__main__":
    _check_standard_case()
    _check_flow_angles()
    model = BodyFrameModel()
    model.declare({
        "alpha": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma": {"right_wing_toward": "body_down", "sign": "positive"},
        "alpha_m": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta_m": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma_m": {"right_wing_toward": "body_down", "sign": "positive"},
        "placement": "upright",
        "body_frame": {
            "chirality": "right",
            "XTZ": {"along": "body_axial", "points": "body_forward"},
            "YTZ": {"along": "body_normal", "points": "body_up"},
        },
        "wind_frame": {
            "chirality": "right",
            "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
            "YFD": {"along": "tunnel_normal", "points": "tunnel_up"},
        },
        "alpha_e": {"axial_bends": "balance_up", "sign": "positive"},
        "beta_e": {"axial_bends": "balance_left", "sign": "positive"},
        "gamma_e": {"lateral_moves": "balance_down", "sign": "positive"},
        "alpha_v": {"airspeed_toward": "body_down", "sign": "positive"},
        "beta_v": {"airspeed_toward": "body_right", "sign": "positive"},
        "gamma_tp": {"sense": "same"},
        "gamma_model": {"sense": "same"},
        "balance_frame": {
            "chirality": "right",
            "XTP": {"along": "balance_axial", "points": "balance_forward"},
            "YTP": {"along": "balance_normal", "points": "balance_up"},
        },
    })
    print(model.describe())
    print("\n--- beta positive toward body_left ---\n")
    model.declare({"beta": {"airspeed_toward": "body_left", "sign": "positive"}})
    print(model.describe())
