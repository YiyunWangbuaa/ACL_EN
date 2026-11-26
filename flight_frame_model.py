# -*- coding: utf-8 -*-
"""Definition model for flight-mechanics frames and Euler-angle axis bindings.

Five frames keep fixed absolute axes. A declaration chooses chirality, which
coordinate axis lies on which absolute axis, and which way each Euler angle is
positive. From that, every angle is bound to the coordinate axis on its
rotation axis, together with the sense of a positive angle about that axis.

The shared embedding is the wind-tunnel basis, forward x geometric-right = down.
It is only used for cross products. It is not a coordinate frame.

Absolute lateral axes follow one right-hand rule on every frame:
lateral = normal x axial. The track note lists axial before normal; the same
rule is used there. Every normal points down, including the ground normal, so
every rightward direction is geometric right. "Right of the ground" uses this named direction.

The industry default puts every frame on the same right-handed habit: X lies
on the axial axis and points with it, Z lies on the normal axis and points
down with that normal, and Y is fixed by the right-hand rule. Track azimuth,
when its sign is positive, is positive on the geometric right of xg.

Rotation sense is the right-hand rule about the coordinate axis's pointing
direction. A left-handed frame does not flip that label; its passive matrix
flips the sign of the angle instead. Angle of attack, sideslip, attitude,
and flight-path angles rotate the named vector itself. Positive angle of
attack turns the body axial axis down, the reverse sense about the rightward
lateral axis. Positive sideslip turns the stability axial axis to the right,
the forward sense about the downward normal.

Bindings are stored on directed transforms, not as one list on an axis.
Ground to body carries the attitude angles; body to ground carries the same
angles with the opposite sense. Ground to airflow carries azimuth, inclination,
and bank. Ground to track carries azimuth and inclination only. Track to
airflow carries bank only. Body to airflow carries sideslip and angle of
attack. Each reverse transform flips those senses. One
coordinate axis can therefore hold several bindings, selected by the origin
and the destination.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


class DefinitionError(ValueError):
    """A declaration contradicts an absolute axis or the declared chirality."""


_GEOM = {
    "forward": (1, 0, 0),
    "rearward": (-1, 0, 0),
    "right": (0, 1, 0),
    "left": (0, -1, 0),
    "down": (0, 0, 1),
    "up": (0, 0, -1),
}
_GEOM_NAME = {vector: name for name, vector in _GEOM.items()}
_GEOM_ZH = {
    "forward": "forward",
    "rearward": "rearward",
    "right": "right",
    "left": "left",
    "up": "up",
    "down": "down",
}
_OPPOSITE_GEOM = {
    "forward": "rearward",
    "rearward": "forward",
    "right": "left",
    "left": "right",
    "up": "down",
    "down": "up",
}

POSITIVE = "positive"
NEGATIVE = "negative"
POSITIVE_ROTATION = "positive_rotation"
NEGATIVE_ROTATION = "negative_rotation"
UNDEFINED_ROTATION = "undefined"
_SENSE_ZH = {
    POSITIVE_ROTATION: "positive",
    NEGATIVE_ROTATION: "negative",
    UNDEFINED_ROTATION: "undefined",
}
RIGHT_HANDED = "right"
LEFT_HANDED = "left"
DECLARED = "declared"
INFERRED = "inferred"
FIXED = "fixed"
ROTATION_ANGLE = sp.symbols("t")

OWNERS = ("ground", "body", "airflow", "stability", "track")
COORDINATE_AXES = {
    "ground": ("Xg", "Yg", "Zg"),
    "body": ("Xb", "Yb", "Zb"),
    "airflow": ("Xa", "Ya", "Za"),
    "stability": ("Xs", "Ys", "Zs"),
    "track": ("Xk", "Yk", "Zk"),
}
FRAME_KEY = {owner: f"{owner}_frame" for owner in OWNERS}
# Every normal points down. Lateral is then normal x axial, which is geometric right.
_NORMAL_GEOM = {
    "ground": "down",
    "body": "down",
    "airflow": "down",
    "stability": "down",
    "track": "down",
}
_ROLES = ("axial", "normal", "lateral")

THETA = "theta"
PSI = "psi"
PHI = "phi"
MU = "mu"
VARPHI = "varphi"
GAMMA = "gamma"
ALPHA = "alpha"
BETA = "beta"
ATTITUDE_ORDER = (PSI, THETA, PHI)
TRACK_ORDER = (VARPHI, MU, GAMMA)
AIRFLOW_ORDER = (BETA, ALPHA)
ANGLE_NAMES = ATTITUDE_ORDER + TRACK_ORDER + AIRFLOW_ORDER


def _cross(a: tuple[int, int, int], b: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _dot(a: tuple[int, int, int], b: tuple[int, int, int]) -> int:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _mul(vector: tuple[int, int, int], scale: int) -> tuple[int, int, int]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _geom_name(vector: tuple[int, int, int]) -> str:
    try:
        return _GEOM_NAME[vector]
    except KeyError as exc:
        raise DefinitionError(f"Direction {vector} is outside the six absolute directions.") from exc


@dataclass(frozen=True)
class FrameGeometry:
    """Absolute axes of one frame. Coordinate bindings do not move these."""

    owner: str
    coordinate_axes: tuple[str, str, str]
    positive: dict[str, str]
    vector: dict[str, tuple[int, int, int]]
    axis_of: dict[str, str]
    opposite: dict[str, str]
    lateral_embeds: str

    def direction(self, role: str, sense: str = POSITIVE) -> str:
        name = self.positive[role]
        if sense == NEGATIVE:
            return self.opposite[name]
        return name


def _build_geometry(owner: str) -> FrameGeometry:
    normal = _GEOM[_NORMAL_GEOM[owner]]
    axial = _GEOM["forward"]
    lateral = _cross(normal, axial)
    positive = {
        "axial": f"{owner}_forward",
        "normal": f"{owner}_{_geom_name(normal)}",
        "lateral": f"{owner}_right",
    }
    vector = {
        f"{owner}_forward": axial,
        f"{owner}_rearward": _mul(axial, -1),
        f"{owner}_up": _GEOM["up"],
        f"{owner}_down": _GEOM["down"],
        f"{owner}_right": lateral,
        f"{owner}_left": _mul(lateral, -1),
    }
    axis_of = {}
    for role, direction in positive.items():
        axis_of[direction] = f"{owner}_{role}"
        axis_of[f"{owner}_{_OPPOSITE_GEOM[direction.removeprefix(owner + '_')]}"] = f"{owner}_{role}"
    opposite = {
        name: f"{owner}_{_OPPOSITE_GEOM[name.removeprefix(owner + '_')]}" for name in vector
    }
    return FrameGeometry(
        owner=owner,
        coordinate_axes=COORDINATE_AXES[owner],
        positive=positive,
        vector=vector,
        axis_of=axis_of,
        opposite=opposite,
        lateral_embeds=_geom_name(lateral),
    )


GEOMETRY = {owner: _build_geometry(owner) for owner in OWNERS}


def _embed_direction(owner: str, geom_name: str) -> str:
    """Direction of this frame that lies along a shared embedding axis."""
    vector = _GEOM[geom_name]
    for name, value in GEOMETRY[owner].vector.items():
        if value == vector:
            return name
    raise DefinitionError(f"{owner} has no direction along geometric {geom_name}.")


@dataclass(frozen=True)
class AngleSpec:
    """Which absolute axis an Euler angle turns, and what a positive angle moves.

    ``pair`` is the two frames the angle sits between, written in the order of
    the positive (forward) transform. The rotation matrix is written in the
    coordinates of the frame the rotation starts from: ``pair[0]`` on the forward
    transform, ``pair[1]`` on the inverse one.
    """

    name: str
    symbol: str
    title: str
    group: str
    pair: tuple[str, str]
    role: str
    kind: str
    moving: str
    toward: str
    motion: str

    @property
    def frame(self) -> str:
        """Frame of the forward transform (the positive direction)."""
        return self.pair[0]


def _angle_spec(name: str, symbol: str, title: str, group: str, pair: tuple[str, str], role: str, kind: str, moving: str, toward: str, motion: str) -> AngleSpec:
    return AngleSpec(name, symbol, title, group, pair, role, kind, moving, toward, motion)


ANGLE_SPEC = {
    item.name: item
    for item in (
        _angle_spec(THETA, "θ", "pitch", "attitude", ("ground", "body"), "lateral", "active", "body_forward", "ground_up", "the nose pitches above the ground"),
        _angle_spec(PSI, "ψ", "yaw", "attitude", ("ground", "body"), "normal", "active", "body_forward", "ground_right", "the nose yaws to the right of the ground"),
        _angle_spec(PHI, "Φ", "roll", "attitude", ("ground", "body"), "axial", "roll", "body_right", "body_down", "the aircraft rolls to its right"),
        _angle_spec(MU, "μ", "flight-path inclination", "track", ("ground", "track"), "lateral", "active", "track_forward", "ground_up", "the airspeed pitches above the ground"),
        _angle_spec(VARPHI, "φ", "track azimuth", "track", ("ground", "track"), "normal", "active", "airflow_forward", _embed_direction("ground", "right"), "the airspeed projection lies to the right of xg"),
        _angle_spec(GAMMA, "γ", "flight-path bank", "track", ("track", "airflow"), "axial", "roll", "body_right", "body_down", "the aircraft rolls to its right"),
        _angle_spec(ALPHA, "α", "angle of attack", "airflow", ("body", "stability"), "lateral", "active", "body_forward", "body_down", "the airspeed projection lies below the airframe axis"),
        _angle_spec(BETA, "β", "sideslip", "airflow", ("stability", "airflow"), "normal", "active", "body_forward", "body_right", "the airspeed lies to the right of the aircraft"),
    )
}
# Geometric definitions. Sign still comes from the sample and does not change the rotation direction.
_ATTITUDE_CLAUSE = {
    THETA: "the angle between the positive airframe axis and the ground. When the nose pitches above the ground, the angle is",
    PSI: "the angle between the ground projection of the positive airframe axis and the ground axis Xg. When the nose yaws to the right of the ground, the angle is",
    PHI: "the angle between the positive airframe normal and the vertical plane through the airframe axis. When the aircraft rolls to its right, the angle is",
    ALPHA: "the angle between the projection of the airspeed onto the aircraft plane of symmetry and the positive airframe axis. When that projection lies below the airframe axis, the angle is",
    BETA: "the angle between the airspeed vector and the aircraft plane of symmetry. When the airspeed lies to the right of the aircraft, the angle is",
    MU: "the angle between the airspeed vector (the positive airflow axis) and the ground. When the aircraft climbs, the angle is",
    VARPHI: "the angle between the ground projection of the airspeed vector (the positive airflow axis) and the ground axis Xg. When the projection lies to the right of Xg, the angle is",
    GAMMA: "the angle between the positive airflow normal and the vertical plane through the airspeed. When the aircraft rolls to its right, the angle is",
}
GROUP_ORDER = {
    "attitude": ATTITUDE_ORDER,
    "track": TRACK_ORDER,
    "airflow": AIRFLOW_ORDER,
}
# Directed transforms. The last flag is true when this direction is the inverse
# of the angle definitions, so every positive sense on that transform is flipped.
# Attitude carries the ground into the body. Azimuth and inclination carry the
# ground into the track, and also into the airflow. Bank carries the track into
# the airflow, and is part of the ground-to-airflow rotation as well.
# Sideslip then angle of attack carry the body into the airflow frame.
DIRECTED_TRANSFORMS = (
    ("ground", "body", ATTITUDE_ORDER, False),
    ("body", "ground", ATTITUDE_ORDER, True),
    ("ground", "airflow", TRACK_ORDER, False),
    ("airflow", "ground", TRACK_ORDER, True),
    ("ground", "track", (VARPHI, MU), False),
    ("track", "ground", (VARPHI, MU), True),
    ("track", "airflow", (GAMMA,), False),
    ("airflow", "track", (GAMMA,), True),
    ("body", "stability", (ALPHA,), False),
    ("stability", "body", (ALPHA,), True),
    ("stability", "airflow", (BETA,), False),
    ("airflow", "stability", (BETA,), True),
    ("body", "airflow", AIRFLOW_ORDER, False),
    ("airflow", "body", AIRFLOW_ORDER, True),
)
FRAME_ZH = {
    "ground": "ground",
    "body": "body",
    "airflow": "airflow",
    "stability": "stability",
    "track": "track",
}


@dataclass
class AxisRecord:
    """One coordinate axis after a declaration."""

    name: str
    owner: str
    absolute_axis: str = ""
    points_to: str = ""
    source: str = ""
    transform_bindings: tuple[TransformBinding, ...] = ()
    rotation_matrix: sp.Matrix | None = None

    def vector(self, geometry: FrameGeometry) -> tuple[int, int, int]:
        return geometry.vector[self.points_to]


@dataclass
class AngleRecord:
    """Binding of one Euler angle to a coordinate axis."""

    name: str
    symbol: str
    title: str
    group: str
    frame: str
    absolute_axis: str
    motion: str
    sign: str = ""
    positive_toward: str = ""
    definition: str = ""
    associated_axis: str = ""
    rotation_when_positive: str = UNDEFINED_ROTATION


@dataclass(frozen=True)
class TransformBinding:
    """One angle bound to one absolute direction for one directed transform.

    ``absolute_axis`` is the direction the rotation turns about (for example
    ``body_lateral``). ``axis`` is the coordinate axis of ``frame`` that lies on
    that direction; it is inferred from the declaration, never declared itself.
    """

    origin: str
    destination: str
    angle: str
    frame: str
    absolute_axis: str
    axis: str
    rotation_when_positive: str
    inverse: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.origin, self.destination)


def _passive_matrix(axis_name: str, chirality: str) -> sp.Matrix:
    """Passive positive rotation about the X, Y, or Z slot. Left-handed frames flip t."""
    cosine = sp.cos(ROTATION_ANGLE)
    sine = sp.sin(ROTATION_ANGLE)
    if chirality == LEFT_HANDED:
        sine = -sine
    slot = axis_name[0]
    if slot == "X":
        return sp.Matrix([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]])
    if slot == "Y":
        return sp.Matrix([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    if slot == "Z":
        return sp.Matrix([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    raise DefinitionError(f"No coordinate axis named {axis_name}.")


class FlightFrameModel:
    """Read a flight-mechanics declaration and bind every Euler angle to an axis."""

    def __init__(self) -> None:
        self.geometry = GEOMETRY
        self.frames = {
            owner: {name: AxisRecord(name, owner) for name in COORDINATE_AXES[owner]}
            for owner in OWNERS
        }
        self.chirality = {owner: "" for owner in OWNERS}
        self.angles = {
            name: AngleRecord(
                name, spec.symbol, spec.title, spec.group, spec.frame,
                f"{spec.frame}_{spec.role}", spec.motion,
            )
            for name, spec in ANGLE_SPEC.items()
        }
        self._frame_spec = {owner: None for owner in OWNERS}
        self._sign = {name: "" for name in ANGLE_NAMES}
        self.transforms: dict[tuple[str, str], tuple[TransformBinding, ...]] = {}

    def declare(self, spec: dict) -> FlightFrameModel:
        """Store the given definitions and recompute every axis binding."""
        if "euler_order" in spec:
            raise DefinitionError(
                "Euler composition order is fixed: yaw, pitch, roll; "
                "track azimuth, inclination, bank; sideslip, then angle of attack."
            )
        unknown = set(spec) - set(FRAME_KEY.values()) - set(ANGLE_NAMES)
        if unknown:
            raise DefinitionError(f"Unknown definition: {', '.join(sorted(unknown))}.")
        for owner, key in FRAME_KEY.items():
            if key in spec:
                self._frame_spec[owner] = spec[key]
        for name in ANGLE_NAMES:
            if name in spec:
                self._set_sign(name, spec[name])
        self._refresh()
        return self

    def conversion(self, origin: str, destination: str) -> tuple[TransformBinding, ...]:
        """Angle bindings for the rotation that carries origin into destination."""
        try:
            return self.transforms[(origin, destination)]
        except KeyError as exc:
            raise DefinitionError(f"No directed transform from {origin} to {destination}.") from exc

    def binding(self, name: str) -> AngleRecord:
        if name not in self.angles:
            raise DefinitionError(f"No Euler angle named {name}.")
        return self.angles[name]

    def alignment(self, source: str, target: str) -> sp.Matrix:
        """Map source coordinates into target coordinates at zero Euler angles.

        Column j is the positive direction of source axis j in target components.
        """
        if source not in OWNERS or target not in OWNERS:
            raise DefinitionError("alignment frames must be ground, body, airflow, stability, or track.")
        source_axes = [self.frames[source][name] for name in COORDINATE_AXES[source]]
        target_axes = [self.frames[target][name] for name in COORDINATE_AXES[target]]
        if any(not axis.points_to for axis in source_axes + target_axes):
            raise DefinitionError(f"{source} and {target} frames must both be declared.")
        source_geometry = self.geometry[source]
        target_geometry = self.geometry[target]
        return sp.Matrix([
            [
                _dot(target_axis.vector(target_geometry), source_axis.vector(source_geometry))
                for source_axis in source_axes
            ]
            for target_axis in target_axes
        ])

    def describe(self) -> str:
        lines = [
            "Embedding: forward x geometric right = down. Used only for cross products, not as a coordinate frame.",
            "The positive lateral direction is always normal x axial. All five normals point down, so every rightward direction is geometric right.",
            "Euler order is fixed, not a definition. Attitude: yaw, pitch, roll. Track: azimuth, inclination, bank. Airflow: sideslip, angle of attack.",
            "Sense follows the right-hand rule about the coordinate axis's pointing direction. A left-handed frame does not rename that sense; the passive matrix flips the angle sign instead.",
            "Angle of attack, sideslip, attitude angles, and flight-path angles rotate the vector named in the definition.",
            "Bindings are split by directed transform. Ground to body, ground to track, track to airflow, and body to airflow are the forward definitions; the reverse transform flips the sense.",
            "Each angle is first bound to an absolute-direction axis, then the concrete coordinate axis is inferred from how that frame's axes correspond to the absolute directions.",
            "A directed transform looks only at its origin and destination. Each transform uses the absolute directions of its origin frame. The inverse starts where the forward transform ends and only flips the sense. It does not consult intermediate frames.",
            "Ground to airflow includes every flight-path angle. Ground to track omits flight-path bank. Between airflow and track there is only flight-path bank.",
            "Sideslip and angle of attack are bound between the body and the airflow.",
            "",
        ]
        for owner in OWNERS:
            geometry = self.geometry[owner]
            lines.append(
                f"{owner} absolute axes: axial {geometry.positive['axial']}, "
                f"normal {geometry.positive['normal']}, "
                f"lateral {geometry.positive['lateral']}"
                f" (after embedding this points geometrically {_GEOM_ZH[geometry.lateral_embeds]})."
            )
        lines.append("")
        for owner in OWNERS:
            _append_frame(lines, self.frames[owner], self.chirality[owner], COORDINATE_AXES[owner])
            lines.append("")
        lines.append("Euler-angle definitions")
        for group, names in GROUP_ORDER.items():
            lines.append(f"  [{group}]")
            for name in names:
                _append_angle(lines, self.angles[name], self.frames)
        lines.append("")
        lines.append("Directed transforms")
        for origin, destination, _angles, inverse in DIRECTED_TRANSFORMS:
            direction = "inverse" if inverse else "forward"
            lines.append(f"  {FRAME_ZH[origin]} -> {FRAME_ZH[destination]} ({direction})")
            for item in self.transforms.get((origin, destination), ()):
                _append_transform(lines, item)
        return "\n".join(lines)

    def _set_sign(self, name: str, spec: dict) -> None:
        if set(spec) - {"sign"}:
            raise DefinitionError(f"{ANGLE_SPEC[name].title} only accepts sign.")
        sign = spec.get("sign", POSITIVE)
        if sign not in (POSITIVE, NEGATIVE):
            raise DefinitionError(f"{ANGLE_SPEC[name].title} sign must be positive or negative.")
        self._sign[name] = sign

    def _refresh(self) -> None:
        for owner in OWNERS:
            spec = self._frame_spec[owner]
            store = self.frames[owner]
            for axis in store.values():
                axis.absolute_axis = ""
                axis.points_to = ""
                axis.source = ""
                axis.transform_bindings = ()
                axis.rotation_matrix = None
            self.chirality[owner] = ""
            if spec is not None:
                self._build_frame(owner, spec, store)
        for name in ANGLE_NAMES:
            self._bind_angle(name)
        self._bind_transforms()

    def _build_frame(self, owner: str, spec: dict, store: dict[str, AxisRecord]) -> None:
        geometry = self.geometry[owner]
        chirality = spec.get("chirality")
        if chirality not in (RIGHT_HANDED, LEFT_HANDED):
            raise DefinitionError(f"{owner} chirality must be right or left.")
        if "rotation" in spec or "rotation_type" in spec:
            raise DefinitionError("Frame rotation is passive and is not a definition.")
        names = geometry.coordinate_axes
        allowed = set(names) | {"chirality"}
        unknown = set(spec) - allowed
        if unknown:
            raise DefinitionError(f"Unknown {owner} frame entry: {', '.join(sorted(unknown))}.")
        declared = {}
        sources = {}
        if owner == "ground":
            fixed = {"along": f"{owner}_axial", "points": geometry.positive["axial"]}
            if "Xg" in spec and spec["Xg"] != fixed:
                raise DefinitionError("Xg is fixed on the ground axial axis and points forward.")
            declared["Xg"] = (f"{owner}_axial", geometry.positive["axial"])
            sources["Xg"] = FIXED
        for name in names:
            if name in spec and not (owner == "ground" and name == "Xg"):
                declared[name] = self._parse_axis(name, spec[name], geometry)
                sources[name] = DECLARED
        if len(declared) < 2:
            raise DefinitionError(
                "Declare Yg or Zg; Xg is fixed and chirality fixes the other."
                if owner == "ground"
                else f"Declare at least two {owner} axes; chirality fixes the remaining one."
            )
        used = [item[0] for item in declared.values()]
        if len(used) != len(set(used)):
            raise DefinitionError(f"Two {owner} axes were declared along the same absolute axis.")
        if len(declared) == 3:
            ordered = [geometry.vector[declared[name][1]] for name in names]
            expected = _cross(ordered[0], ordered[1]) if chirality == RIGHT_HANDED else _cross(ordered[1], ordered[0])
            if expected != ordered[2]:
                raise DefinitionError(
                    f"{owner} axis directions contradict a {chirality}-handed frame. "
                    f"Chirality requires {names[2]} to point {_direction_name(expected, geometry)}."
                )
        elif len(declared) == 2:
            missing, vector = _missing_axis(declared, names, geometry, chirality)
            declared[missing] = (geometry.axis_of[_direction_name(vector, geometry)], _direction_name(vector, geometry))
            sources[missing] = INFERRED
        else:
            raise DefinitionError(f"{owner} needs two declared axes.")
        self.chirality[owner] = chirality
        for name in names:
            absolute_axis, points_to = declared[name]
            axis = store[name]
            axis.absolute_axis = absolute_axis
            axis.points_to = points_to
            axis.source = sources[name]
            axis.rotation_matrix = _passive_matrix(name, chirality)

    def _parse_axis(self, name: str, spec: dict, geometry: FrameGeometry) -> tuple[str, str]:
        absolute_axis = spec.get("along")
        points_to = spec.get("points")
        allowed_axes = tuple(f"{geometry.owner}_{role}" for role in _ROLES)
        if absolute_axis not in allowed_axes:
            raise DefinitionError(f"{name} along must be one of {', '.join(allowed_axes)}.")
        if points_to not in geometry.vector:
            raise DefinitionError(f"{name} points must be one of the six {geometry.owner} directions.")
        if geometry.axis_of[points_to] != absolute_axis:
            raise DefinitionError(f"{name} is along {absolute_axis} but points {points_to}; the two do not match.")
        return absolute_axis, points_to

    def _axis_on(self, owner: str, absolute_axis: str) -> AxisRecord | None:
        """The coordinate axis of ``owner`` that lies on an absolute direction."""
        return next(
            (
                item for item in self.frames[owner].values()
                if item.absolute_axis == absolute_axis
            ),
            None,
        )

    def _axis_for(self, name: str, frame: str) -> AxisRecord | None:
        """The axis of ``frame`` that carries the role this angle turns about."""
        role = ANGLE_SPEC[name].role
        return self._axis_on(frame, f"{frame}_{role}")

    def _sense_in_frame(self, name: str, axis: AxisRecord, frame: str) -> str:
        """Turn sense of a positive angle about this axis, in this frame's coordinates."""
        spec = ANGLE_SPEC[name]
        sign = self._sign[name]
        if not sign:
            return UNDEFINED_ROTATION
        toward = spec.toward if sign == POSITIVE else geometry_opposite(spec.toward)
        return _rotation_sense(
            axis.vector(self.geometry[frame]),
            self._vector(spec.moving),
            self._vector(toward),
            spec.kind,
        )

    def _bind_angle(self, name: str) -> None:
        spec = ANGLE_SPEC[name]
        record = self.angles[name]
        sign = self._sign[name]
        start = spec.pair[0]
        record.frame = start
        record.absolute_axis = f"{start}_{spec.role}"
        record.sign = sign
        record.positive_toward = ""
        record.definition = ""
        record.associated_axis = ""
        record.rotation_when_positive = UNDEFINED_ROTATION
        axis = self._axis_for(name, start)
        if axis is None:
            return
        record.associated_axis = axis.name
        if not sign:
            record.definition = "The coordinate axis is bound, and its positive sense is not declared."
            return
        toward = spec.toward if sign == POSITIVE else geometry_opposite(spec.toward)
        record.positive_toward = toward
        sense = " positive" if sign == POSITIVE else " negative"
        clause = _ATTITUDE_CLAUSE.get(name)
        record.definition = f"{clause}{sense}" if clause else f"{spec.motion}{sense}"
        record.rotation_when_positive = self._sense_in_frame(name, axis, start)

    def _bind_transforms(self) -> None:
        """Bind each angle to an absolute direction axis of the transform's start frame.

        The start frame is the origin of the transform. The inverse flag only
        flips the sense of a positive angle, because the definitions are stated
        for the forward direction.

        Forward ground to body writes theta and phi about ground directions;
        backward body to ground writes the same angles about body directions.
        A direct body-to-airflow transform writes both airflow angles about body
        directions, while the stability-to-airflow transform writes sideslip
        about the stability normal.
        """
        transforms: dict[tuple[str, str], tuple[TransformBinding, ...]] = {}
        for origin, destination, angles, inverse in DIRECTED_TRANSFORMS:
            start = origin
            rows = []
            for name in angles:
                role = ANGLE_SPEC[name].role
                absolute = f"{start}_{role}"
                axis = self._axis_on(start, absolute)
                axis_name = ""
                sense = UNDEFINED_ROTATION
                if axis is not None:
                    axis_name = axis.name
                    sense = self._sense_in_frame(name, axis, start)
                    if inverse:
                        sense = _inverse_sense(sense)
                item = TransformBinding(
                    origin, destination, name, start, absolute, axis_name, sense, inverse,
                )
                rows.append(item)
                if axis is not None:
                    axis.transform_bindings = (*axis.transform_bindings, item)
            transforms[(origin, destination)] = tuple(rows)
        self.transforms = transforms

    def _vector(self, direction: str) -> tuple[int, int, int]:
        owner = direction.split("_", 1)[0]
        try:
            return self.geometry[owner].vector[direction]
        except KeyError as exc:
            raise DefinitionError(f"No direction named {direction}.") from exc


def geometry_opposite(direction: str) -> str:
    owner = direction.split("_", 1)[0]
    return GEOMETRY[owner].opposite[direction]


def _direction_name(vector: tuple[int, int, int], geometry: FrameGeometry) -> str:
    for name, value in geometry.vector.items():
        if value == vector:
            return name
    raise DefinitionError(f"Inferred direction {vector} is not one of the six {geometry.owner} directions.")


def _missing_axis(declared: dict, names: tuple[str, str, str], geometry: FrameGeometry, chirality: str) -> tuple[str, tuple[int, int, int]]:
    missing = next(name for name in names if name not in declared)
    index = names.index(missing)
    first = names[(index + 1) % 3]
    second = names[(index + 2) % 3]
    vector = _cross(geometry.vector[declared[first][1]], geometry.vector[declared[second][1]])
    if chirality == LEFT_HANDED:
        vector = _mul(vector, -1)
    return missing, vector


def _inverse_sense(sense: str) -> str:
    if sense == POSITIVE_ROTATION:
        return NEGATIVE_ROTATION
    if sense == NEGATIVE_ROTATION:
        return POSITIVE_ROTATION
    return sense


def _rotation_sense(axis: tuple[int, int, int], moving: tuple[int, int, int], toward: tuple[int, int, int], kind: str) -> str:
    carried = _cross(axis, moving)
    if kind == "passive":
        carried = _mul(carried, -1)
    alignment = _dot(carried, toward)
    if alignment > 0:
        return POSITIVE_ROTATION
    if alignment < 0:
        return NEGATIVE_ROTATION
    raise DefinitionError("The declared positive motion is not a rotation about this axis.")


def _append_frame(lines: list[str], store: dict[str, AxisRecord], chirality: str, names: tuple[str, ...]) -> None:
    owner = next(iter(store.values())).owner
    lines.append(f"{owner} coordinate axes ({chirality or 'undeclared'})")
    for name in names:
        axis = store[name]
        if not axis.points_to:
            lines.append(f"  {name}: not established")
            continue
        lines.append(f"  {name}: along {axis.absolute_axis}, pointing {axis.points_to} ({axis.source})")
        for item in axis.transform_bindings:
            spec = ANGLE_SPEC[item.angle]
            lines.append(
                f"    {FRAME_ZH[item.origin]} -> {FRAME_ZH[item.destination]}: "
                f"{spec.symbol} {_SENSE_ZH[item.rotation_when_positive]}"
            )


def _append_angle(lines: list[str], record: AngleRecord, frames: dict[str, dict[str, AxisRecord]]) -> None:
    lines.append(f"  {record.symbol} {record.title}: {record.definition or 'undeclared'}")
    if not record.associated_axis:
        lines.append(f"    frame {record.frame} is not established, so {record.absolute_axis} cannot be bound")
        return
    axis = frames[record.frame][record.associated_axis]
    lines.append(
        f"    bound to {record.associated_axis} of {record.frame}, "
        f"along {axis.absolute_axis}, pointing {axis.points_to}"
    )
    lines.append(f"    forward sense about that axis: {_SENSE_ZH[record.rotation_when_positive]}")


def _append_transform(lines: list[str], item: TransformBinding) -> None:
    spec = ANGLE_SPEC[item.angle]
    if not item.axis:
        lines.append(f"    {spec.symbol} {spec.title}: absolute direction {item.absolute_axis} is not established")
        return
    lines.append(
        f"    {spec.symbol} {spec.title}: absolute direction {item.absolute_axis} -> "
        f"{item.axis} of {FRAME_ZH[item.frame]}, positive-angle sense {_SENSE_ZH[item.rotation_when_positive]}"
    )


def _standard_frame(owner: str) -> dict:
    """Right-handed forward-right-down binding. X follows the axial axis; Z points down."""
    x_name, _y_name, z_name = COORDINATE_AXES[owner]
    down = _embed_direction(owner, "down")
    spec = {
        "chirality": RIGHT_HANDED,
        z_name: {"along": GEOMETRY[owner].axis_of[down], "points": down},
    }
    if owner != "ground":
        spec[x_name] = {"along": f"{owner}_axial", "points": GEOMETRY[owner].positive["axial"]}
    return spec


STANDARD_SPEC = {
    "ground_frame": _standard_frame("ground"),
    "body_frame": _standard_frame("body"),
    "airflow_frame": _standard_frame("airflow"),
    "stability_frame": _standard_frame("stability"),
    "track_frame": _standard_frame("track"),
    THETA: {"sign": POSITIVE},
    PSI: {"sign": POSITIVE},
    PHI: {"sign": POSITIVE},
    MU: {"sign": POSITIVE},
    VARPHI: {"sign": POSITIVE},
    GAMMA: {"sign": POSITIVE},
    ALPHA: {"sign": POSITIVE},
    BETA: {"sign": POSITIVE},
}


def _check() -> None:
    ground = GEOMETRY["ground"]
    body = GEOMETRY["body"]
    assert ground.positive["normal"] == "ground_down"
    assert ground.vector["ground_down"] == _GEOM["down"]
    assert ground.vector["ground_right"] == _GEOM["right"]
    assert ground.lateral_embeds == "right"
    assert body.vector["body_right"] == _GEOM["right"]
    assert body.vector["body_down"] == _GEOM["down"]
    assert GEOMETRY["track"].vector["track_right"] == _GEOM["right"]
    assert _cross(ground.vector["ground_down"], ground.vector["ground_forward"]) == ground.vector["ground_right"]
    assert _cross(body.vector["body_down"], body.vector["body_forward"]) == body.vector["body_right"]

    model = FlightFrameModel().declare(STANDARD_SPEC)
    assert model.frames["ground"]["Xg"].source == FIXED
    assert model.frames["ground"]["Xg"].points_to == "ground_forward"
    assert model.frames["ground"]["Zg"].points_to == "ground_down"
    assert model.frames["ground"]["Zg"].source == DECLARED
    assert model.frames["ground"]["Yg"].points_to == _embed_direction("ground", "right")
    assert model.frames["ground"]["Yg"].source == INFERRED
    for owner, (x_name, y_name, z_name) in COORDINATE_AXES.items():
        frame = model.frames[owner]
        geometry = model.geometry[owner]
        assert geometry.vector[frame[x_name].points_to] == _GEOM["forward"]
        assert geometry.vector[frame[y_name].points_to] == _GEOM["right"]
        assert geometry.vector[frame[z_name].points_to] == _GEOM["down"]
        assert model.chirality[owner] == RIGHT_HANDED
    assert model.frames["body"]["Yb"].points_to == "body_right"
    assert model.frames["stability"]["Ys"].points_to == "stability_right"
    assert model.frames["track"]["Yk"].points_to == "track_right"

    expected = {
        THETA: ("Yg", POSITIVE_ROTATION),
        PSI: ("Zg", POSITIVE_ROTATION),
        PHI: ("Xg", POSITIVE_ROTATION),
        MU: ("Yg", POSITIVE_ROTATION),
        VARPHI: ("Zg", POSITIVE_ROTATION),
        GAMMA: ("Xk", POSITIVE_ROTATION),
        ALPHA: ("Yb", NEGATIVE_ROTATION),
        BETA: ("Zs", POSITIVE_ROTATION),
    }
    for name, (axis_name, sense) in expected.items():
        record = model.binding(name)
        assert record.associated_axis == axis_name, (name, record.associated_axis)
        assert record.rotation_when_positive == sense, (name, record.rotation_when_positive)

    # forward transform: the matrix is written in the starting frame, here ground
    direct = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
              for item in model.conversion("ground", "body")}
    assert direct == {
        (PSI, "ground", "Zg", POSITIVE_ROTATION),
        (THETA, "ground", "Yg", POSITIVE_ROTATION),
        (PHI, "ground", "Xg", POSITIVE_ROTATION),
    }
    # inverse transform: the same angles written in body axes, with the sense flipped
    inverse = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
               for item in model.conversion("body", "ground")}
    assert inverse == {
        (PSI, "body", "Zb", NEGATIVE_ROTATION),
        (THETA, "body", "Yb", NEGATIVE_ROTATION),
        (PHI, "body", "Xb", NEGATIVE_ROTATION),
    }
    airflow = [item.angle for item in model.conversion("ground", "airflow")]
    assert airflow == [VARPHI, MU, GAMMA]
    assert [item.angle for item in model.conversion("ground", "track")] == [VARPHI, MU]
    assert [item.angle for item in model.conversion("track", "airflow")] == [GAMMA]
    assert [item.angle for item in model.conversion("airflow", "track")] == [GAMMA]
    # the frame a binding is written in follows the direction of travel
    assert [(i.angle, i.frame) for i in model.conversion("ground", "airflow")] == [
        (VARPHI, "ground"), (MU, "ground"), (GAMMA, "ground"),
    ]
    assert [(i.angle, i.frame) for i in model.conversion("airflow", "ground")] == [
        (VARPHI, "airflow"), (MU, "airflow"), (GAMMA, "airflow"),
    ]
    assert [(i.angle, i.frame) for i in model.conversion("ground", "track")] == [
        (VARPHI, "ground"), (MU, "ground"),
    ]
    assert [(i.angle, i.frame) for i in model.conversion("track", "ground")] == [
        (VARPHI, "track"), (MU, "track"),
    ]
    zg = [(item.origin, item.destination, item.angle) for item in model.frames["ground"]["Zg"].transform_bindings]
    assert zg == [
        ("ground", "body", PSI),
        ("ground", "airflow", VARPHI),
        ("ground", "track", VARPHI),
    ]
    yg = [(item.origin, item.destination, item.angle) for item in model.frames["ground"]["Yg"].transform_bindings]
    assert yg == [
        ("ground", "body", THETA),
        ("ground", "airflow", MU),
        ("ground", "track", MU),
    ]
    assert model.conversion("track", "airflow")[0].rotation_when_positive == POSITIVE_ROTATION
    assert model.conversion("airflow", "track")[0].rotation_when_positive == NEGATIVE_ROTATION
    assert model.conversion("ground", "airflow")[2].angle == GAMMA
    assert model.conversion("ground", "airflow")[2].rotation_when_positive == POSITIVE_ROTATION
    assert model.conversion("airflow", "ground")[2].rotation_when_positive == NEGATIVE_ROTATION
    assert [item.angle for item in model.conversion("body", "airflow")] == [BETA, ALPHA]
    assert [item.angle for item in model.conversion("airflow", "body")] == [BETA, ALPHA]
    # a direct body-to-airflow transform binds both angles to body directions
    body_air = {(item.angle, item.frame, item.absolute_axis, item.axis, item.rotation_when_positive)
                for item in model.conversion("body", "airflow")}
    assert body_air == {
        (BETA, "body", "body_normal", "Zb", POSITIVE_ROTATION),
        (ALPHA, "body", "body_lateral", "Yb", NEGATIVE_ROTATION),
    }
    air_body = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
                for item in model.conversion("airflow", "body")}
    assert air_body == {
        (BETA, "airflow", "Za", NEGATIVE_ROTATION),
        (ALPHA, "airflow", "Ya", POSITIVE_ROTATION),
    }
    # the leg transforms bind to their own start frames
    leg_bs = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
              for item in model.conversion("body", "stability")}
    assert leg_bs == {(ALPHA, "body", "Yb", NEGATIVE_ROTATION)}
    leg_sb = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
              for item in model.conversion("stability", "body")}
    assert leg_sb == {(ALPHA, "stability", "Ys", POSITIVE_ROTATION)}
    leg_sa = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
              for item in model.conversion("stability", "airflow")}
    assert leg_sa == {(BETA, "stability", "Zs", POSITIVE_ROTATION)}
    leg_as = {(item.angle, item.frame, item.axis, item.rotation_when_positive)
              for item in model.conversion("airflow", "stability")}
    assert leg_as == {(BETA, "airflow", "Za", NEGATIVE_ROTATION)}
    zb = [(item.origin, item.destination, item.angle) for item in model.frames["body"]["Zb"].transform_bindings]
    assert zb == [("body", "ground", PSI), ("body", "airflow", BETA)]
    assert model.binding(VARPHI).positive_toward == _embed_direction("ground", "right")
    assert model.alignment("body", "ground").equals(sp.eye(3))
    cosine = sp.cos(ROTATION_ANGLE)
    sine = sp.sin(ROTATION_ANGLE)
    assert model.frames["body"]["Xb"].rotation_matrix.equals(sp.Matrix([
        [1, 0, 0], [0, cosine, sine], [0, -sine, cosine],
    ]))

    model.declare({THETA: {"sign": NEGATIVE}})
    assert model.binding(THETA).positive_toward == "ground_down"
    assert model.binding(THETA).rotation_when_positive == NEGATIVE_ROTATION
    assert model.binding(PSI).rotation_when_positive == POSITIVE_ROTATION
    theta = {item.key: item.rotation_when_positive
             for item in model.frames["ground"]["Yg"].transform_bindings if item.angle == THETA}
    assert theta == {("ground", "body"): NEGATIVE_ROTATION}
    inverse_theta = [item for item in model.conversion("body", "ground") if item.angle == THETA][0]
    assert inverse_theta.rotation_when_positive == POSITIVE_ROTATION
    yk_angles = {(item.origin, item.destination, item.angle) for item in model.frames["track"]["Yk"].transform_bindings}
    assert GAMMA not in {angle for _origin, _destination, angle in yk_angles}
    assert yk_angles == {("track", "ground", MU)}
    ya_angles = {(item.origin, item.destination, item.angle) for item in model.frames["airflow"]["Ya"].transform_bindings}
    assert ya_angles == {("airflow", "ground", MU), ("airflow", "body", ALPHA)}

    rearward = FlightFrameModel().declare({
        "body_frame": {
            "chirality": RIGHT_HANDED,
            "Xb": {"along": "body_axial", "points": "body_rearward"},
            "Zb": {"along": "body_normal", "points": "body_down"},
        },
        PHI: {"sign": POSITIVE},
        BETA: {"sign": POSITIVE},
    })
    assert rearward.frames["body"]["Xb"].points_to == "body_rearward"
    rearward_inverse = {item.angle: item for item in rearward.conversion("body", "ground")}
    assert rearward_inverse[PHI].axis == "Xb"
    assert rearward_inverse[PHI].rotation_when_positive == POSITIVE_ROTATION
    assert rearward.binding(ALPHA).associated_axis == "Yb"
    assert rearward.binding(BETA).associated_axis == ""

    left = FlightFrameModel().declare({
        "body_frame": {
            "chirality": LEFT_HANDED,
            "Xb": {"along": "body_axial", "points": "body_forward"},
            "Yb": {"along": "body_lateral", "points": "body_right"},
        },
        BETA: {"sign": POSITIVE},
    })
    assert left.frames["body"]["Zb"].points_to == "body_up"
    assert left.frames["body"]["Zb"].source == INFERRED
    assert left.binding(BETA).associated_axis == ""
    assert left.frames["body"]["Zb"].rotation_matrix.equals(sp.Matrix([
        [cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1],
    ]))
    assert left.frames["body"]["Xb"].rotation_matrix[1, 2] == -sine

    errors = [
        {"ground_frame": {"chirality": RIGHT_HANDED, "Xg": {"along": "ground_normal", "points": "ground_up"}, "Yg": {"along": "ground_axial", "points": "ground_forward"}}},
        {"ground_frame": {"chirality": RIGHT_HANDED}},
        {"body_frame": {"chirality": RIGHT_HANDED, "Xb": {"along": "body_axial", "points": "body_down"}, "Yb": {"along": "body_lateral", "points": "body_right"}}},
        {"body_frame": {"chirality": RIGHT_HANDED, "Xb": {"along": "body_axial", "points": "body_forward"}, "Yb": {"along": "body_axial", "points": "body_rearward"}}},
        {"body_frame": {"chirality": RIGHT_HANDED, "Xb": {"along": "body_axial", "points": "body_forward"}, "Yb": {"along": "body_lateral", "points": "body_right"}, "Zb": {"along": "body_normal", "points": "body_up"}}},
        {"euler_order": ["phi", "theta", "psi"]},
        {THETA: {"toward": "ground_down"}},
        {"placement": "upright"},
    ]
    for spec in errors:
        try:
            FlightFrameModel().declare(spec)
        except DefinitionError:
            continue
        raise AssertionError(f"contradictory definition should fail: {spec}")


if __name__ == "__main__":
    _check()
    print(FlightFrameModel().declare(STANDARD_SPEC).describe())
    print("flight frame bindings checked")
