# -*- coding: utf-8 -*-
"""Where the cloze sets are stored.

Generators stay in domains/. This module is the only place that names the
data directories.

Four parallel domains. Their manuscripts do not mix. data/ has no
cross / reconstruction / basic selection layer.

wind_tunnel       风洞。主实验 data/wind_tunnel/single
flight            飞行力学。主实验 data/flight/review/flow
robotics          机器人学。主实验 data/robotics/single
vision            机器视觉。主实验 data/vision/single，只含物体与双目长篇幅稿

SETS keeps legacy aliases for old generator scripts. Those keys are not a
second taxonomy; they resolve into the four domains above, or into removed
write targets that no longer hold selectable data.

EVAL is the cloze runner's main-experiment entry: one standard definition
and nine single-definition perturbations per domain. Larger multi-factor
corpora stay on disk and are not the selector.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

_FLIGHT = DATA / "flight"
_ROBOTICS = DATA / "robotics"
_VISION = DATA / "vision"
_WIND = DATA / "wind_tunnel"

# Removed 3-tier write targets. Generators may still name these paths; they
# are not part of DOMAINS and must not be used as gold selection.
_REMOVED_ARCHIVE = DATA / "_removed" / "archive"
_REMOVED_FLIGHT_BASIC = DATA / "_removed" / "flight_basic"
_REMOVED_FLIGHT_RECON = DATA / "_removed" / "flight_reconstruction"


def _slices(base: Path) -> dict[str, Path]:
    return {"review": base / "review", "stress": base / "stress", "batch": base / "batch"}


DOMAINS = {
    "wind_tunnel": {
        "zh": "风洞",
        "status": "已有数据",
        "root": _WIND,
        "review": _WIND / "review",
        "stress": _WIND / "stress",
        "batch": _WIND / "batch",
        "pair": _WIND / "pair",
    },
    "aerodynamics": {
        "zh": "飞行力学",
        "status": "已有数据",
        "root": _FLIGHT,
        "review": _FLIGHT / "review",
        "flow": {"zh": "空间坐标转换", "review": _FLIGHT / "review" / "flow"},
    },
    "robotics": {
        "zh": "机器人学",
        "status": "已有数据",
        "root": _ROBOTICS,
        "review": _ROBOTICS / "review",
        "kinematics": {"zh": "运动学", "review": _ROBOTICS / "review" / "kinematics"},
    },
    "computer_vision": {
        "zh": "机器视觉",
        "status": "已有数据",
        "root": _VISION,
        "review": _VISION / "review",
        "scene": {"zh": "物体与双目环节", "review": _VISION / "review" / "scene"},
        "theory": {"zh": "基础理论环节", "review": _VISION / "review" / "scene"},
    },
}


SETS = {
    "cross.wind_flight": {
        "root": _WIND,
        "review": _WIND / "review",
        "stress": _WIND / "stress",
        "batch": _WIND / "batch",
        "pair": _WIND / "pair",
    },
    "cross.wind_vision": {
        "root": _REMOVED_ARCHIVE / "wind_vision",
        "review": _REMOVED_ARCHIVE / "wind_vision" / "review",
        "samples": _REMOVED_ARCHIVE / "wind_vision" / "samples.jsonl",
        "pilot": _REMOVED_ARCHIVE / "wind_vision" / "pilot_10.json",
    },
    "cross.wind_arm": {
        "root": _REMOVED_ARCHIVE / "wind_arm",
        "review": _REMOVED_ARCHIVE / "wind_arm" / "review",
        "stress": _REMOVED_ARCHIVE / "wind_arm" / "stress",
        "batch": _REMOVED_ARCHIVE / "wind_arm" / "batch",
    },
    "reconstruction.flight": {
        "root": _REMOVED_FLIGHT_RECON,
        "review": _REMOVED_FLIGHT_RECON / "review",
        "synced": _REMOVED_FLIGHT_RECON / "synced_10.jsonl",
        "samples": _REMOVED_FLIGHT_RECON / "samples.jsonl",
    },
    "reconstruction.vision": {
        "root": _VISION,
        "review": _VISION / "review" / "scene",
        "samples": _VISION / "review" / "scene" / "samples.jsonl",
    },
    "reconstruction.arm": {
        "root": _ROBOTICS,
        "review": _ROBOTICS / "review" / "kinematics",
    },
    "basic.flight": {
        "root": _REMOVED_FLIGHT_BASIC,
        "pilot": _REMOVED_FLIGHT_BASIC / "pilot",
        "processes": {
            "ground_to_body": {
                "zh": "地面系到机体系",
                **_slices(_REMOVED_FLIGHT_BASIC / "ground_to_body"),
            },
            "ground_to_track_air": {
                "zh": "地面系转航迹系和气流系",
                **_slices(_REMOVED_FLIGHT_BASIC / "ground_to_track_air"),
            },
            "body_to_airflow": {
                "zh": "体轴系到气流系",
                **_slices(_REMOVED_FLIGHT_BASIC / "body_to_airflow"),
            },
        },
    },
    "basic.vision": {
        "root": _VISION,
        "review": _VISION / "review" / "scene",
        "samples": _VISION / "review" / "scene" / "samples.jsonl",
        "pilot": _VISION / "review" / "scene" / "pilot_10.json",
    },
    "basic.arm": {
        "root": _ROBOTICS,
        "review": _ROBOTICS / "review" / "kinematics",
    },
}


# Percentage of variable definitions changed. The older wind-tunnel batch and
# the two-stage vision review still carry these four tiers. The main-experiment
# sets use one unperturbed sample and nine single-factor samples, all tier 0.
TIERS = (25, 50, 75, 100)

EVAL_DOMAINS = ("wind_tunnel", "flight", "cv", "robotics")

EVAL = {
    "wind_tunnel": {
        "zh": "风洞",
        "data": _WIND / "single" / "samples.jsonl",
        "kind": "jsonl",
        "tiers": (),
    },
    "flight": {
        "zh": "飞行力学",
        "data": _FLIGHT / "review" / "flow",
        "kind": "manuscript",
        "tiers": (),
    },
    "cv": {
        "zh": "机器视觉",
        "data": _VISION / "single",
        "kind": "manuscript",
        "tiers": (),
    },
    "robotics": {
        "zh": "机器人学",
        "data": _ROBOTICS / "single",
        "kind": "manuscript",
        "tiers": (),
    },
}


def process_dirs(domain: str, process: str) -> dict[str, Path]:
    """Legacy generator outputs for removed flight basic slices."""
    if domain != "aerodynamics":
        raise KeyError(domain)
    found = SETS["basic.flight"]["processes"][process]
    return {"review": found["review"], "stress": found["stress"], "batch": found["batch"]}
