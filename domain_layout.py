# -*- coding: utf-8 -*-
"""Where the cloze sets are stored.

Generators stay in domains/. This module is the only place that names the
data directories.

Four parallel domains. Their manuscripts do not mix.

wind_tunnel       风洞。直接采用 data/wind_tunnel
flight            飞行力学。成稿只在 data/flight/review/flow
robotics          机器人学。重构数据加基础理论
vision            机器视觉。重构数据加基础理论

SETS below still names the folders the generators write. The keys
cross.wind_vision and cross.wind_arm stay, and now point at data/archive.
Those two folders are archives, not domains.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def _slices(base: Path) -> dict[str, Path]:
    return {"review": base / "review", "stress": base / "stress", "batch": base / "batch"}


_FLIGHT = DATA / "flight"
_ROBOTICS = DATA / "robotics"
_VISION = DATA / "vision"
_WIND_FLIGHT = DATA / "wind_tunnel"
_WIND_VISION = DATA / "archive" / "wind_vision"
_WIND_ARM = DATA / "archive" / "wind_arm"
_RECON_FLIGHT = _FLIGHT / "reconstruction"
_RECON_VISION = _VISION / "reconstruction"
_RECON_ARM = _ROBOTICS / "review" / "kinematics"
_BASIC_FLIGHT = _FLIGHT / "basic"
_BASIC_VISION = _VISION / "basic"
_BASIC_ARM = _ROBOTICS / "review" / "theory"

_GROUND_BODY = _slices(_BASIC_FLIGHT / "ground_to_body")
_GROUND_TRACK = _slices(_BASIC_FLIGHT / "ground_to_track_air")
_BODY_AIR = _slices(_BASIC_FLIGHT / "body_to_airflow")


SETS = {
    "cross.wind_flight": {
        "class": "cross",
        "class_zh": "交叉数据",
        "zh": "风洞-飞行力学",
        "root": _WIND_FLIGHT,
        "review": _WIND_FLIGHT / "review",
        "stress": _WIND_FLIGHT / "stress",
        "batch": _WIND_FLIGHT / "batch",
        "pair": _WIND_FLIGHT / "pair",
    },
    "cross.wind_vision": {
        "class": "cross",
        "class_zh": "交叉数据",
        "zh": "风洞-机器视觉",
        "root": _WIND_VISION,
        "review": _WIND_VISION / "review",
        "samples": _WIND_VISION / "samples.jsonl",
        "pilot": _WIND_VISION / "pilot_10.json",
    },
    "cross.wind_arm": {
        "class": "cross",
        "class_zh": "交叉数据",
        "zh": "风洞-机械臂",
        "root": _WIND_ARM,
        "review": _WIND_ARM / "review",
        "stress": _WIND_ARM / "stress",
        "batch": _WIND_ARM / "batch",
    },
    "reconstruction.flight": {
        "class": "reconstruction",
        "class_zh": "重构数据",
        "zh": "飞行力学重构",
        "root": _RECON_FLIGHT,
        "review": _RECON_FLIGHT / "review",
        "synced": _RECON_FLIGHT / "synced_10.jsonl",
        "samples": _RECON_FLIGHT / "samples.jsonl",
    },
    "reconstruction.vision": {
        "class": "reconstruction",
        "class_zh": "重构数据",
        "zh": "机器视觉重构",
        "root": _RECON_VISION,
        "review": _RECON_VISION / "review",
        "samples": _RECON_VISION / "samples.jsonl",
    },
    "reconstruction.arm": {
        "class": "reconstruction",
        "class_zh": "重构数据",
        "zh": "机器人学重构",
        "root": _RECON_ARM,
        "review": _RECON_ARM,
    },
    "basic.flight": {
        "class": "basic",
        "class_zh": "基础数据",
        "zh": "飞行力学基础变换",
        "root": _BASIC_FLIGHT,
        "pilot": _BASIC_FLIGHT / "pilot",
        "processes": {
            "ground_to_body": {"zh": "地面系到机体系", **_GROUND_BODY},
            "ground_to_track_air": {"zh": "地面系转航迹系和气流系", **_GROUND_TRACK},
            "body_to_airflow": {"zh": "体轴系到气流系", **_BODY_AIR},
        },
    },
    "basic.vision": {
        "class": "basic",
        "class_zh": "基础数据",
        "zh": "机器视觉基础理论",
        "root": _BASIC_VISION,
        "review": _BASIC_VISION / "review",
        "samples": _BASIC_VISION / "samples.jsonl",
        "pilot": _BASIC_VISION / "pilot_10.json",
    },
    "basic.arm": {
        "class": "basic",
        "class_zh": "基础数据",
        "zh": "机器人学理论",
        "root": _BASIC_ARM,
        "review": _BASIC_ARM,
    },
}


DOMAINS = {
    "wind_tunnel": {
        "zh": "风洞",
        "status": "已有数据",
        "root": SETS["cross.wind_flight"]["root"],
        "review": SETS["cross.wind_flight"]["review"],
        "stress": SETS["cross.wind_flight"]["stress"],
        "batch": SETS["cross.wind_flight"]["batch"],
        "pair": SETS["cross.wind_flight"]["pair"],
    },
    "aerodynamics": {
        "zh": "飞行力学",
        "status": "已有数据",
        "root": _FLIGHT,
        "review": _FLIGHT / "review",
        "pilot": SETS["basic.flight"]["pilot"],
        "mechanism": {
            "zh": "机构角到理论气流角",
            "review": _FLIGHT / "review" / "mechanism",
        },
        "basic_review": _FLIGHT / "review" / "basic",
        "flow": {"zh": "空间坐标转换", "review": _FLIGHT / "review" / "flow"},
        "processes": {
            key: {"zh": item["zh"], "review": item["review"], "stress": item["stress"], "batch": item["batch"]}
            for key, item in SETS["basic.flight"]["processes"].items()
        },
    },
    "robotics": {
        "zh": "机器人学",
        "status": "已有数据",
        "root": _ROBOTICS,
        "review": _ROBOTICS / "review",
        "kinematics": {"zh": "运动学", "review": _ROBOTICS / "review" / "kinematics"},
        "theory": {"zh": "基础理论", "review": _ROBOTICS / "review" / "theory"},
    },
    "computer_vision": {
        "zh": "机器视觉",
        "status": "已有数据",
        "root": _VISION,
        "review": _VISION / "review",
        "scene": {"zh": "物体与双目", "review": _VISION / "review" / "scene"},
        "theory": {"zh": "基础理论", "review": _VISION / "review" / "theory"},
    },
}


def process_dirs(domain: str, process: str) -> dict[str, Path]:
    found = DOMAINS[domain]["processes"][process]
    return {"review": found["review"], "stress": found["stress"], "batch": found["batch"]}
