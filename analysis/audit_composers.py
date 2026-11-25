# -*- coding: utf-8 -*-
"""Exposure of composer factors: latent counts hidden factors, alias counts collisions.

Each target has a hand-written factor table. A factor is either

    {"绑定": (题号, 选项)}
    {"内部": (选项, 来源标签)}

and never both. latent is the number of 内部 factors. alias is the number of
unordered pairs of assignments whose pasted strings are equal. An empty table
is 未填: it is not latent=0, and it does not enter the completeness verdict.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.contract_verifiers import _compose_moment  # noqa: E402
from domains.wind_tunnel.ref_point_generate import (  # noqa: E402
    ARM_OPTIONS,
    DENOM_OPTIONS,
    FORCE_OPTIONS,
)


def _bind(question_id: str, options: tuple[str, ...]) -> dict:
    return {"绑定": (question_id, options)}


def _hide(options: tuple[str, ...], source: str, repair: str) -> dict:
    """repair is 独立 (add a question) or 派生 (compute it from questions that already exist)."""
    if repair not in {"独立", "派生"}:
        raise RuntimeError(f"内部因子的修法只能是独立或派生：{repair}")
    return {"内部": (options, source, repair)}


_SIGN = ("正", "负")

# 已填的目标。拆前是历史对照，用来卡住 latent=3，不代表当前题面。
FACTORS: dict[str, dict[str, dict]] = {
    "力矩公式/拆后": {
        "符号": _bind("ID190", _SIGN),
        "力系数": _bind("ID202", FORCE_OPTIONS),
        "坐标": _bind("ID208", ARM_OPTIONS),
        "分母": _bind("ID213", DENOM_OPTIONS),
    },
    "力矩公式/拆前": {
        "符号": _bind("ID190", _SIGN),
        "力系数": _hide(FORCE_OPTIONS, "力矩名", "独立"),
        "坐标": _hide(ARM_OPTIONS, "ID184", "独立"),
        "分母": _hide(DENOM_OPTIONS, "常量 L_RT", "独立"),
    },
    "attitude_target": {
        "坐标系": _bind("ID23/ID31/ID40", ("中间体轴系", "天平系", "风轴系")),
        "轴": _bind("ID24/ID32/ID41", ("XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP")),
        "转向": _bind("ID25/ID33/ID42", ("正", "负", "不确定")),
        "手性": _hide(
            ("左", "右"),
            "按所绕轴读声明：体轴 chirality 对应 ID5/ID6/ID7，"
            "风轴 wind_chirality 对应 ID2/ID3/ID4，"
            "天平 balance_chirality 对应 ID89/ID90/ID91；三元组没有传入",
            "派生",
        ),
    },
    "roll_target": {
        "安装手": _bind("ID83", ("正", "负")),
        "机体手": _bind("ID84", ("正", "负")),
        "轴": _bind("ID86", ("XTP", "YTP", "ZYP")),
        "转向": _bind("ID87", ("正", "逆")),
        "天平三元组": _bind("ID89/ID90/ID91", ("XTP", "YTP", "ZYP", "-XTP", "-YTP", "-ZYP")),
        "手性": _hide(
            ("左", "右"),
            "轴只在天平系，矩阵手性仍读 balance_chirality；ID89/ID90/ID91 已传入，只用于转角",
            "派生",
        ),
    },
    "_installation": {
        "轴": _bind("ID130", ("XTP", "YTP", "ZYP")),
        "转向": _bind("ID131", ("正", "逆")),
        "手性": _hide(
            ("左", "右"),
            "轴只在天平系，读 balance_chirality；输入只有 ID130/ID131，没有 ID89/ID90/ID91",
            "派生",
        ),
    },
    "_product": {
        "连乘码": _bind("ID47/ID48/ID49", ("RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M")),
        "约定": _bind("ID44/ID45/ID46", ("主动", "被动", "动轴", "定轴", "内旋", "外旋")),
    },
    "_alignment": {
        "行绑定": _bind("ID5/ID6/ID7", ("XTZ", "YTZ", "ZTZ", "-XTZ", "-YTZ", "-ZTZ")),
        "列绑定": _bind("ID2/ID3/ID4", ("XFD", "YFD", "ZFD", "-XFD", "-YFD", "-ZFD")),
    },
    "_mount": {
        "轴绑定": _bind("ID5/ID6/ID7", ("XTZ", "YTZ", "ZTZ", "-XTZ", "-YTZ", "-ZTZ")),
        "安装关系": _bind("ID12/ID96", ("相同", "相反", "正装", "倒装")),
    },
}

# 空表不参与判定。这四个合成器已经填进 FACTORS。
PENDING: tuple[str, ...] = ()


def compose(sign: str, force: str, arm: str, denom: str) -> str:
    """Same paste as _compose_moment: optional minus, then force·arm/denom."""
    prefix = "-" if sign == "负" else ""
    return f"{prefix}{force}·{arm}/{denom}"


RENDER = {
    "力矩公式/拆后": compose,
    "力矩公式/拆前": compose,
}


def _kind(slot: dict) -> str:
    if set(slot) == {"绑定"}:
        return "绑定"
    if set(slot) == {"内部"}:
        return "内部"
    raise RuntimeError(f"因子必须只标绑定或内部：{slot}")


def _options(slot: dict) -> tuple[str, ...]:
    kind = _kind(slot)
    if kind == "绑定":
        _question, options = slot["绑定"]
        return tuple(options)
    options, _source, _repair = slot["内部"]
    return tuple(options)


def latent(table: dict) -> int:
    """How many direct factors have no question of their own."""
    if not table:
        raise RuntimeError("空表是未填，不能当成 latent=0")
    return sum(1 for slot in table.values() if _kind(slot) == "内部")


def sources(table: dict) -> tuple[tuple[str, str, str], ...]:
    """Internal factors, where the value comes from, and whether to add a question."""
    found = []
    for name, slot in table.items():
        if _kind(slot) == "内部":
            _options, source, repair = slot["内部"]
            found.append((name, source, repair))
    return tuple(found)


def bindings(table: dict) -> tuple[str, ...]:
    found = []
    for slot in table.values():
        if _kind(slot) == "绑定":
            question, _options = slot["绑定"]
            found.append(question)
    return tuple(found)


def alias(table: dict, render) -> int:
    """Unordered pairs of distinct assignments that paste the same string.

    The set notation (f, f') with f≠f' counts each collision twice. This
    function counts each collision once.
    """
    if not table:
        raise RuntimeError("空表是未填，不能当成 alias=0")
    if render is None:
        raise RuntimeError("没有合成函数时不能把 alias 当成 0")
    counts: dict[str, int] = {}
    lists = [_options(slot) for slot in table.values()]
    for values in product(*lists):
        text = render(*values)
        counts[text] = counts.get(text, 0) + 1
    return sum(count * (count - 1) // 2 for count in counts.values())


def exposed(table: dict, render) -> bool:
    """Every direct factor has a question, and the pasted value tells them apart."""
    return latent(table) == 0 and alias(table, render) == 0


def audit() -> tuple[dict, ...]:
    """One row per filled target, then one 未填 row per reserved composer."""
    rows = []
    for name, table in FACTORS.items():
        render = RENDER.get(name)
        if render is None:
            rows.append({
                "target": name,
                "status": "已填",
                "latent": latent(table),
                "alias": None,
                "sources": sources(table),
                "exposed": None,
            })
            continue
        rows.append({
            "target": name,
            "status": "已填",
            "latent": latent(table),
            "alias": alias(table, render),
            "sources": sources(table),
            "exposed": exposed(table, render),
        })
    for name in PENDING:
        rows.append({
            "target": name,
            "status": "未填",
            "latent": None,
            "alias": None,
            "sources": (),
            "exposed": None,
        })
    return tuple(rows)


def _composer_agrees(table: dict) -> None:
    """The live moment composer pastes the same strings as the 拆后 fixture."""
    lists = [_options(slot) for slot in table.values()]
    for sign, force, arm, denom in product(*lists):
        got = _compose_moment(
            {"s": sign, "f": force, "a": arm, "d": denom},
            "s", "f", "a", "d",
        )
        if not got.ok or got.matrix != compose(sign, force, arm, denom):
            raise RuntimeError(f"合成串与 _compose_moment 不一致：{sign, force, arm, denom} -> {got}")


def check() -> None:
    after = FACTORS["力矩公式/拆后"]
    before = FACTORS["力矩公式/拆前"]
    if latent(after) != 0 or alias(after, compose) != 0 or not exposed(after, compose):
        raise RuntimeError("拆后应是 latent=0 且 alias=0")
    if bindings(after) != ("ID190", "ID202", "ID208", "ID213"):
        raise RuntimeError(f"拆后绑定的题号不对：{bindings(after)}")
    if sum(1 for _ in product(*(_options(slot) for slot in after.values()))) != 72:
        raise RuntimeError("拆后赋值应有 72 个")
    if latent(before) != 3:
        raise RuntimeError(f"拆前 latent 应为 3，得到 {latent(before)}")
    expected = (
        ("力系数", "力矩名", "独立"),
        ("坐标", "ID184", "独立"),
        ("分母", "常量 L_RT", "独立"),
    )
    if sources(before) != expected:
        raise RuntimeError(f"拆前来源标签不对：{sources(before)}")
    attitude = FACTORS["attitude_target"]
    if latent(attitude) != 1 or sources(attitude)[0][0] != "手性" or sources(attitude)[0][2] != "派生":
        raise RuntimeError(f"手性应是唯一的派生内部因子：{sources(attitude)}")
    if "ID2/ID3/ID4" not in sources(attitude)[0][1] or "ID89/ID90/ID91" not in sources(attitude)[0][1]:
        raise RuntimeError("手性来源漏了风轴或天平轴")
    for name in ("roll_target", "_installation"):
        found = sources(FACTORS[name])
        if latent(FACTORS[name]) != 1 or len(found) != 1 or found[0][2] != "派生":
            raise RuntimeError(f"{name} 应有一个派生手性因子：{found}")
    for name in ("_product", "_alignment", "_mount"):
        if latent(FACTORS[name]) != 0:
            raise RuntimeError(f"{name} 的直接因子里不该有内部项")
        if name in RENDER:
            raise RuntimeError(f"{name} 还没有合成串，不能把 alias 当成 0")
    if exposed(before, compose):
        raise RuntimeError("拆前被记成已经暴露完整")
    _composer_agrees(after)
    filled = [row for row in audit() if row["status"] == "已填"]
    pending = [row for row in audit() if row["status"] == "未填"]
    if any(row["latent"] == 0 and row["status"] == "未填" for row in pending):
        raise RuntimeError("未填目标被当成了 latent=0")
    if len(pending) != len(PENDING) or len(filled) != len(FACTORS):
        raise RuntimeError("审计行没有分开已填和未填")


def _cell(value) -> str:
    return "—" if value is None else str(value)


def main() -> None:
    check()
    print("| 目标 | 状态 | latent | alias | 来源 |", flush=True)
    print("| --- | --- | ---: | ---: | --- |", flush=True)
    for row in audit():
        labels = "，".join(f"{name}←{label}（{repair}）" for name, label, repair in row["sources"])
        print(
            f"| {row['target']} | {row['status']} | {_cell(row['latent'])} | "
            f"{_cell(row['alias'])} | {labels} |",
            flush=True,
        )
    filled = [row for row in audit() if row["status"] == "已填"]
    passed = [row["target"] for row in filled if row["exposed"] is True]
    hidden = [f"{row['target']} latent={row['latent']}" for row in filled if row["latent"]]
    unaliased = [row["target"] for row in filled if row["latent"] == 0 and row["alias"] is None]
    print(
        f"已填 {len(filled)} 条：暴露完整 {len(passed)}（{'，'.join(passed) or '无'}），"
        f"有内部因子 {len(hidden)}（{'，'.join(hidden) or '无'}）。",
        flush=True,
    )
    print(f"latent=0 但 alias 还没算：{'，'.join(unaliased) or '无'}。", flush=True)
    print(f"未填 {len(PENDING)} 条不计入全表判定。", flush=True)


if __name__ == "__main__":
    main()
