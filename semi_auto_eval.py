# -*- coding: utf-8 -*-
"""半自动对照实验：A 组直接回答，B 组生成代码并执行，人工评判对错。

流程：先把 baseline 问答全文（含问题和答案）交给模型学习，再只问 beta 的题目。
beta 的标准答案不进入提示词，只在人工评判时显示。baseline 本身不是考题。

A 组提示词在 prompts/prompt_a.txt，B 组提示词在 prompts/prompt_b.txt。
文件中的 {baseline}、{definitions}、{question} 会在运行时填入。

用法（在 workflow 目录下）：

  1. 复制 .env.example 为 .env，填入 DEEPSEEK_API_KEY
  2. pip install -r requirements.txt

  # 查看将要跑的题目（不调用模型）
  python semi_auto_eval.py --list

  # 只打印提示词，确认 baseline 全文在内、beta 标准答案不在内
  python semi_auto_eval.py --dry-run --only coord:4

  # 正式实验。默认只评 beta 的 9 道题，每题跑完后在终端里评判
  python semi_auto_eval.py

  # 先全部跑完，稍后再评判（结果写到 results/<时间戳>/）
  python semi_auto_eval.py --grade later

  # 只补评已经跑完的记录
  python semi_auto_eval.py --grade-only results/<时间戳>/records.jsonl

默认题目：坐标系变换 Q1–Q7，联合推理 Q1–Q2。
概念辨析，以及联合推理里的文字判断题，默认不跑。

评判时输入：
  c 正确    w 错误    e 执行失败（仅 B 组）    s 跳过    直接回车 = 稍后再评
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from docx import Document

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
ACL_ROOT = ROOT.parent
RESULTS_ROOT = ROOT / "results"

DATASETS = {
    "baseline": {
        "path": ACL_ROOT / "baseline" / "POT" / "betchmark_baseline_QA_POT.docx",
        "note": "侧滑角右滑为正",
    },
    "beta": {
        "path": ACL_ROOT / "侧滑角定义相反" / "betchmark_β&βM_QA.docx",
        "note": "侧滑角左滑为正",
    },
}

SECTION_BY_ID = {
    "concept": "概念理解和辨析",
    "coord": "坐标系变换能力",
    "joint": "联合推理能力",
}

# 这些题有明确的矩阵/数值推导。概念辨析和联合推理 Q3/Q4 是文字判断，默认不做代码化。
DEFAULT_ONLY = "coord:1-7,joint:1-2"

PROMPT_A_PATH = ROOT / "prompts" / "prompt_a.txt"
PROMPT_B_PATH = ROOT / "prompts" / "prompt_b.txt"


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"找不到提示词文件：{path}")
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, fields: dict[str, str]) -> str:
    text = template
    for key, value in fields.items():
        token = "{" + key + "}"
        if token not in text:
            raise ValueError(f"提示词缺少占位符 {token}")
        text = text.replace(token, value)
    return text.strip() + "\n"


@dataclass
class Question:
    dataset: str
    dataset_note: str
    section: str
    section_id: str
    number: int
    qid: str
    item_id: str
    question: str
    reference: str
    definitions: str
    baseline_document: str
    source: str


@dataclass
class RunRecord:
    item_id: str
    dataset: str
    dataset_note: str
    section: str
    section_id: str
    qid: str
    question: str
    reference_answer: str
    model: str
    temperature: float
    group_a: dict = field(default_factory=dict)
    group_b: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "dataset": self.dataset,
            "dataset_note": self.dataset_note,
            "section": self.section,
            "section_id": self.section_id,
            "qid": self.qid,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "model": self.model,
            "temperature": self.temperature,
            "group_a": self.group_a,
            "group_b": self.group_b,
        }


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def match_section(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if len(text) > 40 or text[:1] in {"Q", "A"}:
        return None
    compact = re.sub(r"\s+", "", text)
    if "概念理解" in compact and "辨析" in compact:
        return "概念理解和辨析", "concept"
    if compact.endswith("坐标系变换能力"):
        return "坐标系变换能力", "coord"
    if "联合推理" in compact and "能力" in compact:
        return "联合推理能力", "joint"
    return None


W_T = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
M_T = "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"


def paragraph_to_text(paragraph) -> str:
    """取出段落可见文字。

    个别公式只存在于公式对象里，paragraph.text 是空的。
    若公式对象里已经带有 LaTeX 文本，就用它补上。
    纯公式对象往往和旁边的 LaTeX 段落重复，那种扁平文本不收。
    """
    element = paragraph._p
    words = "".join(node.text or "" for node in element.iter(W_T)).strip()
    if words:
        return words
    math = "".join(node.text or "" for node in element.iter(M_T)).strip()
    if "\\" in math:
        return math
    return ""


def read_document(path: Path) -> str:
    document = Document(str(path))
    lines = [paragraph_to_text(paragraph) for paragraph in document.paragraphs]
    text = "\n".join(line for line in lines if line).strip()
    if not text:
        raise ValueError(f"文档没有读出文字：{path}")
    return text


def parse_qa_docx(path: Path) -> tuple[str, list[dict]]:
    document = Document(str(path))
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph_to_text(paragraph)
        if not text:
            continue
        text = re.sub(r"(?<!^)(?=[QA]\d+\s*[：:])", "\n", text)
        lines.extend(part.strip() for part in text.splitlines() if part.strip())

    task_at = next((i for i, line in enumerate(lines) if "【任务】" in line), None)
    if task_at is None:
        raise ValueError(f"未在文档中找到【任务】标记：{path}")

    definitions = "\n".join(lines[:task_at]).strip()
    section_name = "未分类"
    section_id = "unknown"
    current: dict | None = None
    mode: str | None = None
    items: list[dict] = []

    def flush() -> None:
        nonlocal current, mode
        if current and current["question"]:
            current["question"] = current["question"].strip()
            current["reference"] = current["reference"].strip()
            items.append(current)
        current = None
        mode = None

    for line in lines[task_at + 1 :]:
        section = match_section(line)
        if section:
            flush()
            section_name, section_id = section
            continue

        marker = re.match(r"^([QA])(\d+)\s*[：:]\s*(.*)$", line)
        if marker and marker.group(1) == "Q":
            flush()
            number = int(marker.group(2))
            current = {
                "section": section_name,
                "section_id": section_id,
                "number": number,
                "qid": f"Q{number}",
                "question": marker.group(3).strip(),
                "reference": "",
            }
            mode = "q"
            continue
        if marker and marker.group(1) == "A" and current is not None:
            mode = "a"
            rest = marker.group(3).strip()
            current["reference"] = rest
            continue
        if current is None:
            continue
        if mode == "q":
            current["question"] = (current["question"] + "\n" + line).strip()
        elif mode == "a":
            current["reference"] = (current["reference"] + "\n" + line).strip()

    flush()
    return definitions, items


def load_questions(dataset_keys: list[str]) -> list[Question]:
    baseline_path = Path(DATASETS["baseline"]["path"])
    if not baseline_path.is_file():
        raise FileNotFoundError(f"找不到 baseline 文档：{baseline_path}")
    baseline_document = read_document(baseline_path)
    questions: list[Question] = []
    for key in dataset_keys:
        spec = DATASETS[key]
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"找不到测试文档：{path}")
        definitions, items = parse_qa_docx(path)
        if not definitions:
            raise ValueError(f"文档里没有解析到定义：{path}")
        for item in items:
            item_id = f"{key}-{item['section_id']}-{item['qid']}"
            questions.append(
                Question(
                    dataset=key,
                    dataset_note=spec["note"],
                    section=item["section"],
                    section_id=item["section_id"],
                    number=item["number"],
                    qid=item["qid"],
                    item_id=item_id,
                    question=item["question"],
                    reference=item["reference"],
                    definitions=definitions,
                    baseline_document=baseline_document,
                    source=str(path),
                )
            )
    return questions


def parse_only(spec: str) -> set[tuple[str, int]] | None:
    """返回允许的 (section_id, number)。'all' 表示不筛选。"""
    text = spec.strip().lower()
    if text in {"", "all", "*"}:
        return None
    allowed: set[tuple[str, int]] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            if part not in SECTION_BY_ID:
                raise ValueError(f"未知题组：{part}。可用 concept / coord / joint / all")
            allowed.add((part, -1))  # -1 表示该组全部题号
            continue
        section_id, numbers = part.split(":", 1)
        section_id = section_id.strip()
        if section_id not in SECTION_BY_ID:
            raise ValueError(f"未知题组：{section_id}。可用 concept / coord / joint")
        for piece in numbers.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece:
                start_s, end_s = piece.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    raise ValueError(f"题号范围无效：{piece}")
                for number in range(start, end + 1):
                    allowed.add((section_id, number))
            else:
                allowed.add((section_id, int(piece)))
    return allowed


def filter_questions(questions: list[Question], only: set[tuple[str, int]] | None) -> list[Question]:
    if only is None:
        return questions
    chosen: list[Question] = []
    for question in questions:
        if (question.section_id, -1) in only or (question.section_id, question.number) in only:
            chosen.append(question)
    return chosen


def build_prompts(question: Question) -> tuple[str, str]:
    fields = {
        "baseline": question.baseline_document,
        "definitions": question.definitions,
        "question": question.question,
    }
    direct = fill_prompt(load_prompt(PROMPT_A_PATH), fields)
    code = fill_prompt(load_prompt(PROMPT_B_PATH), fields)
    return direct, code


def reference_leaked(prompt: str, reference: str, baseline_document: str) -> bool:
    """beta 标准答案不应出现在提示词里。baseline 全文是故意给模型看的，先摘掉再检查。"""
    snippet = re.sub(r"\s+", "", reference)
    if len(snippet) < 24:
        return False
    remainder = prompt.replace(baseline_document, "")
    return snippet[:48] in re.sub(r"\s+", "", remainder)


def extract_code(response: str) -> str | None:
    if not response or not response.strip():
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    stripped = response.strip()
    if any(token in stripped for token in ("print(", "ans =", "ans=", "Matrix(")) or stripped.startswith(("import ", "from ", "def ")):
        return stripped
    return None


def analyze_code(code: str) -> dict:
    trig_call = bool(re.search(r"\b(?:sin|cos|tan|asin|acos|atan|arcsin|arccos|arctan2?)\s*\(", code))
    uses_sympy = "sympy" in code or bool(re.search(r"\bMatrix\s*\(", code))
    uses_numpy = "numpy" in code or "np." in code
    return {
        "uses_trig_call": trig_call,
        "uses_sympy": uses_sympy,
        "uses_numpy": uses_numpy,
        "possible_hardcode": not trig_call,
    }


def run_code(code: str, timeout: int) -> tuple[str, str, int]:
    workdir = tempfile.mkdtemp(prefix="acl_eval_")
    path = os.path.join(workdir, "answer.py")
    env = os.environ.copy()
    for key in list(env):
        if "KEY" in key.upper() or "TOKEN" in key.upper() or "SECRET" in key.upper():
            env.pop(key, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        # 论文要求最终结果存放在变量 ans。若模型只赋值、不 print，这里补打出来。
        trailer = (
            "\n\ntry:\n"
            "    ans\n"
            "except NameError:\n"
            "    pass\n"
            "else:\n"
            "    print('ans =')\n"
            "    import pprint as _acl_pprint\n"
            "    _acl_pprint.pprint(ans)\n"
        )
        Path(path).write_text(code + trailer, encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout, completed.stderr, completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        stderr = (stderr + f"\n[timeout] 代码执行超过 {timeout} 秒").strip()
        return stdout, stderr, -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def resolve_api_config() -> tuple[str, str, str]:
    """按 DeepSeek 官方的 OpenAI 兼容格式读取配置。

    官方参数：base_url=https://api.deepseek.com，model=deepseek-flash，
    密钥环境变量 DEEPSEEK_API_KEY。OPENAI_* 仍可作为别名。
    """
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
        or DEEPSEEK_BASE_URL
    )
    model = (
        os.getenv("DEEPSEEK_MODEL", "").strip()
        or os.getenv("OPENAI_MODEL", "").strip()
        or "deepseek-flash"
    )
    # 走 OpenRouter 时优先用 OPENROUTER_API_KEY，避免误用 DeepSeek key 导致 401。
    if "openrouter" in base_url.lower():
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or api_key
    # 千问直连百炼，使用 QWEN_API_KEY，不走 OpenRouter。
    if "dashscope" in base_url.lower() or "maas.aliyuncs.com" in base_url.lower():
        api_key = os.getenv("QWEN_API_KEY", "").strip() or os.getenv("DASHSCOPE_API_KEY", "").strip() or api_key
    return api_key, base_url, model


def is_deepseek(base_url: str, model: str) -> bool:
    return "deepseek" in f"{base_url} {model}".lower()


def is_qwen(base_url: str) -> bool:
    text = base_url.lower()
    return "dashscope" in text or "maas.aliyuncs.com" in text


def message_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                text = getattr(part, "text", None)
                parts.append(str(text if text is not None else part))
        return "".join(parts)
    return str(content)


def call_model(
    client,
    model: str,
    prompt: str,
    temperature: float,
    thinking: str,
    captured: dict | None = None,
    max_tokens: int | None = None,
    stop: list[str] | None = None,
    seed: int | None = None,
    reasoning_effort: str | None = None,
    messages: list[dict] | None = None,
) -> tuple[str, str]:
    """调用对话补全。DeepSeek 思考模式默认开启，且会忽略 temperature。

    实验需要 temperature=0 时，必须显式传入 thinking.type=disabled。
    见 https://api-docs.deepseek.com/zh-cn/guides/thinking_mode
    max_tokens 含思考内容。思考开启时不要把上限设成只够最终答案的长度。
    官方 V4 接口接受 seed 但不使用它。captured 只接收本次响应的模型名、编号、结束原因和用量。
    未提供 messages 时，每次只发送当前用户提示。提供 messages 时，按该会话逐轮发送。
    reasoning_content 只随本次响应返回，不写进 messages，也不交给下一轮。
    reasoning_effort 传入时覆盖环境变量，用于截断后缩短思考再试一次。
    """
    base_url = str(getattr(client, "base_url", "") or "")
    if messages is None:
        payload = [{"role": "user", "content": prompt}]
    else:
        payload = []
        for item in messages:
            role = item.get("role")
            content = item.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                raise RuntimeError("会话消息只能有 system、user、assistant，且内容是字符串。")
            payload.append({"role": role, "content": content})
    kwargs: dict = {
        "model": model,
        "messages": payload,
        "stream": False,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if stop:
        kwargs["stop"] = stop
    if seed is not None:
        kwargs["seed"] = seed
    if is_deepseek(base_url, model):
        if thinking == "enabled":
            kwargs["reasoning_effort"] = (
                reasoning_effort or os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
            )
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["temperature"] = temperature
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    elif is_qwen(base_url):
        # Qwen3.8 不传 reasoning_effort 时默认 xhigh，thinking_budget=131072。
        # max_tokens 只限制回答，不限制思维链。非流式超过约 300 秒会被服务端截断。
        # qwen3.8-2.4t-a95b 拒绝 enable_thinking=false，也拒绝 reasoning_effort=none。
        # low 映射 4096，是这个模型允许的最短思考。
        kwargs["temperature"] = temperature
        if "2.4t" in model.lower():
            kwargs["reasoning_effort"] = reasoning_effort or "low"
        elif thinking == "enabled":
            kwargs["reasoning_effort"] = reasoning_effort or os.getenv("QWEN_REASONING_EFFORT", "medium")
            kwargs["extra_body"] = {"enable_thinking": True}
        else:
            kwargs["extra_body"] = {"enable_thinking": False}
    else:
        kwargs["temperature"] = temperature
        if reasoning_effort and thinking == "enabled":
            kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    if captured is not None:
        captured["response_model"] = getattr(response, "model", "") or ""
        captured["response_id"] = getattr(response, "id", "") or ""
        captured["system_fingerprint"] = getattr(response, "system_fingerprint", None)
        captured["usage"] = getattr(response, "usage", None)
        captured["finish_reason"] = getattr(choice, "finish_reason", None)
        captured["reasoning_effort"] = kwargs.get("reasoning_effort") or ""
    message = choice.message
    content = message_text(message.content)
    reasoning = message_text(getattr(message, "reasoning_content", None))
    return content, reasoning


def make_client(timeout: float = 180.0):
    api_key, base_url, _model = resolve_api_config()
    if not api_key:
        raise SystemExit(
            "未设置 DEEPSEEK_API_KEY。请在 workflow/.env 中填写，或在当前终端设置该环境变量。"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("未安装 openai。请先运行：pip install -r requirements.txt") from exc
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        default_headers={"Accept-Encoding": "gzip, deflate"},
    )


def check_exec_deps() -> None:
    missing = []
    for name in ("numpy", "sympy"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print("警告：当前 Python 缺少 " + "、".join(missing) + "。B 组代码很可能执行失败。")
        print("请运行：pip install -r requirements.txt")


def normalize_verdict(raw: str, allow_exec_fail: bool) -> str | None:
    text = raw.strip().lower()
    if text == "":
        return None
    mapping = {
        "c": "correct",
        "正确": "correct",
        "对": "correct",
        "y": "correct",
        "1": "correct",
        "w": "wrong",
        "错误": "wrong",
        "错": "wrong",
        "n": "wrong",
        "0": "wrong",
        "s": "skip",
        "跳过": "skip",
        "e": "exec_fail",
        "失败": "exec_fail",
        "执行失败": "exec_fail",
    }
    verdict = mapping.get(text)
    if verdict is None:
        print("无法识别，请输入 c / w / e / s，或直接回车跳过本次评判。")
        return "INVALID"
    if verdict == "exec_fail" and not allow_exec_fail:
        print("A 组没有“执行失败”。请输入 c / w / s。")
        return "INVALID"
    return verdict


def ask_verdict(label: str, allow_exec_fail: bool) -> str | None:
    hint = "c=正确 / w=错误 / e=执行失败 / s=跳过" if allow_exec_fail else "c=正确 / w=错误 / s=跳过"
    while True:
        try:
            raw = input(f"{label} [{hint}，回车=稍后评]：")
        except EOFError:
            print()
            return None
        verdict = normalize_verdict(raw, allow_exec_fail)
        if verdict != "INVALID":
            return verdict


def verdict_zh(value: str | None) -> str:
    return {
        "correct": "正确",
        "wrong": "错误",
        "exec_fail": "执行失败",
        "skip": "跳过",
        None: "未评",
    }.get(value, "未评")


def print_block(title: str, text: str) -> None:
    print(f"\n--- {title} ---")
    print(text if text else "（空）")


def grade_record(record: RunRecord) -> None:
    print("\n--- 请人工判断。评判对象是答案本身，代码写得好看但输出错了算错 ---")
    record.group_a["verdict"] = ask_verdict("A 组", allow_exec_fail=False)
    record.group_b["verdict"] = ask_verdict("B 组", allow_exec_fail=True)
    try:
        note = input("备注（可空）：").strip()
    except EOFError:
        note = ""
        print()
    record.group_a["note"] = note
    record.group_b["note"] = note


def execute_question(
    question: Question,
    client,
    model: str,
    temperature: float,
    timeout: int,
    groups: set[str],
    code_dir: Path,
    thinking: str,
) -> RunRecord:
    prompt_direct, prompt_code = build_prompts(question)
    record = RunRecord(
        item_id=question.item_id,
        dataset=question.dataset,
        dataset_note=question.dataset_note,
        section=question.section,
        section_id=question.section_id,
        qid=question.qid,
        question=question.question,
        reference_answer=question.reference,
        model=model,
        temperature=temperature,
    )

    print("\n" + "=" * 72)
    print(f"[{question.dataset} | {question.dataset_note}] {question.section} {question.qid}")
    print("=" * 72)
    print_block("问题", question.question)

    if "A" in groups:
        print("\n正在请求 A 组（直接回答）...")
        group_a = {
            "prompt": prompt_direct,
            "answer": "",
            "reasoning": "",
            "error": "",
            "verdict": None,
            "note": "",
        }
        try:
            group_a["answer"], group_a["reasoning"] = call_model(
                client, model, prompt_direct, temperature, thinking
            )
        except Exception as exc:  # noqa: BLE001 — 单题失败不应中断整批
            group_a["error"] = f"{type(exc).__name__}: {exc}"
        record.group_a = group_a
        if group_a["error"]:
            print_block("A 组调用失败", group_a["error"])
        else:
            print_block("A 组：直接回答", group_a["answer"])

    if "B" in groups:
        print("\n正在请求 B 组（生成代码）...")
        group_b = {
            "prompt": prompt_code,
            "raw_response": "",
            "code": "",
            "code_file": "",
            "stdout": "",
            "stderr": "",
            "returncode": None,
            "exec_status": "",
            "uses_trig_call": False,
            "uses_sympy": False,
            "uses_numpy": False,
            "possible_hardcode": False,
            "reasoning": "",
            "error": "",
            "verdict": None,
            "note": "",
        }
        try:
            group_b["raw_response"], group_b["reasoning"] = call_model(
                client, model, prompt_code, temperature, thinking
            )
        except Exception as exc:  # noqa: BLE001
            group_b["error"] = f"{type(exc).__name__}: {exc}"
            group_b["exec_status"] = "api_fail"

        if group_b["error"]:
            print_block("B 组调用失败", group_b["error"])
        else:
            code = extract_code(group_b["raw_response"])
            if not code:
                group_b["exec_status"] = "no_code"
                print_block("B 组未解析出代码。原始回复", group_b["raw_response"])
            else:
                group_b["code"] = code
                flags = analyze_code(code)
                group_b.update(flags)
                code_dir.mkdir(parents=True, exist_ok=True)
                code_path = code_dir / f"{question.item_id}.py"
                code_path.write_text(code, encoding="utf-8")
                group_b["code_file"] = str(code_path)
                print_block("B 组：生成的代码", code)
                stdout, stderr, returncode = run_code(code, timeout)
                group_b["stdout"] = stdout
                group_b["stderr"] = stderr
                group_b["returncode"] = returncode
                group_b["exec_status"] = "ok" if returncode == 0 else "fail"
                print_block("B 组：执行 STDOUT（以此作为 B 组答案）", stdout)
                if stderr.strip():
                    print_block("B 组：STDERR", stderr)
                print(f"返回码：{returncode}")
                hint = "调用了 sin/cos" if flags["uses_trig_call"] else "没有调用 sin/cos，可能把答案写死了"
                print(f"代码检查：{hint}；sympy={flags['uses_sympy']}；numpy={flags['uses_numpy']}")
        record.group_b = group_b

    print_block("参考答案（只给你对照，没有提供给模型）", question.reference)
    return record


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def oneline(text: str, limit: int = 180) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def write_grades_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "题目",
        "数据集",
        "定义说明",
        "A组答案",
        "A组对错",
        "B组输出",
        "B组对错",
        "执行状态",
        "是否调用三角函数",
        "可能写死答案",
        "B组代码文件",
        "备注",
        "参考答案",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            group_a = record.get("group_a") or {}
            group_b = record.get("group_b") or {}
            note = group_a.get("note") or group_b.get("note") or ""
            writer.writerow(
                {
                    "题目": f"{record.get('section', '')} {record.get('qid', '')}",
                    "数据集": record.get("dataset", ""),
                    "定义说明": record.get("dataset_note", ""),
                    "A组答案": group_a.get("answer") or group_a.get("error") or "",
                    "A组对错": verdict_zh(group_a.get("verdict")),
                    "B组输出": group_b.get("stdout") or group_b.get("error") or "",
                    "B组对错": verdict_zh(group_b.get("verdict")),
                    "执行状态": group_b.get("exec_status") or "",
                    "是否调用三角函数": "是" if group_b.get("uses_trig_call") else "否",
                    "可能写死答案": "是" if group_b.get("possible_hardcode") else "否",
                    "B组代码文件": group_b.get("code_file") or "",
                    "备注": note,
                    "参考答案": record.get("reference_answer") or "",
                }
            )


def write_report(path: Path, records: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 半自动对照实验记录",
        "",
        f"- 模型：{meta.get('model', '')}",
        f"- 温度：{meta.get('temperature', '')}",
        f"- 时间：{meta.get('created_at', '')}",
        "",
        "A 组看文字答案。B 组以代码执行后的 STDOUT 为准；跑不通记为执行失败，也算错。",
        "「可能写死答案」只是提示：代码若没有调用 sin/cos，就值得打开代码看一眼。",
        "",
    ]
    for record in records:
        group_a = record.get("group_a") or {}
        group_b = record.get("group_b") or {}
        lines.append(f"## {record.get('item_id')}")
        lines.append("")
        lines.append(f"- 数据集：{record.get('dataset')}（{record.get('dataset_note')}）")
        lines.append(f"- A 组：{verdict_zh(group_a.get('verdict'))}")
        lines.append(f"- B 组：{verdict_zh(group_b.get('verdict'))}；执行状态={group_b.get('exec_status') or '无'}")
        if group_b:
            lines.append(
                f"- 代码检查：三角函数调用={'是' if group_b.get('uses_trig_call') else '否'}；"
                f"可能写死={'是' if group_b.get('possible_hardcode') else '否'}"
            )
        note = group_a.get("note") or group_b.get("note") or ""
        if note:
            lines.append(f"- 备注：{note}")
        lines.append("")
        lines.append("### 问题")
        lines.append("")
        lines.append(record.get("question") or "")
        lines.append("")
        lines.append("### A 组答案")
        lines.append("")
        lines.append(group_a.get("answer") or group_a.get("error") or "（未跑）")
        lines.append("")
        lines.append("### B 组代码")
        lines.append("")
        lines.append("```python")
        lines.append(group_b.get("code") or "（无代码）")
        lines.append("```")
        lines.append("")
        lines.append("### B 组 STDOUT")
        lines.append("")
        lines.append("```")
        lines.append(group_b.get("stdout") or "（无输出）")
        lines.append("```")
        if (group_b.get("stderr") or "").strip():
            lines.append("")
            lines.append("### B 组 STDERR")
            lines.append("")
            lines.append("```")
            lines.append(group_b["stderr"])
            lines.append("```")
        lines.append("")
        lines.append("### 参考答案（未提供给模型）")
        lines.append("")
        lines.append(record.get("reference_answer") or "（文档中该题没有抽出文字答案）")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize(records: list[dict]) -> str:
    def count(group: str, verdict: str) -> int:
        return sum(1 for record in records if (record.get(group) or {}).get("verdict") == verdict)

    graded_a = sum(1 for record in records if (record.get("group_a") or {}).get("verdict"))
    graded_b = sum(1 for record in records if (record.get("group_b") or {}).get("verdict"))
    exec_fail = sum(1 for record in records if (record.get("group_b") or {}).get("exec_status") in {"fail", "no_code", "api_fail"})
    hardcode = sum(1 for record in records if (record.get("group_b") or {}).get("possible_hardcode"))
    ran_b = sum(1 for record in records if record.get("group_b"))
    return (
        f"共 {len(records)} 题。\n"
        f"A 组已评 {graded_a}：正确 {count('group_a', 'correct')}，错误 {count('group_a', 'wrong')}，跳过 {count('group_a', 'skip')}。\n"
        f"B 组已评 {graded_b}：正确 {count('group_b', 'correct')}，错误 {count('group_b', 'wrong')}，"
        f"执行失败（评判） {count('group_b', 'exec_fail')}，跳过 {count('group_b', 'skip')}。\n"
        f"B 组执行未成功 {exec_fail}/{ran_b}；可能写死答案 {hardcode}/{ran_b}。"
    )


def save_bundle(out_dir: Path, records: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "records.jsonl", records)
    write_grades_csv(out_dir / "grades.csv", records)
    write_report(out_dir / "report.md", records, meta)


def print_question_list(questions: list[Question]) -> None:
    print(f"共 {len(questions)} 题\n")
    for question in questions:
        ref = oneline(question.reference, 60) or "（无文字参考答案）"
        print(f"{question.item_id}  {question.dataset_note}")
        print(f"  问：{oneline(question.question, 100)}")
        print(f"  参考：{ref}")


def dry_run(questions: list[Question]) -> int:
    leaked = 0
    for question in questions:
        prompt_direct, prompt_code = build_prompts(question)
        direct_leak = reference_leaked(prompt_direct, question.reference, question.baseline_document)
        code_leak = reference_leaked(prompt_code, question.reference, question.baseline_document)
        leaked += int(direct_leak or code_leak)
        has_baseline = question.baseline_document in prompt_direct and "右滑为正" in question.baseline_document
        print("=" * 72)
        print(question.item_id, question.dataset_note)
        print(f"baseline 全文已放入提示词：{has_baseline}（{len(question.baseline_document)} 字）")
        print(f"beta 标准答案是否出现在 baseline 之外：A={direct_leak} B={code_leak}")
        print(f"新定义长度={len(question.definitions)} 问题长度={len(question.question)} 参考答案长度={len(question.reference)}")
        print_block("问题", question.question)
        print_block("A 组提示词末尾", prompt_direct[-500:])
        print_block("B 组要求", prompt_code.split("【要求】", 1)[-1])
    print(f"\n检查完成：{len(questions)} 题，beta 标准答案疑似漏进提示词 {leaked} 题。")
    return 1 if leaked else 0


def grade_existing(path: Path) -> None:
    records = read_jsonl(path)
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    print(f"读取 {len(records)} 条记录：{path}")
    for index, record in enumerate(records, start=1):
        group_a = record.setdefault("group_a", {})
        group_b = record.setdefault("group_b", {})
        if group_a.get("verdict") and group_b.get("verdict"):
            continue
        print("\n" + "=" * 72)
        print(f"[{index}/{len(records)}] {record.get('item_id')}  {record.get('dataset_note')}")
        print_block("问题", record.get("question") or "")
        print_block("A 组答案", group_a.get("answer") or group_a.get("error") or "（无）")
        print_block("B 组 STDOUT", group_b.get("stdout") or "（无）")
        if group_b.get("exec_status") and group_b.get("exec_status") != "ok":
            print(f"B 组执行状态：{group_b.get('exec_status')}")
            if group_b.get("stderr"):
                print_block("B 组 STDERR", group_b["stderr"])
        if group_b.get("possible_hardcode"):
            print("代码检查：没有调用 sin/cos，建议看代码是否写死了答案。")
            if group_b.get("code_file"):
                print(f"代码文件：{group_b['code_file']}")
        print_block("参考答案（未提供给模型）", record.get("reference_answer") or "")
        wrapper = RunRecord(
            item_id=record.get("item_id", ""),
            dataset=record.get("dataset", ""),
            dataset_note=record.get("dataset_note", ""),
            section=record.get("section", ""),
            section_id=record.get("section_id", ""),
            qid=record.get("qid", ""),
            question=record.get("question", ""),
            reference_answer=record.get("reference_answer", ""),
            model=record.get("model", ""),
            temperature=record.get("temperature", 0),
            group_a=group_a,
            group_b=group_b,
        )
        grade_record(wrapper)
        record["group_a"] = wrapper.group_a
        record["group_b"] = wrapper.group_b
        save_bundle(path.parent, records, meta)
    print("\n" + summarize(records))
    print(f"已写回：{path.parent}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A 组直接回答 vs B 组代码执行，人工评判。")
    parser.add_argument(
        "--dataset",
        choices=["beta"],
        default="beta",
        help="只评 beta（左滑为正）。baseline 全文只作为学习材料",
    )
    parser.add_argument("--only", default=DEFAULT_ONLY, help="题目筛选，例如 coord:1-7,joint:1-2；all 为全部")
    parser.add_argument("--groups", default="AB", help="跑哪些组：A、B 或 AB")
    parser.add_argument("--grade", choices=["now", "later"], default="now", help="now=每题立刻评判，later=先跑完再评")
    parser.add_argument("--grade-only", metavar="JSONL", help="只对已有 records.jsonl 做人工评判")
    parser.add_argument("--list", action="store_true", help="列出题目后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印提示词并检查参考答案是否泄漏")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题，0 表示不限制")
    parser.add_argument("--timeout", type=int, default=30, help="单次代码执行超时（秒）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="", help="默认 deepseek-flash，或读取 DEEPSEEK_MODEL")
    parser.add_argument("--base-url", default="", help="默认 https://api.deepseek.com")
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="",
        help="DeepSeek 思考模式。默认 disabled，这样 temperature=0 才会生效",
    )
    parser.add_argument("--out", default="", help="结果目录，默认 results/<时间戳>")
    return parser


def main() -> int:
    load_env_file(ROOT / ".env")
    args = build_parser().parse_args()

    if args.grade_only:
        grade_existing(Path(args.grade_only))
        return 0

    if args.base_url:
        os.environ["DEEPSEEK_BASE_URL"] = args.base_url
    _api_key, base_url, env_model = resolve_api_config()
    model = args.model.strip() or env_model
    thinking = args.thinking or os.getenv("DEEPSEEK_THINKING", "").strip() or "disabled"

    dataset_keys = ["baseline", "beta"] if args.dataset == "both" else [args.dataset]
    questions = filter_questions(load_questions(dataset_keys), parse_only(args.only))
    if args.limit > 0:
        questions = questions[: args.limit]
    if not questions:
        print("没有匹配的题目。可用 --list 查看。")
        return 1

    if args.list:
        print_question_list(questions)
        return 0
    if args.dry_run:
        return dry_run(questions)

    if thinking not in {"enabled", "disabled"}:
        raise SystemExit("DEEPSEEK_THINKING 只能是 enabled 或 disabled。")

    groups = {ch for ch in args.groups.upper() if ch in {"A", "B"}}
    if not groups:
        raise SystemExit("--groups 需要包含 A 或 B。")

    check_exec_deps()
    client = make_client()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else RESULTS_ROOT / stamp
    code_dir = out_dir / "code"
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "base_url": base_url,
        "thinking": thinking,
        "temperature": args.temperature,
        "datasets": dataset_keys,
        "only": args.only,
        "groups": "".join(sorted(groups)),
        "sources": {
            "baseline": str(DATASETS["baseline"]["path"]),
            "beta": str(DATASETS["beta"]["path"]),
        },
    }
    records: list[dict] = []
    print(f"模型：{model}    接口：{base_url}")
    print(f"思考模式：{thinking}    温度：{args.temperature}    题目：{len(questions)}")
    if is_deepseek(base_url, model) and thinking == "enabled":
        print("DeepSeek 思考模式会忽略 temperature，本次温度设置不会生效。")
    print(f"结果目录：{out_dir}")
    print("baseline 问答全文会交给模型。beta 标准答案只在评判时显示。B 组以 STDOUT 为准。")

    for index, question in enumerate(questions, start=1):
        print(f"\n######## {index}/{len(questions)} ########")
        record = execute_question(
            question,
            client,
            model,
            args.temperature,
            args.timeout,
            groups,
            code_dir,
            thinking,
        )
        records.append(record.to_dict())
        save_bundle(out_dir, records, meta)
        if args.grade == "now":
            grade_record(record)
            records[-1] = record.to_dict()
            save_bundle(out_dir, records, meta)

    print("\n" + summarize(records))
    print(f"记录：{out_dir / 'records.jsonl'}")
    print(f"评判表：{out_dir / 'grades.csv'}")
    print(f"阅读版：{out_dir / 'report.md'}")
    if args.grade == "later":
        print(f"稍后评判：python semi_auto_eval.py --grade-only \"{out_dir / 'records.jsonl'}\"")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已中断。已经完成的题目保存在结果目录中。")
        raise SystemExit(130)
