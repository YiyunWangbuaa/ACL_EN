# -*- coding: utf-8 -*-
"""Semi-automatic comparison: group A answers directly, group B writes code that is executed, and a person marks each result.

The model first sees the full baseline question-and-answer text, then only the beta questions.
The beta gold answers stay out of the prompt and appear only when a person grades. The baseline itself is not a test item.

Group A uses prompts/prompt_a.txt. Group B uses prompts/prompt_b.txt.
The placeholders {baseline}, {definitions}, and {question} are filled at runtime.

Usage, from the workflow directory:

  1. Copy .env.example to .env and set DEEPSEEK_API_KEY
  2. pip install -r requirements.txt

  # List the items that would run, without calling the model
  python semi_auto_eval.py --list

  # Print the prompts and check that the baseline text is included and the beta gold is not
  python semi_auto_eval.py --dry-run --only coord:4

  # Formal run. By default the 9 beta items, graded in the terminal after each one
  python semi_auto_eval.py

  # Run everything first and grade later. Results go to results/<timestamp>/
  python semi_auto_eval.py --grade later

  # Grade records that have already been run
  python semi_auto_eval.py --grade-only results/<timestamp>/records.jsonl

Default items: coordinate transforms Q1-Q7 and joint reasoning Q1-Q2.
Concept items, and the prose judgment items in joint reasoning, are skipped by default.

Grading keys:
  c correct    w wrong    e execution failed (group B only)    s skip    Enter = grade later
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


def _beta_workbook() -> Path:
    """The beta workbook sits in a sibling folder. Locate it by its file name."""
    found = sorted(ACL_ROOT.glob("*/betchmark_β&βM_QA.docx"))
    if len(found) == 1:
        return found[0]
    return ACL_ROOT / "beta" / "betchmark_β&βM_QA.docx"


DATASETS = {
    "baseline": {
        "path": ACL_ROOT / "baseline" / "POT" / "betchmark_baseline_QA_POT.docx",
        "note": "sideslip positive to the right",
    },
    "beta": {
        "path": _beta_workbook(),
        "note": "sideslip positive to the left",
    },
}

SECTION_BY_ID = {
    "concept": "Concept understanding and discrimination",
    "coord": "Coordinate-transform ability",
    "joint": "Joint reasoning",
}

# These items have an explicit matrix or numeric derivation. Concept items and joint Q3/Q4 are prose judgments and are not turned into code by default.
DEFAULT_ONLY = "coord:1-7,joint:1-2"

PROMPT_A_PATH = ROOT / "prompts" / "prompt_a.txt"
PROMPT_B_PATH = ROOT / "prompts" / "prompt_b.txt"


def load_prompt(path: Path) -> str:
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def fill_prompt(template: str, fields: dict[str, str]) -> str:
    text = template
    for key, value in fields.items():
        token = "{" + key + "}"
        if token not in text:
            raise ValueError(f"Prompt is missing placeholder {token}")
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
    compact = re.sub(r"\s+", "", text).lower()
    if "conceptunderstandinganddiscrimination" in compact:
        return "Concept understanding and discrimination", "concept"
    if compact.endswith("coordinate-transformability"):
        return "Coordinate-transform ability", "coord"
    if "jointreasoning" in compact:
        return "Joint reasoning", "joint"
    return None


W_T = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
M_T = "{http://schemas.openxmlformats.org/officeDocument/2006/math}t"


def paragraph_to_text(paragraph) -> str:
    """Visible text of one paragraph.

    Some formulas exist only in the formula object, so paragraph.text is empty.
    If that object already carries LaTeX, use it. A formula object that merely
    repeats the neighboring LaTeX paragraph is dropped.
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
        raise ValueError(f"No text could be read from the document: {path}")
    return text


def parse_qa_docx(path: Path) -> tuple[str, list[dict]]:
    document = Document(str(path))
    lines: list[str] = []
    for paragraph in document.paragraphs:
        text = paragraph_to_text(paragraph)
        if not text:
            continue
        text = re.sub(r"(?<!^)(?=[QA]\d+\s*:)", "\n", text)
        lines.extend(part.strip() for part in text.splitlines() if part.strip())

    task_at = next((i for i, line in enumerate(lines) if re.search(r"\[Task\]|\bTask\b", line)), None)
    if task_at is None:
        raise ValueError(f"Task marker not found in the document: {path}")

    definitions = "\n".join(lines[:task_at]).strip()
    section_name = "unclassified"
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

        marker = re.match(r"^([QA])(\d+)\s*:\s*(.*)$", line)
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
        raise FileNotFoundError(f"Baseline document not found: {baseline_path}")
    baseline_document = read_document(baseline_path)
    questions: list[Question] = []
    for key in dataset_keys:
        spec = DATASETS[key]
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Test document not found: {path}")
        definitions, items = parse_qa_docx(path)
        if not definitions:
            raise ValueError(f"No definitions were parsed from the document: {path}")
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
    """Allowed (section_id, number) pairs. 'all' means no filter."""
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
                raise ValueError(f"Unknown section: {part}. Use concept / coord / joint / all")
            allowed.add((part, -1))  # -1 means every item in that section
            continue
        section_id, numbers = part.split(":", 1)
        section_id = section_id.strip()
        if section_id not in SECTION_BY_ID:
            raise ValueError(f"Unknown section: {section_id}. Use concept / coord / joint")
        for piece in numbers.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "-" in piece:
                start_s, end_s = piece.split("-", 1)
                start, end = int(start_s), int(end_s)
                if end < start:
                    raise ValueError(f"Invalid item range: {piece}")
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
    """The beta gold answer should not appear in the prompt. The baseline text is shown to the model on purpose, so remove it before checking."""
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
        # The paper stores the final result in ans. If the model assigns it and does not print, print it here.
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
        stderr = (stderr + f"\n[timeout] code execution exceeded {timeout} seconds").strip()
        return stdout, stderr, -1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def resolve_api_config() -> tuple[str, str, str]:
    """Read the config in DeepSeek's official OpenAI-compatible form.

    Official parameters: base_url=https://api.deepseek.com, model=deepseek-flash,
    and the key in DEEPSEEK_API_KEY. OPENAI_* still works as an alias.
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
    # On OpenRouter, prefer OPENROUTER_API_KEY so a DeepSeek key is not sent and rejected with 401.
    if "openrouter" in base_url.lower():
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip() or api_key
    # Qwen goes to DashScope directly with QWEN_API_KEY, not through OpenRouter.
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
    """Call chat completions. DeepSeek thinking mode is on by default and ignores temperature.

    When the experiment needs temperature=0, pass thinking.type=disabled explicitly.
    See https://api-docs.deepseek.com/guides/thinking_mode
    max_tokens includes the chain of thought. With thinking on, do not set the ceiling
    to a length that only fits the final answer.
    The official V4 API accepts seed and does not use it. captured receives only this
    response's model name, id, finish reason, and usage.
    Without messages, each call sends only the current user prompt. With messages,
    the call sends that conversation turn by turn.
    reasoning_content comes back with this response only. It is not written into
    messages and is not sent on the next turn.
    When reasoning_effort is passed, it overrides the environment variable so a
    truncated reply can be retried with a shorter chain of thought.
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
                raise RuntimeError("Session messages may only use system, user, and assistant, and the content must be a string.")
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
        # Qwen3.8 defaults to reasoning_effort xhigh and thinking_budget=131072 when reasoning_effort is omitted.
        # max_tokens limits the answer only, not the chain of thought. A non-streaming call past about 300 seconds is cut off by the server.
        # qwen3.8-2.4t-a95b rejects enable_thinking=false and also rejects reasoning_effort=none.
        # low maps to 4096, the shortest chain of thought this model allows.
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
            "DEEPSEEK_API_KEY is not set. Put it in workflow/.env, or set the variable in this terminal."
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("openai is not installed. Run: pip install -r requirements.txt") from exc
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
        print("Warning: this Python is missing " + ", ".join(missing) + ". Group B code will probably fail to run.")
        print("Run: pip install -r requirements.txt")


def normalize_verdict(raw: str, allow_exec_fail: bool) -> str | None:
    text = raw.strip().lower()
    if text == "":
        return None
    mapping = {
        "c": "correct",
        "correct": "correct",
        "y": "correct",
        "1": "correct",
        "w": "wrong",
        "wrong": "wrong",
        "n": "wrong",
        "0": "wrong",
        "s": "skip",
        "skip": "skip",
        "e": "exec_fail",
        "fail": "exec_fail",
    }
    verdict = mapping.get(text)
    if verdict is None:
        print("Unrecognized. Enter c / w / e / s, or press Enter to skip this grade.")
        return "INVALID"
    if verdict == "exec_fail" and not allow_exec_fail:
        print("Group A has no execution-failed grade. Enter c / w / s.")
        return "INVALID"
    return verdict


def ask_verdict(label: str, allow_exec_fail: bool) -> str | None:
    hint = "c=correct / w=wrong / e=execution failed / s=skip" if allow_exec_fail else "c=correct / w=wrong / s=skip"
    while True:
        try:
            raw = input(f"{label} [{hint}, Enter=grade later]: ")
        except EOFError:
            print()
            return None
        verdict = normalize_verdict(raw, allow_exec_fail)
        if verdict != "INVALID":
            return verdict


def verdict_label(value: str | None) -> str:
    return {
        "correct": "correct",
        "wrong": "wrong",
        "exec_fail": "execution failed",
        "skip": "skipped",
        None: "ungraded",
    }.get(value, "ungraded")


def print_block(title: str, text: str) -> None:
    print(f"\n--- {title} ---")
    print(text if text else "(empty)")


def grade_record(record: RunRecord) -> None:
    print("\n--- Grade the answer itself. Code that looks tidy but prints the wrong value is wrong ---")
    record.group_a["verdict"] = ask_verdict("Group A", allow_exec_fail=False)
    record.group_b["verdict"] = ask_verdict("Group B", allow_exec_fail=True)
    try:
        note = input("Note (optional): ").strip()
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
    print_block("Question", question.question)

    if "A" in groups:
        print("\nRequesting group A (direct answer)...")
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
        except Exception as exc:  # noqa: BLE001 - one item failing must not stop the batch
            group_a["error"] = f"{type(exc).__name__}: {exc}"
        record.group_a = group_a
        if group_a["error"]:
            print_block("Group A call failed", group_a["error"])
        else:
            print_block("Group A: direct answer", group_a["answer"])

    if "B" in groups:
        print("\nRequesting group B (write code)...")
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
            print_block("Group B call failed", group_b["error"])
        else:
            code = extract_code(group_b["raw_response"])
            if not code:
                group_b["exec_status"] = "no_code"
                print_block("Group B produced no code. Raw reply", group_b["raw_response"])
            else:
                group_b["code"] = code
                flags = analyze_code(code)
                group_b.update(flags)
                code_dir.mkdir(parents=True, exist_ok=True)
                code_path = code_dir / f"{question.item_id}.py"
                code_path.write_text(code, encoding="utf-8")
                group_b["code_file"] = str(code_path)
                print_block("Group B: generated code", code)
                stdout, stderr, returncode = run_code(code, timeout)
                group_b["stdout"] = stdout
                group_b["stderr"] = stderr
                group_b["returncode"] = returncode
                group_b["exec_status"] = "ok" if returncode == 0 else "fail"
                print_block("Group B: execution STDOUT (this is the group B answer)", stdout)
                if stderr.strip():
                    print_block("Group B: STDERR", stderr)
                print(f"Return code: {returncode}")
                hint = "called sin/cos" if flags["uses_trig_call"] else "did not call sin/cos; the answer may be hardcoded"
                print(f"Code check: {hint}; sympy={flags['uses_sympy']}; numpy={flags['uses_numpy']}")
        record.group_b = group_b

    print_block("Reference answer (for you only; it was not given to the model)", question.reference)
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
        "item",
        "dataset",
        "definition note",
        "group A answer",
        "group A grade",
        "group B output",
        "group B grade",
        "execution status",
        "called a trig function",
        "possible hardcoded answer",
        "group B code file",
        "note",
        "reference answer",
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
                    "item": f"{record.get('section', '')} {record.get('qid', '')}",
                    "dataset": record.get("dataset", ""),
                    "definition note": record.get("dataset_note", ""),
                    "group A answer": group_a.get("answer") or group_a.get("error") or "",
                    "group A grade": verdict_label(group_a.get("verdict")),
                    "group B output": group_b.get("stdout") or group_b.get("error") or "",
                    "group B grade": verdict_label(group_b.get("verdict")),
                    "execution status": group_b.get("exec_status") or "",
                    "called a trig function": "yes" if group_b.get("uses_trig_call") else "no",
                    "possible hardcoded answer": "yes" if group_b.get("possible_hardcode") else "no",
                    "group B code file": group_b.get("code_file") or "",
                    "note": note,
                    "reference answer": record.get("reference_answer") or "",
                }
            )


def write_report(path: Path, records: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Semi-automatic comparison record",
        "",
        f"- Model: {meta.get('model', '')}",
        f"- Temperature: {meta.get('temperature', '')}",
        f"- Time: {meta.get('created_at', '')}",
        "",
        "Group A is graded on the written answer. Group B is graded on the STDOUT of the executed code. A run that does not succeed is an execution failure and counts as wrong.",
        "Possible hardcoded answer is only a hint: if the code never calls sin/cos, open it and look.",
        "",
    ]
    for record in records:
        group_a = record.get("group_a") or {}
        group_b = record.get("group_b") or {}
        lines.append(f"## {record.get('item_id')}")
        lines.append("")
        lines.append(f"- Dataset: {record.get('dataset')} ({record.get('dataset_note')})")
        lines.append(f"- Group A: {verdict_label(group_a.get('verdict'))}")
        lines.append(f"- Group B: {verdict_label(group_b.get('verdict'))}; execution status={group_b.get('exec_status') or 'none'}")
        if group_b:
            lines.append(
                f"- Code check: trig call={'yes' if group_b.get('uses_trig_call') else 'no'};"
                f"possible hardcode={'yes' if group_b.get('possible_hardcode') else 'no'}"
            )
        note = group_a.get("note") or group_b.get("note") or ""
        if note:
            lines.append(f"- Note: {note}")
        lines.append("")
        lines.append("### Question")
        lines.append("")
        lines.append(record.get("question") or "")
        lines.append("")
        lines.append("### Group A answer")
        lines.append("")
        lines.append(group_a.get("answer") or group_a.get("error") or "(not run)")
        lines.append("")
        lines.append("### Group B code")
        lines.append("")
        lines.append("```python")
        lines.append(group_b.get("code") or "(no code)")
        lines.append("```")
        lines.append("")
        lines.append("### Group B STDOUT")
        lines.append("")
        lines.append("```")
        lines.append(group_b.get("stdout") or "(no output)")
        lines.append("```")
        if (group_b.get("stderr") or "").strip():
            lines.append("")
            lines.append("### Group B STDERR")
            lines.append("")
            lines.append("```")
            lines.append(group_b["stderr"])
            lines.append("```")
        lines.append("")
        lines.append("### Reference answer (not given to the model)")
        lines.append("")
        lines.append(record.get("reference_answer") or "(this item has no extracted text answer in the document)")
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
        f"{len(records)} items.\n"
        f"Group A graded {graded_a}: correct {count('group_a', 'correct')}, wrong {count('group_a', 'wrong')}, skipped {count('group_a', 'skip')}.\n"
        f"Group B graded {graded_b}: correct {count('group_b', 'correct')}, wrong {count('group_b', 'wrong')},"
        f" execution failed (grade) {count('group_b', 'exec_fail')}, skipped {count('group_b', 'skip')}.\n"
        f"Group B did not execute successfully {exec_fail}/{ran_b}; possible hardcoded answer {hardcode}/{ran_b}."
    )


def save_bundle(out_dir: Path, records: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir / "records.jsonl", records)
    write_grades_csv(out_dir / "grades.csv", records)
    write_report(out_dir / "report.md", records, meta)


def print_question_list(questions: list[Question]) -> None:
    print(f"{len(questions)} items\n")
    for question in questions:
        ref = oneline(question.reference, 60) or "(no text reference answer)"
        print(f"{question.item_id}  {question.dataset_note}")
        print(f"  Q: {oneline(question.question, 100)}")
        print(f"  Reference: {ref}")


def dry_run(questions: list[Question]) -> int:
    leaked = 0
    for question in questions:
        prompt_direct, prompt_code = build_prompts(question)
        direct_leak = reference_leaked(prompt_direct, question.reference, question.baseline_document)
        code_leak = reference_leaked(prompt_code, question.reference, question.baseline_document)
        leaked += int(direct_leak or code_leak)
        has_baseline = bool(question.baseline_document) and question.baseline_document in prompt_direct
        print("=" * 72)
        print(question.item_id, question.dataset_note)
        print(f"Full baseline is in the prompt: {has_baseline} ({len(question.baseline_document)} characters)")
        print(f"Beta gold appears outside the baseline: A={direct_leak} B={code_leak}")
        print(f"definition length={len(question.definitions)} question length={len(question.question)} reference length={len(question.reference)}")
        print_block("Question", question.question)
        print_block("End of the group A prompt", prompt_direct[-500:])
        print_block("Group B requirements", prompt_code.split("Requirements", 1)[-1])
    print(f"\nCheck finished: {len(questions)} items, beta gold may have leaked into the prompt for {leaked}.")
    return 1 if leaked else 0


def grade_existing(path: Path) -> None:
    records = read_jsonl(path)
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    print(f"Read {len(records)} records: {path}")
    for index, record in enumerate(records, start=1):
        group_a = record.setdefault("group_a", {})
        group_b = record.setdefault("group_b", {})
        if group_a.get("verdict") and group_b.get("verdict"):
            continue
        print("\n" + "=" * 72)
        print(f"[{index}/{len(records)}] {record.get('item_id')}  {record.get('dataset_note')}")
        print_block("Question", record.get("question") or "")
        print_block("Group A answer", group_a.get("answer") or group_a.get("error") or "(none)")
        print_block("Group B STDOUT", group_b.get("stdout") or "(none)")
        if group_b.get("exec_status") and group_b.get("exec_status") != "ok":
            print(f"Group B execution status: {group_b.get('exec_status')}")
            if group_b.get("stderr"):
                print_block("Group B STDERR", group_b["stderr"])
        if group_b.get("possible_hardcode"):
            print("Code check: sin/cos was not called. Look at the code in case the answer is hardcoded.")
            if group_b.get("code_file"):
                print(f"Code file: {group_b['code_file']}")
        print_block("Reference answer (not given to the model)", record.get("reference_answer") or "")
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
    print(f"Wrote back: {path.parent}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group A answers directly, group B executes code, and a person grades both.")
    parser.add_argument(
        "--dataset",
        choices=["beta"],
        default="beta",
        help="Grade beta only (sideslip positive to the left). The full baseline is study material",
    )
    parser.add_argument("--only", default=DEFAULT_ONLY, help="Item filter, for example coord:1-7,joint:1-2; all means every item")
    parser.add_argument("--groups", default="AB", help="Which groups to run: A, B, or AB")
    parser.add_argument("--grade", choices=["now", "later"], default="now", help="now grades each item immediately, later runs first and grades afterward")
    parser.add_argument("--grade-only", metavar="JSONL", help="Grade an existing records.jsonl without calling the model")
    parser.add_argument("--list", action="store_true", help="List the items and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print the prompts and check whether the reference answer leaked")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N items; 0 means no limit")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for one code execution, in seconds")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="", help="Default deepseek-flash, or the value of DEEPSEEK_MODEL")
    parser.add_argument("--base-url", default="", help="Default https://api.deepseek.com")
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled"],
        default="",
        help="DeepSeek thinking mode. The default is disabled, so temperature=0 takes effect",
    )
    parser.add_argument("--out", default="", help="Result directory. The default is results/<timestamp>")
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
        print("No items matched. Use --list to see them.")
        return 1

    if args.list:
        print_question_list(questions)
        return 0
    if args.dry_run:
        return dry_run(questions)

    if thinking not in {"enabled", "disabled"}:
        raise SystemExit("DEEPSEEK_THINKING must be enabled or disabled.")

    groups = {ch for ch in args.groups.upper() if ch in {"A", "B"}}
    if not groups:
        raise SystemExit("--groups must include A or B.")

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
    print(f"Model: {model}    Endpoint: {base_url}")
    print(f"Thinking: {thinking}    Temperature: {args.temperature}    Items: {len(questions)}")
    if is_deepseek(base_url, model) and thinking == "enabled":
        print("DeepSeek thinking mode ignores temperature, so this temperature setting has no effect.")
    print(f"Result directory: {out_dir}")
    print("The full baseline question-and-answer text is given to the model. The beta gold is shown only while grading. Group B is judged on STDOUT.")

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
    print(f"Records: {out_dir / 'records.jsonl'}")
    print(f"Grade sheet: {out_dir / 'grades.csv'}")
    print(f"Readable report: {out_dir / 'report.md'}")
    if args.grade == "later":
        print(f"Grade later: python semi_auto_eval.py --grade-only \"{out_dir / 'records.jsonl'}\"")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Items that already finished are saved in the result directory.")
        raise SystemExit(130)
