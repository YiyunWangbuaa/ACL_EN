# -*- coding: utf-8 -*-
"""Extract text from all PDFs in the literature folder to .txt files."""
import os
import sys
from pathlib import Path

from pypdf import PdfReader

SRC = Path(r"C:\Users\minec\Desktop\文献\英文文献\强化学习后训练\LLM_工程专题\工程异构问题研究\符号有界性证明")
OUT = Path(r"C:\Users\minec\Desktop\ACL\workflow\analysis\lit_txt")
OUT.mkdir(exist_ok=True)


def extract(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            text = f"[page {i} error: {exc}]"
        parts.append(text)
    return "\n\n".join(parts)


def main():
    for pdf in sorted(SRC.glob("*.pdf")):
        name = pdf.stem
        txt_path = OUT / f"{name}.txt"
        if txt_path.exists() and txt_path.stat().st_size > 500:
            print(f"skip {name} (exists)")
            continue
        try:
            text = extract(pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {name}: {exc}")
            continue
        txt_path.write_text(text, encoding="utf-8")
        print(f"{name}: {len(text)} chars -> {txt_path.name}")


if __name__ == "__main__":
    main()
