# -*- coding: utf-8 -*-
"""Model contract cognition, shared by the four domains.

Recognition answers overwrite the gold declaration. Derived answers do not.
A calculator then sees that cognition plus the direct parent answers it executes.
"""
from __future__ import annotations


def answer_sheet(answers: dict) -> dict[str, str]:
    """Non-empty model answers, keyed by node id."""
    sheet: dict[str, str] = {}
    for node, raw in answers.items():
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            sheet[str(node)] = text
    return sheet


def build_cognition(gold: dict, answers: dict, recognition) -> dict[str, str]:
    """Gold declaration, then every recognition answer the model actually wrote."""
    cognition = {str(node): str(value) for node, value in gold.items()}
    recognized = {str(node) for node in recognition}
    for node, raw in answers.items():
        node = str(node)
        if node not in recognized or raw is None:
            continue
        text = str(raw).strip()
        if text:
            cognition[node] = text
    return cognition


def working_texts(gold: dict, answers: dict, recognition, parents) -> dict[str, str]:
    """Cognition base, with this node's direct parent answers written over it."""
    texts = build_cognition(gold, answers, recognition)
    for parent in parents:
        parent = str(parent)
        raw = answers.get(parent)
        if raw is None:
            continue
        text = str(raw).strip()
        if text:
            texts[parent] = text
    return texts
