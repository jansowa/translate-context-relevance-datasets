#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from typing import Any


_BAD_END_RE = re.compile(r"[\s,:;{\[(\-]$")
_JSON_NOISE_RE = re.compile(
    r"""(Now,\s*in\s*JSON\s*format\s*:|Nowe tłumaczenie\s*:|technical_errors|["']\d+["']\s*:)""",
    re.IGNORECASE,
)


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def balanced_brackets(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    openings = set(pairs.values())
    stack = []
    for ch in text:
        if ch in openings:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def has_parser_noise(text: str) -> bool:
    if _JSON_NOISE_RE.search(text):
        return True
    if "}  {" in text or "}{" in text:
        return True
    if re.search(r"""\(\s*['"]\d+['"]\s*:\s*['"]?""", text):
        return True
    return False


def weird_symbol_ratio(text: str) -> bool:
    if len(text) < 40:
        return False
    weird = len(re.findall(r"""[{}\[\]|\\]""", text))
    return weird / max(len(text), 1) > 0.03


def looks_truncated(text: str) -> bool:
    t = normalize_spaces(text)
    if len(t) < 25:
        return True
    if _BAD_END_RE.search(t):
        return True
    if re.search(r"""(^|[\s([{-])(czes\.|ang\.|ur\.)$""", t, re.IGNORECASE):
        return True
    return False


def normalize_for_dup(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def repeated_large_chunk(text: str) -> bool:
    t = normalize_for_dup(text)

    parts = re.split(r"""\s*\}\s*|\s*\{\s*|\n{2,}""", t)
    parts = [p.strip() for p in parts if len(p.strip()) >= 80]

    if len(parts) < 2:
        return False

    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            a, b = parts[i], parts[j]

            if a == b:
                return True

            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if len(shorter) >= 120 and shorter in longer:
                return True

    return False


def normalize_letters_only(text: str) -> str:
    return "".join(re.findall(r"[A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż]+", str(text)))


def anchor_copied_into_positive(anchor: str, positive: str) -> bool:
    na = normalize_letters_only(anchor).lower()
    np = normalize_letters_only(positive).lower()

    if len(na) < 25:
        return False

    return na in np


def evaluate_text(anchor: str, positive: str) -> tuple[bool, list[str]]:
    reasons = []

    if not positive or not str(positive).strip():
        reasons.append("empty_positive")
        return False, reasons

    p = normalize_spaces(positive)

    if has_parser_noise(p):
        reasons.append("parser_noise")

    if not balanced_brackets(p):
        reasons.append("unbalanced_brackets")

    if looks_truncated(p):
        reasons.append("looks_truncated")

    if weird_symbol_ratio(p):
        reasons.append("too_many_weird_symbols")

    if repeated_large_chunk(p):
        reasons.append("repeated_large_chunk")

    if anchor_copied_into_positive(anchor, p):
        reasons.append("anchor_copied_into_positive")

    return len(reasons) == 0, reasons


def evaluate_row(anchor: str, positive: str) -> dict[str, Any]:
    is_good, reasons = evaluate_text(anchor, positive)
    return {
        "anchor": anchor,
        "positive": positive,
        "is_good": is_good,
        "reasons": reasons,
        "reasons_str": "|".join(reasons),
    }
