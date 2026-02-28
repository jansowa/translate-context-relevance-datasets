#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class RateLimitReached(Exception):
    """Raised when provider returns status indicating quota/rate limit (typically 429)."""


def escape_control_chars_in_json_strings(s: str) -> str:
    """
    Replaces raw control characters inside JSON strings with escaped forms.
    """
    out = []
    in_string = False
    escape = False

    for ch in s:
        c = ord(ch)

        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue

            if ch == "\\":
                out.append(ch)
                escape = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if c < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{c:04x}")
                continue

            out.append(ch)
            continue

        if ch == '"':
            out.append(ch)
            in_string = True
            continue
        out.append(ch)

    return "".join(out)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in the model response.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                return json.loads(chunk)
    raise ValueError("Failed to extract a complete JSON object from the response.")


_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def checkpoint_stem_from_id(rid: str) -> str:
    cleaned = _WINDOWS_FORBIDDEN.sub("_", rid).strip(" .")
    cleaned = cleaned[:60] if cleaned else "id"
    h = hashlib.sha1(rid.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned}__{h}"


def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids_from_jsonl(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if "id" in o:
                    done.add(o["id"])
            except Exception:
                pass
    return done


def spans_to_pieces(text: str, spans: List[List[int]]) -> Tuple[List[str], List[str]]:
    if not spans:
        return [text], []

    gaps: List[str] = []
    span_texts: List[str] = []
    prev_end = 0

    for (s, e) in spans:
        s = int(s)
        e = int(e)
        if s < prev_end:
            raise ValueError(f"Spans overlap: prev_end={prev_end}, start={s}")
        if s > len(text) or e > len(text) or s < 0 or e < 0 or e < s:
            raise ValueError(f"Invalid span [{s},{e}) for text length {len(text)}")
        gaps.append(text[prev_end:s])
        span_texts.append(text[s:e])
        prev_end = e

    gaps.append(text[prev_end:])
    return gaps, span_texts


def rebuild_text_and_spans(gaps: List[str], translated_spans: List[str]) -> Tuple[str, List[List[int]]]:
    if len(gaps) != len(translated_spans) + 1:
        raise ValueError("Length mismatch: gaps must have len(spans)+1")

    out_parts: List[str] = []
    new_spans: List[List[int]] = []
    cursor = 0

    for span_pl in translated_spans:
        gap = gaps[len(new_spans)]
        out_parts.append(gap)
        cursor += len(gap)

        start = cursor
        out_parts.append(span_pl)
        cursor += len(span_pl)
        end = cursor
        new_spans.append([start, end])

    out_parts.append(gaps[-1])
    return "".join(out_parts), new_spans


SYSTEM_QUERY = (
    "You are an EN->PL translator. Return ONLY valid JSON. "
    "Do not add any comments or markdown. "
    "Use natural Polish, preserve proper names, numbers, dates, and meaning."
)

SYSTEM_TEXT = (
    "You are an EN->PL translator for NLP data. Return ONLY valid JSON. "
    "Do not add any comments or markdown. "
    "Do not change the number of spans: each span must have its own translated text. "
    "Preserve meaning, proper names, numbers, and quotations. "
    "In 'think_process', keep indices like 'Sentence [1]' etc. (translate only the surrounding text)."
)

SYSTEM_TEXT_SIMPLE = (
    "You are an EN->PL translator for NLP data. Return ONLY valid JSON. "
    "Do not add any comments or markdown. "
    "Preserve meaning, intent, proper names, numbers, and quotations."
)

TOXIC_LABEL_DESCRIPTIONS: dict[str, str] = {
    "toxic": "ogolnie toksyczny komentarz (wrogi, obrazliwy, agresywny)",
    "severe_toxic": "silnie toksyczny / ekstremalnie toksyczny komentarz (bardziej skrajny podzbior toxic)",
    "obscene": "tresc obsceniczna lub wulgarna",
    "threat": "grozby przemocy lub zastraszanie",
    "insult": "wyzwiska lub bezposrednie obrazanie osoby albo grupy",
    "identity_hate": "mowa nienawisci wobec tozsamosci (np. rasa, religia, plec, orientacja)",
}

WILDGUARD_SUBCATEGORY_DESCRIPTIONS: dict[str, str] = {
    "benign": "Nieszkodliwy, bezpieczny prompt (brak intencji wywolania szkody).",
    "causing_material_harm_by_disseminating_misinformation": (
        "Misinformacja, ktora moze prowadzic do realnej szkody (np. zdrowie, prawo, bezpieczenstwo)."
    ),
    "copyright_violations": "Prosby o naruszenie praw autorskich (piractwo, nielegalne kopiowanie lub udostepnianie).",
    "cyberattack": "Instrukcje lub pomoc w atakach cybernetycznych (malware, wlamania, exploitacja).",
    "defamation_encouraging_unethical_or_unsafe_actions": (
        "Tresci znieslawiajace lub zachecajace do nieetycznych albo niebezpiecznych dzialan."
    ),
    "disseminating_false_or_misleading_information_encouraging_disinformation_campaigns": (
        "Tworzenie lub rozpowszechnianie falszywych lub wprowadzajacych w blad informacji, "
        "w tym wsparcie kampanii dezinformacyjnych."
    ),
    "fraud_assisting_illegal_activities": (
        "Wsparcie oszustw i innych dzialan nielegalnych (scamy, wyludzenia, obejscia prawa)."
    ),
    "mental_health_over-reliance_crisis": (
        "Tresci ryzykowne dla zdrowia psychicznego lub promujace niebezpieczne nadmierne poleganie na asystencie."
    ),
    "others": "Inne szkody, ktore nie pasuja jasno do pozostalych kategorii.",
    "private_information_individual": (
        "Prywatne dane osoby fizycznej (PII, dane medyczne, dane kontaktowe itp.)."
    ),
    "sensitive_information_organization_government": (
        "Wrazliwe informacje organizacji lub instytucji lub rzadu (np. dane poufne, operacyjne, bezpieczenstwa)."
    ),
    "sexual_content": "Tresci seksualne lub erotyczne.",
    "social_stereotypes_and_unfair_discrimination": (
        "Stereotypy spoleczne i niesprawiedliwa dyskryminacja grup."
    ),
    "toxic_language_hate_speech": "Jezyk toksyczny, obrazliwy, mowa nienawisci.",
    "violence_and_physical_harm": "Przemoc i tresci dotyczace wyrazdzania szkody fizycznej.",
}


def normalize_wildguard_subcategories(raw_value: Any) -> List[str]:
    if raw_value is None:
        raise ValueError("subcategories is null")

    if isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, list):
        values = [x for x in raw_value if isinstance(x, str)]
        if len(values) != len(raw_value):
            raise ValueError("subcategories contains non-string values")
    else:
        raise ValueError(f"subcategories has unsupported type: {type(raw_value).__name__}")

    cleaned = []
    seen = set()
    for item in values:
        key = item.strip()
        if not key:
            continue
        if key not in seen:
            seen.add(key)
            cleaned.append(key)

    if not cleaned:
        raise ValueError("subcategories is empty")

    unknown = [x for x in cleaned if x not in WILDGUARD_SUBCATEGORY_DESCRIPTIONS]
    if unknown:
        raise ValueError(f"unknown subcategories: {unknown}")

    return cleaned


def build_wildguard_prompt(prompt_en: str, subcategories: List[str]) -> str:
    if subcategories == ["benign"]:
        risk_instruction = (
            "The source prompt is benign and does not request harm.\n"
            "Keep the Polish translation benign and safe. Do not introduce harmful intent."
        )
    else:
        subcats_block = "\n".join(
            [f"- {name}: {WILDGUARD_SUBCATEGORY_DESCRIPTIONS[name]}" for name in subcategories]
        )
        risk_instruction = (
            "The source prompt has harmful-risk intent.\n"
            "Preserve the same intent and risk profile in Polish.\n"
            "Do not sanitize or neutralize the harmful aspects.\n"
            "Active subcategories:\n"
            f"{subcats_block}"
        )

    return (
        "Translate the English prompt into Polish.\n"
        "Return ONLY valid JSON.\n"
        "Do not add explanations, markdown, or extra keys.\n\n"
        "Output format:\n"
        '{"prompt_pl": "..."}\n\n'
        f"{risk_instruction}\n\n"
        "INPUT DATA:\n"
        f"PROMPT (EN): {prompt_en}"
    )


def build_toxic_comment_prompt(comment_text_en: str, active_toxic_types: List[str]) -> str:
    if active_toxic_types:
        active_block = "\n".join(
            [f"- {label}: {TOXIC_LABEL_DESCRIPTIONS[label]}" for label in active_toxic_types]
        )
        toxicity_instruction = (
            "The source comment is toxic. Preserve the same toxicity types in the Polish translation.\n"
            "Do not sanitize, soften, or neutralize the toxicity. Keep intent and severity.\n"
            "Active toxicity types in this comment:\n"
            f"{active_block}"
        )
    else:
        toxicity_instruction = (
            "The source comment is NOT toxic (all toxicity labels are 0).\n"
            "Keep the Polish translation non-toxic as well. Do not introduce any toxic content."
        )

    return (
        "Translate the English comment into Polish.\n"
        "Return ONLY valid JSON.\n"
        "Do not add explanations, markdown, or extra keys.\n\n"
        "Output format:\n"
        '{"comment_text_pl": "..."}\n\n'
        f"{toxicity_instruction}\n\n"
        "INPUT DATA:\n"
        f"COMMENT (EN): {comment_text_en}"
    )


def build_query_prompt(query_en: str, positive_fragments: List[str]) -> str:
    frags = "\n".join([f"[{i + 1}] {t}" for i, t in enumerate(positive_fragments)]) or "(none)"
    return (
        "Translate the query below into Polish so that it is semantically consistent and sounds natural. "
        "Also adapt the phrasing to the context using the relevant fragments.\n\n"
        "Return JSON exactly in the following format:\n"
        '{"query_pl": "..."}\n\n'
        "INPUT DATA:\n"
        f"QUERY (EN): {query_en}\n\n"
        "Relevant context fragments (EN):\n"
        f"{frags}"
    )


def build_text_prompt(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    spans_block = []
    for i, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {i} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "Your task: translate into Polish the data for ONE candidate document.\n\n"
        "The text is provided as a list of SPANS (in order). "
        "Do NOT merge and do NOT split spans. Translate each span separately.\n\n"
        "Requirements:\n"
        "1) Return a 'translated_spans' list of the same length as the number of spans.\n"
        "2) Translate 'think_process' into Polish as 'translated_think_process'.\n"
        "   - If the text contains the quoted query (e.g. in quotes), replace it with the Polish query (exactly as above).\n"
        "   - Preserve indices like 'Sentence [1]' (do not change numbers or brackets).\n\n"
        "Return JSON exactly in the following format:\n"
        '{"translated_spans": ["...", "..."], "translated_think_process": "..."}\n\n'
        "INPUT DATA:\n"
        f"QUERY (PL) - use this wording consistently: {query_pl}\n"
        f"QUERY (EN) - for reference only: {query_en}\n\n"
        f"DOCUMENT LABEL (labels): {int(doc_label)}  (1 = relevant, 0 = not relevant)\n\n"
        "SPANS (EN):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN) associated with this document:\n"
        f"{think_process_en}"
    )


def build_text_prompt_strict(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    n = len(spans_texts_en)
    spans_block = []
    for idx, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {idx}/{n} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "EN->PL TRANSLATION. This is a structural task.\n"
        "You must return ONLY valid JSON.\n"
        "KEY CONSTRAINT: the number of elements in 'translated_spans' MUST be exactly N (provided in INPUT DATA).\n"
        "You must not remove, merge, or split spans. One EN span -> one PL span.\n\n"
        "Return JSON exactly in the following format:\n"
        '{"translated_spans": ["SPAN1_PL", "SPAN2_PL", "..."], "translated_think_process": "..."}\n\n'
        "Note: if you want to use a newline inside a JSON string, use the \\n escape sequence (do not insert a raw newline).\n\n"
        "INPUT DATA:\n"
        f"N_SPANS: {n}\n"
        f"QUERY (PL): {query_pl}\n"
        f"QUERY (EN): {query_en}\n"
        f"DOCUMENT LABEL: {int(doc_label)}\n\n"
        "SPANS (EN) (numbered):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN):\n"
        f"{think_process_en}"
    )


def build_text_prompt_dictforced(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    n = len(spans_texts_en)
    spans_block = []
    for idx, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {idx}/{n} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "EN->PL TRANSLATION. You must return ONLY valid JSON.\n"
        "You must return exactly N spans (provided in INPUT DATA).\n"
        "Return them as a 'translated_spans_dict' object with string keys from \"1\" to \"N\".\n"
        "You MUST NOT skip any number.\n\n"
        "Return JSON exactly in the following format:\n"
        "{"
        '"translated_spans_dict": {"1": "...", "2": "...", "3": "..."}'
        ', "translated_think_process": "..."'
        "}\n\n"
        "Note: use \\n in JSON strings instead of raw newlines.\n\n"
        "INPUT DATA:\n"
        f"N_SPANS: {n}\n"
        f"QUERY (PL): {query_pl}\n"
        f"QUERY (EN): {query_en}\n"
        f"DOCUMENT LABEL: {int(doc_label)}\n\n"
        "SPANS (EN):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN):\n"
        f"{think_process_en}"
    )
