import pytest

from translation_core import (
    checkpoint_stem_from_id,
    escape_control_chars_in_json_strings,
    extract_first_json_object,
    rebuild_text_and_spans,
    spans_to_pieces,
)


def test_checkpoint_stem_from_id_sanitizes_windows_chars() -> None:
    stem = checkpoint_stem_from_id('a<b>:c"d/e\\f|g?*')
    assert "__" in stem
    assert "<" not in stem
    assert ">" not in stem
    assert ":" not in stem
    assert '"' not in stem
    assert "/" not in stem
    assert "\\" not in stem
    assert "|" not in stem
    assert "?" not in stem
    assert "*" not in stem


def test_spans_roundtrip_rebuilds_expected_text_and_spans() -> None:
    text = "alpha BETA gamma"
    spans = [[6, 10]]
    gaps, extracted = spans_to_pieces(text, spans)
    rebuilt_text, rebuilt_spans = rebuild_text_and_spans(gaps, ["OMEGA"])

    assert extracted == ["BETA"]
    assert rebuilt_text == "alpha OMEGA gamma"
    assert rebuilt_spans == [[6, 11]]


def test_spans_to_pieces_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        spans_to_pieces("abcdef", [[0, 3], [2, 4]])


def test_extract_first_json_object_from_wrapped_response() -> None:
    payload = 'Answer:\\n\\n{"ok": true, "n": 1}\\nThanks'
    obj = extract_first_json_object(payload)
    assert obj == {"ok": True, "n": 1}


def test_escape_control_chars_in_json_strings() -> None:
    raw = '{"k":"line1\nline2"}'
    fixed = escape_control_chars_in_json_strings(raw)
    assert fixed == '{"k":"line1\\nline2"}'
