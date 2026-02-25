"""Tests for extract_json robustness."""

from __future__ import annotations

import json

from paperbanana.core.utils import extract_json


def test_plain_json():
    raw = '{"critic_suggestions": ["fix layout"], "revised_description": "new desc"}'
    result = extract_json(raw)
    assert json.loads(result)["critic_suggestions"] == ["fix layout"]


def test_markdown_json_fence():
    raw = '```json\n{"selected_ids": [1, 2, 3]}\n```'
    result = extract_json(raw)
    assert json.loads(result)["selected_ids"] == [1, 2, 3]


def test_markdown_plain_fence():
    raw = '```\n{"key": "value"}\n```'
    result = extract_json(raw)
    assert json.loads(result)["key"] == "value"


def test_json_with_leading_text():
    raw = 'Here is my analysis:\n{"critic_suggestions": [], "revised_description": null}'
    result = extract_json(raw)
    assert json.loads(result)["critic_suggestions"] == []


def test_json_with_surrounding_text():
    raw = 'The result is:\n{"score": 42}\nHope this helps!'
    result = extract_json(raw)
    assert json.loads(result)["score"] == 42


def test_empty_string():
    assert extract_json("") == ""


def test_no_json_at_all():
    raw = "I could not generate a valid critique."
    result = extract_json(raw)
    # Should return original text (caller handles the JSONDecodeError)
    assert result == raw


def test_nested_braces():
    raw = 'Result: {"outer": {"inner": [1, 2]}, "key": "val"}'
    result = extract_json(raw)
    data = json.loads(result)
    assert data["outer"]["inner"] == [1, 2]


def test_json_array():
    raw = '[1, 2, 3]'
    result = extract_json(raw)
    assert json.loads(result) == [1, 2, 3]
