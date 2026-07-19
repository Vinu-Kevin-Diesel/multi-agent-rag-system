"""Decomposition output parsing and fallback.

Replaces the old code-fence + `\\[.*?\\]` regex, which failed silently on malformed output. The
parser accepts both shapes seen in practice — the bare JSON array the local qwen3-router returns
(following the few-shot examples) and the {"sub_questions": [...]} object a schema-enforcing
provider returns — and degrades to the original question rather than raising.
"""

import pytest

from app.agents.decompose import _parse_sub_questions, decompose_question

ORIGINAL = "Which drug requires a test dose, and what ICD-10 code applies?"


def test_parses_bare_json_array():
    """What the local no-think variant returns, matching the few-shot format."""
    raw = '["drug requiring a test dose", "ICD-10 code for drug indication"]'
    assert _parse_sub_questions(raw, ORIGINAL) == [
        "drug requiring a test dose",
        "ICD-10 code for drug indication",
    ]


def test_parses_sub_questions_object():
    """What a schema-enforcing provider (NIM) returns."""
    raw = '{"sub_questions": ["a", "b"]}'
    assert _parse_sub_questions(raw, ORIGINAL) == ["a", "b"]


def test_strips_code_fences():
    raw = '```json\n["a", "b"]\n```'
    assert _parse_sub_questions(raw, ORIGINAL) == ["a", "b"]


def test_caps_at_four():
    raw = '["a", "b", "c", "d", "e", "f"]'
    assert _parse_sub_questions(raw, ORIGINAL) == ["a", "b", "c", "d"]


def test_drops_blank_entries():
    raw = '["a", "  ", "", "b"]'
    assert _parse_sub_questions(raw, ORIGINAL) == ["a", "b"]


def test_unparseable_falls_back_to_original():
    """Markdown prose (what the no-think model emits without the few-shot prompt) is not a crash."""
    raw = "**Sub-question 1:** which drug\n**Sub-question 2:** which code"
    assert _parse_sub_questions(raw, ORIGINAL) == [ORIGINAL]


def test_empty_array_falls_back_to_original():
    assert _parse_sub_questions("[]", ORIGINAL) == [ORIGINAL]


def test_wrong_json_shape_falls_back():
    """Valid JSON, wrong structure — a bare string or number is not a sub-question list."""
    assert _parse_sub_questions('"just a string"', ORIGINAL) == [ORIGINAL]
    assert _parse_sub_questions("42", ORIGINAL) == [ORIGINAL]


@pytest.mark.asyncio
async def test_decompose_question_end_to_end(make_llm_client):
    client = make_llm_client(content='["first hop", "second hop"]')
    assert await decompose_question(client, ORIGINAL) == ["first hop", "second hop"]


@pytest.mark.asyncio
async def test_decompose_falls_back_on_error(make_llm_client):
    """A decompose failure must fall back to single-hop retrieval, never 500 the query."""
    client = make_llm_client(content="[]")
    client.chat.completions.create.side_effect = RuntimeError("connection reset")
    assert await decompose_question(client, ORIGINAL) == [ORIGINAL]
