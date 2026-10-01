"""The output contract. Every branch of the failure taxonomy is pinned here."""

import pytest

from inference.schema import (
    UNKNOWN_SEMANTIC,
    Failure,
    extract_json_object,
    validate,
)


def test_valid_output_passes_through_unchanged():
    semantic, report = validate('{"path_status": "blocked", "obstacle_location": "front_left"}')
    assert report.ok
    assert report.failure is Failure.NONE
    assert semantic == {"path_status": "blocked", "obstacle_location": "front_left"}


def test_explicit_unknown_is_a_valid_answer():
    semantic, report = validate('{"path_status": "unknown", "obstacle_location": "unknown"}')
    assert report.ok
    assert semantic == UNKNOWN_SEMANTIC


@pytest.mark.parametrize(
    "text",
    [
        "the path ahead looks blocked",
        "",
        "{not json at all",
        "[1, 2, 3",
    ],
)
def test_malformed_json(text):
    semantic, report = validate(text)
    assert report.failure is Failure.MALFORMED_JSON
    assert semantic == UNKNOWN_SEMANTIC


@pytest.mark.parametrize(
    "text",
    [
        '{"path_status": "blocked"}',                                  # missing key
        '{"obstacle_location": "left"}',                               # missing key
        '{"path_status": "obstructed", "obstacle_location": "left"}',  # value not in enum
        '{"path_status": "clear", "obstacle_location": 3}',            # wrong type
    ],
)
def test_schema_violation(text):
    semantic, report = validate(text)
    assert report.failure is Failure.SCHEMA_VIOLATION
    assert semantic == UNKNOWN_SEMANTIC


@pytest.mark.parametrize(
    "text",
    [
        '{"path_status": "blocked", "obstacle_location": "none"}',
        '{"path_status": "clear", "obstacle_location": "front_left"}',
        '{"path_status": "unknown", "obstacle_location": "left"}',
    ],
)
def test_individually_legal_values_can_still_be_unusable(text):
    semantic, report = validate(text)
    assert report.failure is Failure.UNUSABLE_SEMANTICS
    assert semantic == UNKNOWN_SEMANTIC


def test_fenced_json_is_extracted_and_the_fact_is_recorded():
    semantic, report = validate(
        'Sure!\n```json\n{"path_status": "clear", "obstacle_location": "none"}\n```\n'
    )
    assert report.ok
    assert report["extracted"] is True  # visible, so a bad prompt cannot hide
    assert semantic["path_status"] == "clear"


def test_confidence_field_is_stripped_not_published():
    semantic, report = validate(
        '{"path_status": "clear", "obstacle_location": "none", "confidence": "high"}'
    )
    assert report.ok
    assert "confidence" not in semantic
    assert report["extra_keys_stripped"] == ["confidence"]


def test_object_wrapped_in_a_list_is_recovered_by_extraction():
    """Documented behaviour, not an accident.

    The extractor scans for the first balanced top-level object, so a model
    that wraps its answer in a list still yields a usable result. The fact that
    extraction was needed is recorded, so the prompt problem stays visible in
    the extraction rate.
    """
    semantic, report = validate('[{"path_status": "clear", "obstacle_location": "none"}]')
    assert report.ok
    assert report["extracted"] is True


def test_extract_ignores_braces_inside_strings():
    text = '{"path_status": "clear", "obstacle_location": "none", "note": "a } brace"}'
    found, needed = extract_json_object(text)
    assert needed is False and found == text
