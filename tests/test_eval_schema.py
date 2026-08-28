"""Evaluation case contract and dataset quality checks."""

import json

import pytest
from pydantic import ValidationError

from evals.schema import EvalCase, VALID_STAGES, VALID_TOOLS, load_cases


def test_default_dataset_is_valid_and_unique():
    cases = load_cases()
    assert len(cases) == 25
    assert len({case.id for case in cases}) == len(cases)


def test_dataset_covers_all_stages_and_tools():
    cases = load_cases()
    covered_stages = {stage for case in cases for stage in case.expected_stages}
    covered_tools = {tool for case in cases for tool in case.required_tools}
    assert covered_stages == VALID_STAGES
    assert covered_tools == VALID_TOOLS


def test_dataset_contains_no_obvious_credentials_or_personal_data():
    serialized = "\n".join(case.model_dump_json() for case in load_cases()).lower()
    assert "sk-" not in serialized
    assert "ghp_" not in serialized
    assert "github_pat_" not in serialized
    assert "@qq.com" not in serialized
    assert "@gmail.com" not in serialized


def test_contract_rejects_overlapping_tool_expectations():
    with pytest.raises(ValidationError, match="both required and forbidden"):
        EvalCase.model_validate({
            "id": "invalid_overlap",
            "question": "test",
            "expected_stages": ["search"],
            "required_tools": ["get_reviews"],
            "forbidden_tools": ["get_reviews"],
            "expected_retrieval": True,
            "scripted_tool_rounds": [["get_reviews"]],
        })


def test_contract_rejects_required_tool_outside_executable_rounds():
    with pytest.raises(ValidationError, match="executable scripted round"):
        EvalCase.model_validate({
            "id": "invalid_limit",
            "question": "test",
            "expected_stages": ["search"],
            "required_tools": ["get_reviews"],
            "expected_retrieval": True,
            "max_tool_rounds": 1,
            "scripted_tool_rounds": [["get_product_detail"], ["get_reviews"]],
        })


def test_loader_reports_the_invalid_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({
            "id": "valid_case",
            "question": "test",
            "expected_stages": ["discovery"],
            "expected_retrieval": False,
        })
        + "\n{invalid-json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"cases\.jsonl:2"):
        load_cases(path)
