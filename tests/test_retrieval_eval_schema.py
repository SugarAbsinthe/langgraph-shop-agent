"""RAG-specific evaluation contract quality checks."""

import json

import pytest
from pydantic import ValidationError

from evals.retrieval_schema import RetrievalCase, load_retrieval_cases


def test_retrieval_dataset_is_valid_and_unique():
    cases = load_retrieval_cases()
    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30


def test_retrieval_dataset_covers_sources_constraints_and_empty_results():
    cases = load_retrieval_cases()
    sources = {source for case in cases for source in case.required_sources}
    assert sources == {"description", "spec", "sparse"}
    assert any(case.constraints.max_price is not None for case in cases)
    assert any(case.constraints.excluded_brands for case in cases)
    assert any(not case.relevant_ids for case in cases)


def test_retrieval_dataset_contains_no_credentials_or_personal_data():
    serialized = "\n".join(
        case.model_dump_json() for case in load_retrieval_cases()
    ).lower()
    assert "sk-" not in serialized
    assert "ghp_" not in serialized
    assert "@qq.com" not in serialized
    assert "@gmail.com" not in serialized


def test_contract_rejects_conflicting_ids_and_brands():
    with pytest.raises(ValidationError, match="both relevant and forbidden"):
        RetrievalCase.model_validate({
            "id": "conflicting_ids",
            "query": "test",
            "relevant_ids": [1],
            "forbidden_ids": [1],
        })
    with pytest.raises(ValidationError, match="both preferred and excluded"):
        RetrievalCase.model_validate({
            "id": "conflicting_brands",
            "query": "test",
            "constraints": {
                "preferred_brands": ["联想"],
                "excluded_brands": ["联想"],
            },
        })


def test_loader_reports_invalid_line(tmp_path):
    path = tmp_path / "retrieval.jsonl"
    path.write_text(
        json.dumps({"id": "valid_case", "query": "test"})
        + "\n{invalid}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"retrieval\.jsonl:2"):
        load_retrieval_cases(path)
