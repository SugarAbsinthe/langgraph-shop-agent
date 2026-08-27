"""Metric tests for the retrieval evaluation runner."""

from evals.retrieval_runner import evaluate_retrieval
from evals.retrieval_schema import RetrievalCase
from src.retrieval.models import ProductCandidate, RetrievalResult


class FakeRetriever:
    def search(self, query):
        products = {
            "good": [ProductCandidate(
                product_id=1,
                brand="联想",
                category="笔记本电脑",
                price=7000,
                sources=["description", "sparse"],
            )],
            "bad": [ProductCandidate(
                product_id=2,
                brand="华硕",
                category="笔记本电脑",
                price=9000,
                sources=["spec"],
            )],
            "empty": [],
        }[query.text]
        return RetrievalResult(products=products)


def test_runner_reports_quality_and_safety_metrics():
    cases = [
        RetrievalCase(
            id="good_case",
            query="good",
            relevant_ids=[1],
            required_sources=["description"],
            constraints={"max_price": 8000},
        ),
        RetrievalCase(
            id="bad_case",
            query="bad",
            relevant_ids=[1],
            forbidden_ids=[2],
            required_sources=["sparse"],
            constraints={"max_price": 8000},
        ),
        RetrievalCase(id="empty_case", query="empty"),
    ]
    snapshot = evaluate_retrieval(FakeRetriever(), cases).snapshot()
    assert snapshot["hit_rate"] == 0.5
    assert snapshot["recall_at_k"] == 0.5
    assert snapshot["source_coverage"] == 0.5
    assert snapshot["forbidden_violations"] == 1
    assert snapshot["constraint_violations"] == 1
    assert snapshot["empty_result_violations"] == 0
    assert snapshot["failures"][0]["id"] == "bad_case"
