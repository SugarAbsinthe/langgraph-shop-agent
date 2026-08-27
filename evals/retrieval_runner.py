"""Offline/live retrieval evaluation with constraint-safety metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

from evals.retrieval_schema import RetrievalCase, load_retrieval_cases
from src.retrieval.models import RetrievalQuery


@dataclass
class RetrievalEvaluation:
    cases: int = 0
    relevant_cases: int = 0
    hit_cases: int = 0
    relevant_total: int = 0
    relevant_retrieved: int = 0
    forbidden_violations: int = 0
    constraint_violations: int = 0
    empty_result_violations: int = 0
    required_sources_total: int = 0
    required_sources_met: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "cases": self.cases,
            "hit_rate": round(self.hit_cases / self.relevant_cases, 4)
            if self.relevant_cases else None,
            "recall_at_k": round(self.relevant_retrieved / self.relevant_total, 4)
            if self.relevant_total else None,
            "source_coverage": round(
                self.required_sources_met / self.required_sources_total, 4
            ) if self.required_sources_total else None,
            "forbidden_violations": self.forbidden_violations,
            "constraint_violations": self.constraint_violations,
            "empty_result_violations": self.empty_result_violations,
            "failures": self.failures,
        }


def _violates_constraints(product, case: RetrievalCase) -> bool:
    constraints = case.constraints
    if constraints.min_price is not None and (
        product.price is None or product.price < constraints.min_price
    ):
        return True
    if constraints.max_price is not None and (
        product.price is None or product.price > constraints.max_price
    ):
        return True
    if constraints.category and product.category.casefold() != constraints.category.casefold():
        return True
    excluded = {brand.casefold() for brand in constraints.excluded_brands}
    return product.brand.casefold() in excluded


def evaluate_retrieval(retriever, cases: list[RetrievalCase]) -> RetrievalEvaluation:
    evaluation = RetrievalEvaluation(cases=len(cases))
    for case in cases:
        result = retriever.search(RetrievalQuery(
            text=case.query,
            top_k=case.top_k,
            constraints=case.constraints.model_dump(),
        ))
        returned_ids = {product.product_id for product in result.products}
        relevant = set(case.relevant_ids)
        retrieved_relevant = returned_ids & relevant
        case_failures: list[str] = []

        if relevant:
            evaluation.relevant_cases += 1
            evaluation.relevant_total += len(relevant)
            evaluation.relevant_retrieved += len(retrieved_relevant)
            if retrieved_relevant:
                evaluation.hit_cases += 1
            else:
                case_failures.append("miss")
        elif returned_ids:
            evaluation.empty_result_violations += 1
            case_failures.append("expected_empty")

        forbidden = returned_ids & set(case.forbidden_ids)
        evaluation.forbidden_violations += len(forbidden)
        if forbidden:
            case_failures.append("forbidden_product")

        violations = sum(
            _violates_constraints(product, case) for product in result.products
        )
        evaluation.constraint_violations += violations
        if violations:
            case_failures.append("constraint_violation")

        relevant_products = [
            product for product in result.products if product.product_id in relevant
        ]
        observed_sources = {
            source for product in relevant_products for source in product.sources
        }
        evaluation.required_sources_total += len(case.required_sources)
        met_sources = set(case.required_sources) & observed_sources
        evaluation.required_sources_met += len(met_sources)
        if len(met_sources) != len(case.required_sources):
            case_failures.append("required_source_missing")

        if case_failures:
            evaluation.failures.append({
                "id": case.id,
                "reasons": sorted(set(case_failures)),
                "returned_ids": sorted(returned_ids),
            })
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the active retrieval index")
    parser.add_argument("--cases", default=None)
    args = parser.parse_args()

    from src.config import config
    from src.retrieval.product_retriever import ProductRetriever

    retriever = ProductRetriever(
        chroma_dir=config.PRODUCT_CHROMA_DIR,
        catalog_db=config.PRODUCT_DB_PATH,
    )
    cases = load_retrieval_cases(args.cases) if args.cases else load_retrieval_cases()
    snapshot = evaluate_retrieval(retriever, cases).snapshot()
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return int(
        snapshot["forbidden_violations"] > 0
        or snapshot["constraint_violations"] > 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
