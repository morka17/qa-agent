#!/usr/bin/env python3
"""
Benchmarks `ElementResolver` accuracy against a labeled dataset of
(page snapshot, target description, expected element index) triples —
the regression harness `docs/architecture.md`'s roadmap calls for under
"Phase 7 — Benchmarking & Continuous Improvement" (tracked results live
in `benchmarks/selector_resolution_accuracy.md`).

The dataset format is a JSON file: a list of cases, each with a `page`
(the raw DOM snapshot shape `DomSnapshotExtractor` produces), a
`description` (the target NL description to resolve), and an
`expected_index` (which element in `page.elements` is the correct
match).

Usage:
    python scripts/benchmark_selector_accuracy.py --dataset benchmarks/selector_dataset.json
    python scripts/benchmark_selector_accuracy.py --dataset benchmarks/selector_dataset.json --min-accuracy 0.9
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from qa_agent.perception.dom_snapshot import BoundingBox, DomElementNode, DomSnapshot
from qa_agent.perception.element_resolver import ElementResolutionError, ElementResolver
from qa_agent.perception.selector_strategies.selector_cache import SelectorCache


class BenchmarkError(Exception):
    pass


@dataclass
class BenchmarkCase:
    name: str
    snapshot: DomSnapshot
    description: str
    expected_index: int


@dataclass
class BenchmarkResult:
    case_name: str
    correct: bool
    resolved_index: int | None
    strategy: str | None
    confidence: float | None
    error: str | None


def _parse_element(raw: dict) -> DomElementNode:
    bbox_raw = raw.get("bounding_box")
    return DomElementNode(
        index=raw["index"],
        tag=raw["tag"],
        role=raw.get("role"),
        accessible_name=raw.get("accessible_name"),
        text=raw.get("text"),
        value=raw.get("value"),
        attributes=raw.get("attributes", {}),
        dom_path=raw.get("dom_path", f"#{raw['tag']}-{raw['index']}"),
        bounding_box=BoundingBox(**bbox_raw) if bbox_raw else None,
        is_visible=raw.get("is_visible", True),
        is_enabled=raw.get("is_enabled", True),
    )


def load_dataset(path: Path) -> list[BenchmarkCase]:
    raw_cases = json.loads(path.read_text())
    cases = []
    for i, raw_case in enumerate(raw_cases):
        raw_page = raw_case["page"]
        snapshot = DomSnapshot(
            url=raw_page.get("url", f"https://benchmark.local/case-{i}"),
            title=raw_page.get("title", ""),
            elements=[_parse_element(e) for e in raw_page["elements"]],
        )
        cases.append(
            BenchmarkCase(
                name=raw_case.get("name", f"case-{i}"),
                snapshot=snapshot,
                description=raw_case["description"],
                expected_index=raw_case["expected_index"],
            )
        )
    return cases


class NoOpScreenshotPage:
    """The accuracy benchmark focuses on heuristic resolution; vision fallback isn't exercised here."""

    async def screenshot(self) -> bytes:
        raise BenchmarkError(
            "This case required the vision fallback, which the accuracy benchmark doesn't "
            "exercise. Remove it from the dataset, or extend this script with a vision stub."
        )


async def run_benchmark(cases: list[BenchmarkCase]) -> list[BenchmarkResult]:
    results = []
    for case in cases:
        resolver = ElementResolver(vision_client=None, cache=SelectorCache())
        try:
            resolved = await resolver.resolve(NoOpScreenshotPage(), case.snapshot, case.description)
            results.append(
                BenchmarkResult(
                    case_name=case.name,
                    correct=resolved.node.index == case.expected_index,
                    resolved_index=resolved.node.index,
                    strategy=resolved.strategy.value,
                    confidence=resolved.confidence,
                    error=None,
                )
            )
        except ElementResolutionError as exc:
            results.append(
                BenchmarkResult(
                    case_name=case.name,
                    correct=False,
                    resolved_index=None,
                    strategy=None,
                    confidence=None,
                    error=str(exc),
                )
            )
    return results


def print_report(results: list[BenchmarkResult]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / total if total else 0.0

    print(f"{'Case':<30} {'Result':<10} {'Strategy':<18} {'Confidence':<10}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r.correct else "FAIL"
        strategy = r.strategy or "-"
        confidence = f"{r.confidence:.2f}" if r.confidence is not None else "-"
        print(f"{r.case_name:<30} {status:<10} {strategy:<18} {confidence:<10}")
        if not r.correct and r.error:
            print(f"    -> {r.error}")

    print("-" * 70)
    print(f"Accuracy: {correct}/{total} ({accuracy:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the labeled JSON dataset.")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit with a non-zero status if accuracy falls below this threshold (0.0-1.0). "
        "Useful as a CI regression gate.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    cases = load_dataset(args.dataset)
    results = asyncio.run(run_benchmark(cases))
    print_report(results)

    accuracy = sum(1 for r in results if r.correct) / len(results) if results else 0.0
    if accuracy < args.min_accuracy:
        print(f"\nAccuracy {accuracy:.1%} is below the required threshold {args.min_accuracy:.1%}.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
