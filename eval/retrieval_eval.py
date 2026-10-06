"""Score retrieval against the golden question set: is the right page in the top 5?

Run: uv run python -m eval.retrieval_eval --check   (validate the question file; no keys, no network)
     uv run python -m eval.retrieval_eval --list    (show each question with a link to its PDF page)
     uv run python -m eval.retrieval_eval           (score it; needs VOYAGE_API_KEY, PINECONE_API_KEY, an ingested index)

Exit codes: 0 passed, 1 error or invalid question file, 2 scored but below the 80% target.
Every question carries an `evidence` quote copied from its answer page, and --check opens the real PDF to
confirm it, so a wrong page or an invented answer is caught by the machine, not by trust.
"""
import argparse
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError

from app.rag.cli import normalize_dashes
from app.rag.errors import RetrievalError
from app.rag.ingest import SOURCES_DIR
from app.rag.pdf_loader import Page, load_pages
from app.rag.retrieval import retrieve
from app.rag.types import NonEmptyText, RetrievedChunk, SourceName

QUESTIONS_FILE = Path(__file__).resolve().parents[1] / "data" / "retrieval_eval.yaml"
MIN_QUESTIONS = 20
MIN_PER_SOURCE = 5
COPY_WARNING_WORDS = 4  # a question repeating this many words in a row from its page is probably too easy


class EvalSetError(Exception):
    """The question file is missing or wrong."""


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo like `page:` must not pass silently

    question: NonEmptyText
    source: SourceName
    pages: list[PositiveInt]  # printed pages; any of them counts as a hit
    evidence: NonEmptyText  # short quote copied word for word from the answer page
    # Optional short answer in the question's own language, written from the cited page. It is a human aid and a
    # later reference (Phases 3 and 7); it is NOT WHO text and nothing checks it, so a reader must review it.
    reference_answer: str = ""
    note: str = ""

    def model_post_init(self, __context: object) -> None:
        if not self.pages:
            raise ValueError("pages must list at least one page")


def _normalize(text: str) -> str:
    """Lowercase words only, so case, spacing, punctuation and ligatures never hide a match."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def _dehyphenate(text: str) -> str:
    return re.sub(r"-\s*\n\s*", "", text)


def load_questions(path: Path) -> list[EvalQuestion]:
    if not path.exists():
        raise EvalSetError(f"Question file not found: {path}. See the header of data/retrieval_eval.yaml.")
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as error:
        raise EvalSetError(f"{path.name} is not valid YAML: {error}") from error
    if not isinstance(data, dict) or not isinstance(data.get("questions"), list):
        raise EvalSetError(f"{path.name} must contain a `questions:` list")

    questions: list[EvalQuestion] = []
    first_seen: dict[str, int] = {}
    for number, raw in enumerate(data["questions"], start=1):
        try:
            question = EvalQuestion.model_validate(raw)
        except ValidationError as error:
            details = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc']) or 'entry'}: {problem['msg']}"
                for problem in error.errors()
            )
            raise EvalSetError(f"Entry {number}: {details}") from error
        key = _normalize(question.question)
        if key in first_seen:
            raise EvalSetError(f"Entry {number} repeats the question of Entry {first_seen[key]}")
        first_seen[key] = number
        questions.append(question)
    return questions


# --- checking the set against the real PDFs -------------------------------------------------------------


@dataclass
class CheckResult:
    problems: list[str] = field(default_factory=list)  # stop the run
    warnings: list[str] = field(default_factory=list)  # worth fixing, do not stop the run


def _quote_found(quote: str, pages: list[Page]) -> bool:
    needle = f" {_normalize(quote)} "
    for page in pages:
        for text in (page.text, _dehyphenate(page.text)):
            if needle in f" {_normalize(text)} ":
                return True
    return False


def _longest_shared_run(question: str, pages: list[Page]) -> list[str]:
    """The longest run of words that appears, in order, in both the question and one of the pages."""
    words = _normalize(question).split()
    best: list[str] = []
    for page in pages:
        page_words = _normalize(page.text).split()
        lengths = [0] * (len(page_words) + 1)
        for word in words:
            previous = lengths
            lengths = [0] * (len(page_words) + 1)
            for j, page_word in enumerate(page_words, start=1):
                if word == page_word:
                    lengths[j] = previous[j - 1] + 1
                    if lengths[j] > len(best):
                        best = page_words[j - lengths[j] : j]
    return best


def check_set(
    questions: list[EvalQuestion],
    sources_dir: Path,
    min_questions: int = MIN_QUESTIONS,
    min_per_source: int = MIN_PER_SOURCE,
) -> CheckResult:
    result = CheckResult()
    if len(questions) < min_questions:
        result.problems.append(f"Need at least {min_questions} questions, found {len(questions)}")
    for source in get_args(SourceName):
        count = sum(1 for q in questions if q.source == source)
        if count < min_per_source:
            result.problems.append(f"'{source}' has {count} questions; at least {min_per_source} are needed")

    pages_by_source: dict[str, dict[int, Page]] = {}
    for source in sorted({q.source for q in questions}):
        try:
            pages_by_source[source] = {p.page: p for p in load_pages(sources_dir / f"{source}.pdf")}
        except RetrievalError as error:
            result.problems.append(f"Cannot check '{source}': {error}")

    for number, q in enumerate(questions, start=1):
        pages = pages_by_source.get(q.source)
        if pages is None:
            continue
        listed: list[Page] = []
        for page_number in q.pages:
            if page_number in pages:
                listed.append(pages[page_number])
            else:
                result.problems.append(
                    f"Entry {number}: page {page_number} is not a page of '{q.source}' (the index holds printed "
                    f"pages {min(pages)} to {max(pages)}; the page may be a divider or front matter)"
                )
        if not listed:
            continue
        if not _quote_found(q.evidence, listed):
            result.problems.append(
                f"Entry {number}: the evidence quote was not found on page(s) {q.pages} of '{q.source}'"
            )
        run = _longest_shared_run(q.question, listed)
        if len(run) >= COPY_WARNING_WORDS:
            result.warnings.append(
                f"Entry {number}: the question repeats \"{' '.join(run)}\" from the guide; reword it"
            )
    return result


# --- scoring ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Miss:
    question: str
    source: str
    expected_pages: list[int]
    retrieved: list[RetrievedChunk]


@dataclass
class EvalReport:
    k: int
    total: int
    hits: int
    by_source: dict[str, tuple[int, int]]  # source -> (hits, total)
    misses: list[Miss]
    first_hit_ranks: list[int | None]  # per question, in file order; None = not in the top k

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def passed(self) -> bool:
        return self.total > 0 and self.hits * 5 >= self.total * 4  # at least 80%, without float rounding

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "total": self.total,
            "hits": self.hits,
            "hit_rate": self.hit_rate,
            "passed": self.passed,
            "by_source": {source: list(counts) for source, counts in self.by_source.items()},
            "first_hit_ranks": self.first_hit_ranks,
            "misses": [
                {
                    "question": miss.question,
                    "source": miss.source,
                    "expected_pages": miss.expected_pages,
                    "retrieved": [{"source": r.source, "page": r.page, "score": r.score} for r in miss.retrieved],
                }
                for miss in self.misses
            ],
        }


def _is_hit(question: EvalQuestion, chunk: RetrievedChunk) -> bool:
    return chunk.source == question.source and chunk.page in question.pages


def evaluate(
    questions: list[EvalQuestion],
    retrieve_fn: Callable[[str, int], list[RetrievedChunk]],
    k: int = 5,
) -> EvalReport:
    """A RetrievalError from retrieve_fn is not caught: an outage says nothing about retrieval quality."""
    hits = 0
    by_source: dict[str, list[int]] = {}
    misses: list[Miss] = []
    ranks: list[int | None] = []
    for question in questions:
        results = retrieve_fn(question.question, k)[:k]
        rank = next((i for i, chunk in enumerate(results, start=1) if _is_hit(question, chunk)), None)
        ranks.append(rank)
        counts = by_source.setdefault(question.source, [0, 0])
        counts[1] += 1
        if rank is None:
            misses.append(Miss(question.question, question.source, question.pages, results))
        else:
            hits += 1
            counts[0] += 1
    return EvalReport(k, len(questions), hits, {s: (c[0], c[1]) for s, c in by_source.items()}, misses, ranks)


def format_report(report: EvalReport) -> str:
    verdict = "PASS" if report.passed else "FAIL"
    lines = [
        f"hit@{report.k}: {report.hits}/{report.total} ({report.hit_rate:.0%})  {verdict} (target: at least 80%)"
    ]
    lines += [f"  {source}: {hits}/{total}" for source, (hits, total) in report.by_source.items()]
    for miss in report.misses:
        got = ", ".join(f"{r.source} p.{r.page} ({r.score:.2f})" for r in miss.retrieved) or "nothing"
        pages = ", ".join(f"p.{p}" for p in miss.expected_pages)
        lines += [f'MISS  "{miss.question}"', f"      expected: {miss.source} {pages}", f"      got: {got}"]
    return "\n".join(lines)


# --- the command line ----------------------------------------------------------------------------------


def _print_list(questions: list[EvalQuestion], sources_dir: Path) -> None:
    pdf_pages: dict[str, dict[int, int]] = {}
    for source in sorted({q.source for q in questions}):
        try:
            pdf_pages[source] = {p.page: p.pdf_page for p in load_pages(sources_dir / f"{source}.pdf")}
        except RetrievalError as error:
            print(f"WARNING: {error}", file=sys.stderr)
    for number, q in enumerate(questions, start=1):
        print(f"#{number}  {q.source}")
        print(f"    Q: {q.question}")
        print(f'    Quote: "{q.evidence}"')
        if q.reference_answer:
            print(f"    Reference answer: {q.reference_answer}")
        pdf_file = (sources_dir / f"{q.source}.pdf").resolve()
        for page in q.pages:
            pdf_page = pdf_pages.get(q.source, {}).get(page)
            if pdf_page is None:
                print(f"    p.{page} -> not found in the PDF")
            else:
                print(f"    p.{page} -> PDF page {pdf_page}: {pdf_file.as_uri()}#page={pdf_page}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score retrieval against the golden question set.")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--sources-dir", type=Path, default=SOURCES_DIR)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--check", action="store_true", help="validate the question file only; no keys, no network")
    parser.add_argument("--list", action="store_true", help="print each question with a link to its PDF page")
    parser.add_argument("--informational", action="store_true", help="skip the size rules and never fail the exit code")
    parser.add_argument("--save", type=Path, help="write the result as JSON (eval/results/ is gitignored)")
    return parser.parse_args(normalize_dashes(sys.argv[1:] if argv is None else argv))


def main(argv: list[str] | None = None) -> int:
    loader_log = logging.getLogger("app.rag.pdf_loader")
    previous_level = loader_log.level
    loader_log.setLevel(logging.ERROR)  # the loader logs every skipped page; keep the output readable
    try:
        return _run(_parse_args(argv))
    finally:
        loader_log.setLevel(previous_level)  # do not leak the setting into other code in this process


def _run(args: argparse.Namespace) -> int:
    try:
        questions = load_questions(args.questions)
        if args.list:
            _print_list(questions, args.sources_dir)
            return 0

        size_rules = {"min_questions": 1, "min_per_source": 0} if args.informational else {}
        result = check_set(questions, args.sources_dir, **size_rules)
        for warning in result.warnings:
            print(f"WARNING: {warning}", file=sys.stderr)
        if result.problems:
            for problem in result.problems:
                print(f"PROBLEM: {problem}", file=sys.stderr)
            return 1
        if args.check:
            print(f"OK: {len(questions)} questions, every page and quote found in the PDFs")
            return 0

        report = evaluate(questions, retrieve, args.k)
    except (EvalSetError, RetrievalError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(format_report(report))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(report.to_dict(), indent=2))
    if args.informational:
        return 0
    return 0 if report.passed else 2


if __name__ == "__main__":
    sys.exit(main())
