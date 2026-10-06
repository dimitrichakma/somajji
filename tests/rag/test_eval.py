import json
import logging
from pathlib import Path

import pytest
import yaml

from app.rag.errors import RetrievalError
from app.rag.types import RetrievedChunk
from eval import retrieval_eval
from eval.retrieval_eval import (
    EvalSetError,
    check_set,
    evaluate,
    load_questions,
    main,
)
from tests.rag.test_pdf_loader import build_pdf

# Each guide gets 3 pages. Page n of a guide has one known sentence, used as the evidence quote.
SENTENCES = {
    ("pfa", 1): "Stay with the person and keep them away from further harm.",
    ("pfa", 2): "Offer water and ask about basic needs such as food and shelter.",
    ("pfa", 3): "Listen carefully without pushing the person to talk about the event.",
    ("mhgap", 1): "Assess whether the person has thoughts of harming themselves.",
    ("mhgap", 2): "Arrange follow up within two weeks for every person you assess.",
    ("mhgap", 3): "Involve the family when the person agrees to their help.",
    ("pmplus", 1): "Slow breathing can help a person who feels very tense.",
    ("pmplus", 2): "Break a big problem into smaller steps that can be managed.",
    ("pmplus", 3): "Plan one pleasant activity for each day of the week.",
}


@pytest.fixture
def sources_dir(tmp_path) -> Path:
    directory = tmp_path / "sources"
    directory.mkdir()
    for source in ("pfa", "mhgap", "pmplus"):
        pages = [[(300, SENTENCES[(source, n)], 11), (810, str(n), 9)] for n in (1, 2, 3)]
        build_pdf(directory / f"{source}.pdf", pages)
    return directory


def entry(source: str = "pfa", page: int = 1, question: str | None = None, **overrides) -> dict:
    values = {
        "question": question or f"A volunteer question about {source} page {page}",
        "source": source,
        "pages": [page],
        "evidence": SENTENCES.get((source, page), "A quote that exists nowhere"),
    }
    values.update(overrides)
    return values


def valid_set() -> list[dict]:
    """20 questions: 7 pfa, 7 mhgap, 6 pmplus, each with a distinct question text."""
    counts = {"pfa": 7, "mhgap": 7, "pmplus": 6}
    return [
        entry(source, i % 3 + 1, question=f"Question number {source} {i} please")
        for source, count in counts.items()
        for i in range(count)
    ]


def write_file(tmp_path: Path, entries) -> Path:
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump({"questions": entries}, allow_unicode=True))
    return path


def chunk(source: str, page: int, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(text="some text", source=source, page=page, pdf_page=page + 2, section="", score=score)


def questions_from(entries: list[dict], tmp_path: Path):
    return load_questions(write_file(tmp_path, entries))


# --- loading the file -----------------------------------------------------------------------------


def test_a_good_file_loads(tmp_path):
    [question] = questions_from([entry(pages=[1, 2])], tmp_path)
    assert (question.source, question.pages) == ("pfa", [1, 2])
    assert question.evidence == SENTENCES[("pfa", 1)]


@pytest.mark.parametrize(
    "broken, field",
    [
        ({"pages": []}, "pages"),
        ({"pages": [0]}, "pages"),
        ({"pages": [-3]}, "pages"),
        ({"source": "webmd"}, "source"),
        ({"question": "  "}, "question"),
        ({"evidence": ""}, "evidence"),
        ({"page": 3}, "page"),  # a typo for "pages" is an unknown field
    ],
)
def test_a_broken_entry_is_reported_with_its_number_and_field(tmp_path, broken, field):
    entries = [entry(question="Entry one is fine"), {**entry(question="Entry two is broken"), **broken}]
    with pytest.raises(EvalSetError, match=r"Entry 2") as error:
        questions_from(entries, tmp_path)
    assert field in str(error.value)


def test_a_missing_field_is_reported(tmp_path):
    broken = entry()
    del broken["evidence"]
    with pytest.raises(EvalSetError, match=r"Entry 1.*evidence"):
        questions_from([broken], tmp_path)


def test_a_duplicate_question_is_reported_with_both_numbers(tmp_path):
    entries = [entry(question="Same question"), entry(question="  same   QUESTION ")]
    with pytest.raises(EvalSetError, match=r"Entry 2.*Entry 1"):
        questions_from(entries, tmp_path)


def test_a_reference_answer_is_optional_and_kept_when_given(tmp_path):
    with_answer = entry(question="Has an answer", reference_answer="A short plain answer.")
    without = entry(question="Has none")
    first, second = questions_from([with_answer, without], tmp_path)
    assert first.reference_answer == "A short plain answer."
    assert second.reference_answer == ""


def test_a_reference_answer_never_changes_the_page_or_quote_checks(tmp_path, sources_dir):
    entries = valid_set()
    entries[0]["reference_answer"] = "Anything here, even words that are nowhere in the guide."
    assert check_set(questions_from(entries, tmp_path), sources_dir).problems == []


def test_a_missing_file_says_where_it_should_be(tmp_path):
    with pytest.raises(EvalSetError, match="not found"):
        load_questions(tmp_path / "nope.yaml")


@pytest.mark.parametrize("content", ["questions: 5", "other: []", "- just\n- a list", ": : :"])
def test_a_file_with_the_wrong_shape_is_reported(tmp_path, content):
    path = tmp_path / "eval.yaml"
    path.write_text(content)
    with pytest.raises(EvalSetError):
        load_questions(path)


# --- scoring ---------------------------------------------------------------------------------------


def run(entries, answers, tmp_path, k=5):
    """answers: question -> list of (source, page) the fake search returns."""
    questions = questions_from(entries, tmp_path)
    return evaluate(questions, lambda q, _k: [chunk(s, p) for s, p in answers[q]], k)


def test_a_hit_needs_the_same_guide_and_a_listed_page(tmp_path):
    entries = [entry("pfa", 2, question="q-hit"), entry("pfa", 2, question="q-wrong-guide"), entry("pfa", 2, question="q-wrong-page")]
    answers = {
        "q-hit": [("mhgap", 1), ("pfa", 2)],
        "q-wrong-guide": [("mhgap", 2), ("pmplus", 2)],  # right page number, wrong guide
        "q-wrong-page": [("pfa", 1), ("pfa", 3)],  # right guide, wrong page
    }
    report = run(entries, answers, tmp_path)
    assert (report.hits, report.total) == (1, 3)
    assert [miss.question for miss in report.misses] == ["q-wrong-guide", "q-wrong-page"]


def test_any_of_several_listed_pages_counts(tmp_path):
    report = run([entry("pfa", 2, question="q", pages=[2, 3])], {"q": [("pfa", 3)]}, tmp_path)
    assert report.hits == 1


def test_only_the_top_k_results_count(tmp_path):
    answers = {"q": [("pfa", 1), ("pfa", 1), ("pfa", 3), ("pfa", 3), ("pfa", 3), ("pfa", 2)]}  # the hit is 6th
    assert run([entry("pfa", 2, question="q")], answers, tmp_path, k=5).hits == 0
    assert run([entry("pfa", 2, question="q")], answers, tmp_path, k=6).hits == 1


@pytest.mark.parametrize("hits, passed", [(17, True), (16, True), (15, False), (0, False), (20, True)])
def test_the_target_is_80_percent(tmp_path, hits, passed):
    entries = [entry("pfa", 1, question=f"q{i}") for i in range(20)]
    answers = {f"q{i}": [("pfa", 1 if i < hits else 3)] for i in range(20)}
    report = run(entries, answers, tmp_path)
    assert report.hits == hits and report.passed is passed


def test_the_rate_uses_the_real_total_not_a_fixed_twenty(tmp_path):
    entries = [entry("pfa", 1, question=f"q{i}") for i in range(10)]
    answers = {f"q{i}": [("pfa", 1 if i < 8 else 3)] for i in range(10)}
    report = run(entries, answers, tmp_path)
    assert report.hit_rate == 0.8 and report.passed


def test_a_miss_shows_what_was_expected_and_what_came_back(tmp_path):
    report = run([entry("pfa", 2, question="q")], {"q": [("mhgap", 1), ("pfa", 3)]}, tmp_path)
    [miss] = report.misses
    assert (miss.source, miss.expected_pages) == ("pfa", [2])
    assert [(r.source, r.page) for r in miss.retrieved] == [("mhgap", 1), ("pfa", 3)]


def test_the_rank_of_the_first_hit_and_the_per_guide_rates_are_recorded(tmp_path):
    entries = [entry("pfa", 2, question="a"), entry("pfa", 2, question="b"), entry("mhgap", 1, question="c")]
    answers = {"a": [("pfa", 1), ("pfa", 2)], "b": [("pfa", 3)], "c": [("mhgap", 1)]}
    report = run(entries, answers, tmp_path)
    assert report.first_hit_ranks == [2, None, 1]
    assert report.by_source == {"pfa": (1, 2), "mhgap": (1, 1)}


def test_a_search_failure_stops_the_run_instead_of_counting_as_a_miss(tmp_path):
    def broken(question, k):
        raise RetrievalError("Voyage embedding failed: down")

    with pytest.raises(RetrievalError, match="Voyage"):
        evaluate(questions_from([entry()], tmp_path), broken, 5)


# --- checking the set ---------------------------------------------------------------------------------


def test_a_valid_set_has_no_problems(tmp_path, sources_dir):
    result = check_set(questions_from(valid_set(), tmp_path), sources_dir)
    assert result.problems == []


def test_too_few_questions_is_a_problem(tmp_path, sources_dir):
    result = check_set(questions_from(valid_set()[:12], tmp_path), sources_dir)
    assert any("20" in problem and "12" in problem for problem in result.problems)


def test_a_guide_with_too_few_questions_is_a_problem(tmp_path, sources_dir):
    entries = [entry("pfa", i % 3 + 1, question=f"pfa question {i}") for i in range(17)]
    entries += [entry("mhgap", 1, question=f"mhgap question {i}") for i in range(3)]
    result = check_set(questions_from(entries, tmp_path), sources_dir)
    problems = " ".join(result.problems)
    assert "mhgap" in problems and "pmplus" in problems


def test_a_page_that_does_not_exist_is_a_problem(tmp_path, sources_dir):
    entries = valid_set()
    entries[0] = entry("pfa", 1, question="Question about a missing page", pages=[99])
    result = check_set(questions_from(entries, tmp_path), sources_dir)
    assert any("Entry 1" in p and "99" in p and "pfa" in p for p in result.problems)


def test_evidence_that_is_not_on_the_listed_pages_is_a_problem(tmp_path, sources_dir):
    entries = valid_set()
    entries[0] = entry("pfa", 1, question="Wrong page for the quote", evidence=SENTENCES[("pfa", 3)])
    result = check_set(questions_from(entries, tmp_path), sources_dir)
    assert any("Entry 1" in p and "quote" in p for p in result.problems)


def test_evidence_found_on_any_listed_page_is_fine(tmp_path, sources_dir):
    entries = valid_set()
    entries[0] = entry("pfa", 1, question="Quote on the second listed page", pages=[1, 3], evidence=SENTENCES[("pfa", 3)])
    assert check_set(questions_from(entries, tmp_path), sources_dir).problems == []


def test_quote_matching_ignores_case_spacing_and_punctuation(tmp_path, sources_dir):
    entries = valid_set()
    messy = "  STAY with the   person, and keep them away from further harm  "
    entries[0] = entry("pfa", 1, question="Messy quote", evidence=messy)
    assert check_set(questions_from(entries, tmp_path), sources_dir).problems == []


def test_a_line_break_hyphen_in_the_pdf_does_not_hide_a_quote(tmp_path):
    directory = tmp_path / "sources"
    directory.mkdir()
    for source in ("pfa", "mhgap", "pmplus"):
        lines = [(300, "Please arrange follow-", 11), (320, "up within two weeks for every person.", 11), (810, "1", 9)]
        build_pdf(directory / f"{source}.pdf", [lines])
    entries = [entry("pfa", 1, question="Hyphenated", evidence="arrange followup within two weeks")]
    result = check_set(questions_from(entries, tmp_path), directory, min_questions=1, min_per_source=0)
    assert result.problems == []


def test_a_missing_pdf_is_a_problem_with_the_download_hint(tmp_path, sources_dir):
    (sources_dir / "mhgap.pdf").unlink()
    result = check_set(questions_from(valid_set(), tmp_path), sources_dir)
    assert any("mhgap" in p and "download_sources" in p for p in result.problems)


def test_a_question_that_copies_four_words_from_its_page_gets_a_warning_not_a_problem(tmp_path, sources_dir):
    entries = valid_set()
    entries[0] = entry("pfa", 1, question="How do I keep them away from further harm today")
    result = check_set(questions_from(entries, tmp_path), sources_dir)
    assert result.problems == []
    assert any("Entry 1" in w and "away from further harm" in w for w in result.warnings)


def test_a_paraphrased_question_gets_no_warning(tmp_path, sources_dir):
    entries = valid_set()
    entries[0] = entry("pfa", 1, question="He is shaking after the accident, how can I protect him")
    assert not any("Entry 1" in w for w in check_set(questions_from(entries, tmp_path), sources_dir).warnings)


# --- the command line -----------------------------------------------------------------------------------


@pytest.fixture
def no_search(monkeypatch):
    def must_not_run(question, k=5):
        raise AssertionError("search ran but should not have")

    monkeypatch.setattr(retrieval_eval, "retrieve", must_not_run)


def test_check_needs_no_search_and_returns_zero_for_a_good_set(tmp_path, sources_dir, no_search, capsys):
    path = write_file(tmp_path, valid_set())
    assert main(["--check", "--questions", str(path), "--sources-dir", str(sources_dir)]) == 0
    assert "OK" in capsys.readouterr().out


def test_a_pasted_long_dash_works_like_the_double_hyphen(tmp_path, sources_dir, no_search, capsys):
    path = write_file(tmp_path, valid_set())
    argv = ["\u2014check", "\u2014questions", str(path), "\u2014sources-dir", str(sources_dir)]
    assert main(argv) == 0
    assert "OK" in capsys.readouterr().out


def test_a_real_unknown_flag_is_still_rejected(tmp_path, sources_dir, no_search):
    with pytest.raises(SystemExit):
        main(["--nonsense"])


def test_check_returns_one_and_prints_the_problems(tmp_path, sources_dir, no_search, capsys):
    path = write_file(tmp_path, valid_set()[:5])
    assert main(["--check", "--questions", str(path), "--sources-dir", str(sources_dir)]) == 1
    assert "20" in capsys.readouterr().err


def test_a_scored_run_prints_the_result_and_exits_zero_on_pass(tmp_path, sources_dir, monkeypatch, capsys):
    entries = valid_set()
    path = write_file(tmp_path, entries)
    right = {e["question"]: [(e["source"], e["pages"][0])] for e in entries}
    monkeypatch.setattr(retrieval_eval, "retrieve", lambda q, k=5: [chunk(*right[q][0])])
    assert main(["--questions", str(path), "--sources-dir", str(sources_dir)]) == 0
    output = capsys.readouterr().out
    assert "hit@5: 20/20" in output and "PASS" in output


def test_a_scored_run_exits_two_below_target_and_lists_the_misses(tmp_path, sources_dir, monkeypatch, capsys):
    entries = valid_set()
    path = write_file(tmp_path, entries)
    monkeypatch.setattr(retrieval_eval, "retrieve", lambda q, k=5: [chunk("pfa", 3)])  # always the same wrong-ish page
    assert main(["--questions", str(path), "--sources-dir", str(sources_dir)]) == 2
    output = capsys.readouterr().out
    assert "FAIL" in output and "MISS" in output


def test_informational_mode_skips_size_rules_and_never_fails_the_exit_code(tmp_path, sources_dir, monkeypatch, capsys):
    path = write_file(tmp_path, [entry("pfa", 1, question="Only one question here")])
    monkeypatch.setattr(retrieval_eval, "retrieve", lambda q, k=5: [chunk("mhgap", 2)])
    assert main(["--informational", "--questions", str(path), "--sources-dir", str(sources_dir)]) == 0
    assert "hit@5: 0/1" in capsys.readouterr().out


def test_the_checks_run_before_any_search_is_paid_for(tmp_path, sources_dir, no_search, capsys):
    entries = valid_set()
    entries[0]["evidence"] = "A sentence that is nowhere in the guide at all"
    path = write_file(tmp_path, entries)
    assert main(["--questions", str(path), "--sources-dir", str(sources_dir)]) == 1  # no_search proves no call was made


def test_save_writes_the_result_as_json(tmp_path, sources_dir, monkeypatch):
    entries = valid_set()
    path = write_file(tmp_path, entries)
    monkeypatch.setattr(retrieval_eval, "retrieve", lambda q, k=5: [chunk("pfa", 3)])
    out = tmp_path / "result.json"
    main(["--questions", str(path), "--sources-dir", str(sources_dir), "--save", str(out)])
    saved = json.loads(out.read_text())
    assert saved["total"] == 20 and saved["k"] == 5 and "misses" in saved and "by_source" in saved
    assert saved["passed"] is False


def test_a_search_failure_in_a_run_returns_one_with_the_message(tmp_path, sources_dir, monkeypatch, capsys):
    def broken(question, k=5):
        raise RetrievalError("PINECONE_API_KEY is not set")

    monkeypatch.setattr(retrieval_eval, "retrieve", broken)
    path = write_file(tmp_path, valid_set())
    assert main(["--questions", str(path), "--sources-dir", str(sources_dir)]) == 1
    assert "PINECONE_API_KEY" in capsys.readouterr().err


def test_list_shows_the_pdf_page_the_quote_and_a_link_that_opens_it(tmp_path, sources_dir, capsys):
    # build_pdf pages are numbered 1, 2, 3 in the margin; the PDF position equals the printed number here.
    path = write_file(tmp_path, [entry("pfa", 2, question="Where is the water page", pages=[2])])
    assert main(["--list", "--questions", str(path), "--sources-dir", str(sources_dir)]) == 0
    output = capsys.readouterr().out
    assert "Where is the water page" in output
    assert SENTENCES[("pfa", 2)] in output
    assert "p.2" in output and "PDF page 2" in output
    assert (sources_dir / "pfa.pdf").resolve().as_uri() + "#page=2" in output


def test_list_shows_the_reference_answer_when_there_is_one(tmp_path, sources_dir, capsys):
    path = write_file(
        tmp_path,
        [
            entry("pfa", 1, question="Has an answer", reference_answer="A short plain answer."),
            entry("pfa", 2, question="Has none"),
        ],
    )
    assert main(["--list", "--questions", str(path), "--sources-dir", str(sources_dir)]) == 0
    output = capsys.readouterr().out
    assert output.count("Reference answer:") == 1
    assert "Reference answer: A short plain answer." in output


def test_the_question_text_is_not_logged(tmp_path, sources_dir, monkeypatch, caplog):
    path = write_file(tmp_path, valid_set())
    monkeypatch.setattr(retrieval_eval, "retrieve", lambda q, k=5: [chunk("pfa", 1)])
    with caplog.at_level(logging.DEBUG):
        main(["--questions", str(path), "--sources-dir", str(sources_dir)])
    assert "Question number" not in caplog.text
