import pytest

from app.rag.cli import normalize_dashes


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("—list", "--list"),  # em dash: what chat and notes apps turn "--" into
        ("–list", "--list"),  # en dash
        ("‑‑list", "--list"),  # two non-breaking hyphens
        ("‐‐dry-run", "--dry-run"),  # two Unicode hyphens, hyphen inside the name kept
        ("−−check", "--check"),  # two minus signs
        ("—questions=data/x.yaml", "--questions=data/x.yaml"),  # a flag with a value
        ("–h", "-h"),  # a single-letter flag keeps one hyphen
        ("-—list", "--list"),  # a mix of ASCII and typographic
    ],
)
def test_typographic_dashes_become_normal_hyphens(typed, expected):
    assert normalize_dashes([typed]) == [expected]


@pytest.mark.parametrize(
    "untouched",
    [
        "--list",
        "-h",
        "--questions=data/x.yaml",
        "list",
        "data/x.yaml",
        "-",
        "--",
        "—",  # a lone long dash is not a flag
        "—5",  # a long dash before a digit is a value, not a flag
        "3.5",
        "",
        "a—b",  # a dash in the middle of a word stays
    ],
)
def test_everything_else_is_left_alone(untouched):
    assert normalize_dashes([untouched]) == [untouched]


def test_a_whole_command_line_is_fixed_argument_by_argument():
    argv = ["—check", "--questions", "data/my—file.yaml", "–k", "5"]
    assert normalize_dashes(argv) == ["--check", "--questions", "data/my—file.yaml", "-k", "5"]


def test_the_input_list_is_not_changed():
    argv = ["—list"]
    normalize_dashes(argv)
    assert argv == ["—list"]
