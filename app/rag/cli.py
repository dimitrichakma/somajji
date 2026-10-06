"""Small helper shared by the command-line tools."""

# Em dash, en dash, figure dash, horizontal bar, Unicode hyphen, non-breaking hyphen, minus sign.
_TYPOGRAPHIC_DASHES = "‐‑‒–—―−"


def _fix_one(arg: str) -> str:
    run = 0
    while run < len(arg) and (arg[run] == "-" or arg[run] in _TYPOGRAPHIC_DASHES):
        run += 1
    leading, rest = arg[:run], arg[run:]
    # Only change an argument that starts with a typographic dash and then a letter (a flag name).
    # A lone dash, a dash before a digit, and a dash inside a word are left alone.
    if not any(char in _TYPOGRAPHIC_DASHES for char in leading) or not rest or not rest[0].isalpha():
        return arg
    flag_name = rest.split("=", 1)[0]
    return ("--" if len(flag_name) > 1 else "-") + rest


def normalize_dashes(argv: list[str]) -> list[str]:
    """Undo the 'smart dashes' of chat, notes and editor apps.

    Copying `--list` through such an app often turns the two hyphens into one long dash. It looks the
    same but the shell passes a different character, and argparse says "unrecognized arguments".
    """
    return [_fix_one(arg) for arg in argv]
