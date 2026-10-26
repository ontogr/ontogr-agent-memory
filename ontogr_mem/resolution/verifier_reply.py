"""The answer the model sends back.

A reply holds one line per pair, in order, and the line ends in YES or NO. The
position of the line names the pair it answers, so a model cannot address a
pair it was not asked about. A line without a verdict leaves its pair
unanswered.
"""

import re


_LINE = re.compile(r"^\s*(\d+)\s*[.)]\s*(.*)$")
_VERDICT = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def parse_reply(text: str, count: int) -> tuple[dict[int, bool | None], int]:
    """Map pair position (0-based) to YES/NO, or ``None`` if the line has none.

    Returns the map and the number of lines whose position is outside ``1..count``.
    A lone pair may be answered with a bare YES or NO.
    """
    out: dict[int, bool | None] = {}
    invalid = 0
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if not 1 <= n <= count:
            invalid += 1
            continue
        segment = m.group(2).split("|")[-1]
        v = _VERDICT.search(segment)
        if n - 1 not in out:
            out[n - 1] = None if v is None else v.group(1).upper() == "YES"
    if count == 1 and not out:
        v = _VERDICT.fullmatch(text.strip())
        if v:
            out[0] = v.group(1).upper() == "YES"
    return out, invalid
