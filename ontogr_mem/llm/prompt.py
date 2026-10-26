"""Prompt blocks that keep untrusted text out of the instructions.

Every delimiter the extraction prompts use is listed in ``TAGS``. A block
neutralizes every tag in ``TAGS`` inside its body, not just the tag it wraps.
Text that closes a block cannot escape into the instructions.
"""

import re


TAGS = (
    "actor",
    "entities",
    "entity_types",
    "episode_text",
    "instructions",
    "language",
    "previous_episodes",
    "previous_reply",
    "reference_time",
    "relation_types",
    "validation_errors",
)
"""Every delimiter a prompt block can use."""

# The longest names come first, so a tag that starts like another one cannot
# cut the match short.
_ALTERNATIVES = "|".join(sorted(TAGS, key=len, reverse=True))
_DELIMITER = re.compile(rf"<\s*(/?)\s*({_ALTERNATIVES})", re.IGNORECASE)


def fence(tag: str, body: str) -> str:
    """Wrap `body` in `<tag>` and neutralize every delimiter inside it.

    The match ignores case and allows whitespace around the slash, so
    `< /EPISODE_TEXT>` is rewritten too. Every tag in `TAGS` is rewritten, not
    only the one being wrapped, so text cannot close a block it does not belong
    to. Other angle brackets are left alone.

    Args:
        tag: The name of the block, without angle brackets.
        body: The untrusted text that goes inside the block.

    Returns:
        The block, with its delimiters neutralized.
    """
    safe = _DELIMITER.sub(r"&lt;\1\2", body)
    return f"<{tag}>\n{safe}\n</{tag}>"
