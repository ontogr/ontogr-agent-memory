"""Pure timeline derivation for one slot.

``derive`` is a function of the slot's whole observation set, so the valid-time
result never depends on arrival order. Events at one instant run in this order:
judged closes, assertions, then stated ends.

Walk rules:

* an assertion of a value that has an open run joins that run;
* otherwise it opens a run and, in a single-valued slot, cuts every other open
  run that is not tied with it;
* a run whose last member carries a ``valid_to`` is closed there before a later
  event, so a later repeat opens a new run;
* a stated end closes the open run of its value, or is reported as unmatched.

``SlotDirectives`` is extra input from the contradiction handler. Excluded
observations are dropped before the walk, ``no_cut`` groups do not cut each
other, and a ``Close`` ends the run that holds its anchor observation.

"Value" below means the side of the fact that varies inside its slot: the
object, or the subject of an inverse slot.

The types live in ``ontogr_mem.temporal.timeline`` and the walk in
``ontogr_mem.temporal.walk``. They are gathered here so the component keeps
one import path.
"""

from ontogr_mem.temporal.timeline import (  # noqa: F401
    Close,
    Conflict,
    ConflictKind,
    Derived,
    Run,
    SlotDirectives,
)
from ontogr_mem.temporal.walk import derive  # noqa: F401
