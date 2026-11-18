"""Temporal walk behavior across events and non-actual facts."""

from ontogr_mem.data_models.facts import FactKind, Modality
from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.walk import derive
from tests.unit.temporal.helpers import at, obs


class TestWalkIsolation:
    """Keep event and non-actual facts out of state replacement."""

    def test_events_do_not_replace_actual_states(self):
        """Leave current state open after an event."""
        state = obs("state", "acme", at(2010))
        event = obs("event", "globex", at(2015), fact_kind=FactKind.EVENT)

        result = derive("slot", [state, event], Cardinality.ONE)

        assert len(result.runs) == 2
        assert all(run.valid_to is None for run in result.runs)

    def test_non_actual_states_do_not_replace_actual_states(self):
        """Leave current state open after a planned fact."""
        state = obs("state", "acme", at(2010))
        plan = obs("plan", "globex", at(2015), modality=Modality.PLANNED)

        result = derive("slot", [state, plan], Cardinality.ONE)

        assert len(result.runs) == 2
        assert all(run.valid_to is None for run in result.runs)

    def test_actual_states_still_replace_earlier_states(self):
        """Close an earlier state when a later actual state arrives."""
        first = obs("first", "acme", at(2010))
        second = obs("second", "globex", at(2015))

        result = derive("slot", [first, second], Cardinality.ONE)

        assert len(result.runs) == 2
        assert next(
            run for run in result.runs if run.value_id == "acme"
        ).valid_to == at(2015)
