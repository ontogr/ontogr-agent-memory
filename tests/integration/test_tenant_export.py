"""Tenant export rows and deterministic JSONL output on the in-memory twins."""

import json
from dataclasses import replace

from ontogr_mem.data_models import EntityStatus
from ontogr_mem.export import export_tenant
from ontogr_mem.temporal.reader import FactReader
from tests.support.episodes import make_episode
from tests.support.p9i import alias, entity, make_world
from tests.unit.temporal.helpers import at, obs


class TestTenantExport:
    """Export each tenant-owned domain row without losing structured fields."""

    async def test_rows_are_sorted_complete_and_repeatable(self):
        """Repeated fixed-clock exports preserve content, order and UTC times."""
        episodes, resolver, store, clock, engine = await make_world(
            ("ana", "absorbed", "archived")
        )
        await resolver.write_entity_batch(
            [
                entity("absorbed").model_copy(
                    update={
                        "status": EntityStatus.MERGED,
                        "merged_into": "ana",
                    }
                ),
                entity("archived").model_copy(
                    update={"status": EntityStatus.ARCHIVED}
                ),
            ],
            [
                alias("ana", "Ana A", "alias-2"),
                alias("ana", "Ana", "alias-1"),
            ],
        )
        episode_rows = [
            make_episode("ep-1", content="Ana lives in Berlin."),
            make_episode(
                "ep-2",
                content={"event": "move", "city": "Munich"},
                minutes=1,
                metadata={"category": "profile"},
            ),
            make_episode("ep-3", content="Ana worked at Acme.", minutes=2),
        ]
        await episodes.write_episodes(episode_rows)
        first = replace(
            obs(
                "o-berlin",
                "ana",
                at(2020),
                subject="absorbed",
                slot="residence",
                episode="ep-1",
            ),
            statement="Ana lives in Berlin.",
            quote="Ana lives in Berlin.",
            source_system="test",
        )
        second = replace(
            obs(
                "o-munich",
                "archived",
                at(2025),
                subject="absorbed",
                slot="residence",
                episode="ep-2",
            ),
            statement="Ana lives in Munich.",
            quote="Ana lives in Munich.",
            source_system="test",
        )
        withdrawn = replace(
            obs(
                "o-work",
                "archived",
                at(2022),
                subject="ana",
                slot="employer",
                episode="ep-3",
            ),
            statement="Ana worked at Acme.",
            quote="Ana worked at Acme.",
            source_system="test",
        )
        await engine.ingest_observations([first])
        clock.value = clock.value.replace(day=23)
        await engine.ingest_observations([second, withdrawn])
        await engine.retract_episodes(["ep-3"])

        first_export = await export_tenant(
            store,
            resolver,
            episodes,
            FactReader(store),
            tenant_id="tenant-1",
            now=clock.value,
        )
        second_export = await export_tenant(
            store,
            resolver,
            episodes,
            FactReader(store),
            tenant_id="tenant-1",
            now=clock.value,
        )
        rows = [json.loads(line) for line in first_export.jsonl.splitlines()]

        assert first_export.jsonl == second_export.jsonl
        assert rows[0] == {
            "exported_at": "2026-10-23T00:00:00.000000+00:00",
            "schema": 1,
            "tenant_id": "tenant-1",
            "type": "meta",
        }
        assert [row["id"] for row in rows if row["type"] == "entity"] == [
            "absorbed",
            "ana",
            "archived",
        ]
        fact_rows = [row for row in rows if row["type"] == "fact"]
        assert [(row["id"], row["version"]) for row in fact_rows] == sorted(
            (row["id"], row["version"]) for row in fact_rows
        )
        tombstone = next(row for row in fact_rows if row["status"] == "withdrawn")
        assert tombstone["closed_by"] is None
        assert tombstone["episode_ids"] == []
        assert tombstone["valid_to"] is None
        episodes_by_id = {row["id"]: row for row in rows if row["type"] == "episode"}
        assert episodes_by_id["ep-2"]["content"] == {
            "event": "move",
            "city": "Munich",
        }
        assert episodes_by_id["ep-2"]["metadata"] == {"category": "profile"}
        assert all(
            value is None or (isinstance(value, str) and value.endswith("+00:00"))
            for row in fact_rows
            for key, value in row.items()
            if key in {"valid_from", "valid_to", "recorded_at", "expired_at"}
        )
        assert first_export.counts == {
            "entity": 3,
            "episode": 3,
            "fact": len(fact_rows),
            "meta": 1,
        }
