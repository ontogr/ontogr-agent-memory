"""Episode store on the graph backends."""

from collections.abc import Sequence
from datetime import datetime

from ontogr_mem.data_models import Episode, to_micros
from ontogr_mem.data_models.episode import EpisodeMeta
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.episode_rows import (
    episode_from_row,
    episode_meta_from_row,
    episode_meta_params,
    episode_params,
)
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.protocol import ScopedRunner


WRITE_BATCH = 500

EPISODE_COLUMNS = """\
e.id AS id, e.kind AS kind, e.occurred_at AS occurred_at,
e.ingested_at AS ingested_at, e.content AS content,
e.content_format AS content_format, e.metadata AS metadata,
e.content_hash AS content_hash, e.temporal_status AS temporal_status,
e.temporal_committed_hash AS temporal_committed_hash,
s.system AS source_system, s.ref AS source_ref, s.connector AS source_connector,
s.description AS source_description,
a.id AS actor_id, a.role AS actor_role, a.display_name AS actor_name"""


class GraphEpisodeStore:
    """`EpisodeStore` over a tenant's graph.

    Args:
        runner: Runner bound to the tenant's graph.
        tenant_keyed: Set explicitly only when a custom runner does not expose
            its backend scope.
    """

    def __init__(
        self, runner: ScopedRunner, *, tenant_keyed: bool | None = None
    ) -> None:
        """Bind the store to a runner."""
        self._runner = runner
        self._scope = (
            TenantScope.from_mode(tenant_keyed)
            if tenant_keyed is not None
            else runner.scope
        )

    def _node(self, name: str, label: str, id_param: str) -> str:
        return (
            f"({quote(name, 'variable')}:{quote(label, 'label')} "
            f"{self._scope.pattern(f'id: {id_param}')})"
        )

    async def get_content_hash(self, episode_id: str) -> str | None:
        """Return the recorded content hash, or None if never fully ingested."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('e', 'Episode', '$id')} "
                "RETURN e.content_hash AS content_hash",
                {"id": episode_id},
                read_only=True,
                idempotent=True,
                tag="episode_hash",
            )
        )
        return rows.scalar("content_hash")

    async def write_episodes(self, episodes: Sequence[Episode]) -> None:
        """Idempotently write episodes with their sources and actors."""
        statement = "\n".join(
            [
                "UNWIND $batch AS item",
                f"MERGE (e:Episode {self._scope.pattern('id: item.id')})",
                "ON CREATE SET e.ingested_at = item.ingested_at",
                "SET e.kind = item.kind, e.occurred_at = item.occurred_at,",
                "    e.content = item.content,",
                "    e.content_format = item.content_format,",
                "    e.metadata = item.metadata",
                f"MERGE (s:Source {self._scope.pattern('id: item.source_id')})",
                "SET s.system = item.source_system, s.ref = item.source_ref,",
                "    s.connector = item.source_connector,",
                "    s.description = item.source_description",
                f"MERGE (a:Actor {self._scope.pattern('id: item.actor_id')})",
                "SET a.role = item.actor_role, a.display_name = item.actor_name",
                f"MERGE (e)-[source:FROM_SOURCE {self._scope.edge()}]->(s)",
                self._scope.stamp_assignment("source"),
                f"MERGE (e)-[actor:BY_ACTOR {self._scope.edge()}]->(a)",
                self._scope.stamp_assignment("actor"),
            ]
        )
        for start in range(0, len(episodes), WRITE_BATCH):
            batch = [
                episode_params(episode)
                for episode in episodes[start : start + WRITE_BATCH]
            ]
            await self._runner.run(
                QuerySpec(
                    statement, {"batch": batch}, idempotent=True, tag="episode_write"
                )
            )

    async def record_content_hash(self, episode_id: str, content_hash: str) -> None:
        """Record the content hash."""
        await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('e', 'Episode', '$id')} SET e.content_hash = $hash",
                {"id": episode_id, "hash": content_hash},
                idempotent=True,
                tag="episode_hash_record",
            )
        )

    async def remove_episode(self, episode_id: str) -> None:
        """Delete an episode, its metadata and unused sources and actors."""
        await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('m', 'EpisodeMeta', '$id')} DETACH DELETE m",
                {"id": episode_id},
                idempotent=True,
                tag="episode_meta_remove",
            )
        )
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('e', 'Episode', '$id')} "
                f"OPTIONAL MATCH (e)-[:FROM_SOURCE {self._scope.edge()}]->(s) "
                f"OPTIONAL MATCH (e)-[:BY_ACTOR {self._scope.edge()}]->(a) "
                "WITH e, s.id AS source_id, a.id AS actor_id "
                "DETACH DELETE e RETURN source_id, actor_id",
                {"id": episode_id},
                idempotent=True,
                tag="episode_remove",
            )
        )
        for label, edge, column in (
            ("Source", "FROM_SOURCE", "source_id"),
            ("Actor", "BY_ACTOR", "actor_id"),
        ):
            for row in rows.rows:
                if row[column] is None:
                    continue
                await self._runner.run(
                    QuerySpec(
                        f"MATCH {self._node('n', label, '$nid')} "
                        f"WHERE NOT (n)<-[:{edge} {self._scope.edge()}]-() DELETE n",
                        {"nid": row[column]},
                        idempotent=True,
                        tag="episode_orphan",
                    )
                )

    async def write_episode_meta(self, meta: EpisodeMeta) -> None:
        """Replace the session, thread, subject reference and fact ids.

        Args:
            meta: Metadata for an episode in this tenant.

        Raises:
            ValueError: The episode does not exist in this tenant.
        """
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('e', 'Episode', '$id')} "
                f"MERGE {self._node('m', 'EpisodeMeta', '$id')} "
                "SET m.session_id = $session_id, m.thread = $thread, "
                "m.subject_ref = $subject_ref, m.fact_ids = $fact_ids "
                "RETURN m.id AS id",
                episode_meta_params(meta),
                idempotent=True,
                tag="episode_meta_write",
            )
        )
        if not rows.rows:
            msg = "unknown episode"
            raise ValueError(msg)

    async def get_episode_meta(self, episode_id: str) -> EpisodeMeta | None:
        """Return the episode metadata, or None if none is stored."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('m', 'EpisodeMeta', '$id')} "
                "RETURN m.id AS id, m.session_id AS session_id, "
                "m.thread AS thread, m.subject_ref AS subject_ref, "
                "m.fact_ids AS fact_ids",
                {"id": episode_id},
                read_only=True,
                idempotent=True,
                tag="episode_meta_get",
            )
        )
        return episode_meta_from_row(rows.rows[0]) if rows.rows else None

    async def episode_ids_by_subject(self, subject_ref: str) -> list[str]:
        """Return episode ids whose metadata has this subject reference."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:EpisodeMeta {self._scope.pattern()}) "
                "WHERE m.subject_ref = $subject_ref RETURN m.id AS id ORDER BY m.id",
                {"subject_ref": subject_ref},
                read_only=True,
                idempotent=True,
                tag="episode_ids_by_subject",
            )
        )
        return [row["id"] for row in rows.rows]

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Return the episode, or None if it does not exist."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH {self._node('e', 'Episode', '$id')}"
                f"-[:FROM_SOURCE {self._scope.edge()}]->(s), "
                f"(e)-[:BY_ACTOR {self._scope.edge()}]->(a) "
                f"RETURN {EPISODE_COLUMNS}",
                {"id": episode_id},
                read_only=True,
                idempotent=True,
                tag="episode_get",
            )
        )
        return episode_from_row(rows.rows[0]) if rows.rows else None

    async def get_episodes(self, ids: Sequence[str]) -> list[Episode]:
        """Return the requested episodes, ordered by id.

        Ids that no episode carries are left out. An empty list of ids returns
        an empty list and runs no statement.
        """
        unique = list(dict.fromkeys(ids))
        if not unique:
            return []
        found: dict[str, Episode] = {}
        for start in range(0, len(unique), WRITE_BATCH):
            rows = await self._runner.run(
                QuerySpec(
                    f"MATCH (e:Episode {self._scope.pattern()}) "
                    "WHERE e.id IN $ids "
                    f"MATCH (e)-[:FROM_SOURCE {self._scope.edge()}]->(s), "
                    f"(e)-[:BY_ACTOR {self._scope.edge()}]->(a) "
                    f"RETURN {EPISODE_COLUMNS}",
                    {"ids": unique[start : start + WRITE_BATCH]},
                    read_only=True,
                    idempotent=True,
                    tag="episode_get_episodes",
                )
            )
            for row in rows.rows:
                episode = episode_from_row(row)
                found[episode.id] = episode
        return [found[key] for key in sorted(found)]

    async def list_episodes(
        self,
        *,
        limit: int,
        before: tuple[datetime, str] | None = None,
        session_id: str | None = None,
    ) -> list[Episode]:
        """Return newest episodes, filtering by metadata when requested."""
        conditions = []
        params: dict[str, object] = {"limit": limit}
        if before is not None:
            conditions.append(
                "(e.ingested_at < $before_at OR "
                "(e.ingested_at = $before_at AND e.id < $before_id))"
            )
            params.update(
                before_at=to_micros(before[0]),
                before_id=before[1],
            )
        if session_id is not None:
            conditions.append("m.session_id = $session_id")
            params["session_id"] = session_id
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:Episode {self._scope.pattern()}) "
                "MATCH (e)-[:FROM_SOURCE "
                f"{self._scope.edge()}]->(s), "
                f"(e)-[:BY_ACTOR {self._scope.edge()}]->(a) "
                f"OPTIONAL MATCH (m:EpisodeMeta "
                f"{self._scope.pattern('id: e.id')}) "
                f"WITH e, s, a, m {where} "
                f"RETURN {EPISODE_COLUMNS} ORDER BY e.ingested_at DESC, e.id DESC "
                "LIMIT $limit",
                params,
                read_only=True,
                idempotent=True,
                tag="episode_list",
            )
        )
        return [episode_from_row(row) for row in rows.rows]

    async def latest_before(
        self, before: datetime, limit: int, *, exclude_id: str | None = None
    ) -> Sequence[Episode]:
        """Return up to `limit` fully ingested episodes at or before `before`.

        The result is ordered oldest first.
        """
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:Episode {self._scope.pattern()})"
                " WHERE e.occurred_at <= $before AND e.id <> $exclude"
                " AND e.content_hash IS NOT NULL "
                f"MATCH (e)-[:FROM_SOURCE {self._scope.edge()}]->(s), "
                f"(e)-[:BY_ACTOR {self._scope.edge()}]->(a) "
                f"RETURN {EPISODE_COLUMNS} "
                "ORDER BY e.occurred_at DESC, e.ingested_at DESC, e.id DESC "
                "LIMIT $limit",
                {
                    "before": to_micros(before),
                    "exclude": exclude_id or "",
                    "limit": limit,
                },
                read_only=True,
                idempotent=True,
                tag="episode_latest",
            )
        )
        return [episode_from_row(row) for row in reversed(rows.rows)]

    async def before(self, episode: Episode, limit: int) -> Sequence[Episode]:
        """Return the episodes that came before `episode`, oldest first."""
        return await self.latest_before(
            episode.occurred_at, limit, exclude_id=episode.id
        )
