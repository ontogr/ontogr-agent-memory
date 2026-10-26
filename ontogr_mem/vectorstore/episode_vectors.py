"""Write a vector for every episode, for the ingestion hand-off."""

from collections.abc import Sequence

from ontogr_mem.data_models import Episode
from ontogr_mem.vectorstore.models import EmbedItem, VectorKind
from ontogr_mem.vectorstore.writer import VectorWriter


async def write_episode_vectors(
    writer: VectorWriter, episodes: Sequence[Episode]
) -> None:
    """Write one vector per episode, from the text of its content.

    The call is the hand-off of the ingestion pipeline. A typed failure is
    kept by the writer instead of raised, because the pipeline never retries
    a failed hand-off.

    Args:
        writer: The writer that stores the vectors.
        episodes: The episodes that were written.
    """
    items = [EmbedItem(episode.id, episode.text) for episode in episodes]
    if not items:
        return
    await writer.write(VectorKind.EPISODE, items)
