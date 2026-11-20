"""Graph-store-only bootstrap lifecycle behavior."""

from unittest.mock import AsyncMock, Mock

from ontogr_mem.service import bootstrap


async def test_open_graph_store_closes_store_after_context(monkeypatch):
    """Close the process-owned graph adapter after leaving the context."""
    store = Mock(close=AsyncMock())
    settings = Mock(graph_url=Mock(get_secret_value=lambda: "falkor://localhost"))
    monkeypatch.setattr(bootstrap, "ServiceSettings", lambda: settings)
    monkeypatch.setattr(bootstrap, "parse_graph_url", lambda _url: object())
    monkeypatch.setattr(bootstrap, "_graph_stores", lambda _target: (store, None))

    async with bootstrap.open_graph_store() as actual:
        assert actual is store
        store.close.assert_not_awaited()
    store.close.assert_awaited_once_with()
