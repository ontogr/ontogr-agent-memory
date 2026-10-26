"""A TCP proxy a test can stop and restart to simulate a server outage."""

import asyncio
import contextlib


class TcpProxy:
    """Forwards connections from a local port to a target host and port."""

    def __init__(self, target_host: str, target_port: int) -> None:
        """Remember the target; call `start` to listen."""
        self._target = (target_host, target_port)
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self.port = 0

    async def start(self, port: int = 0) -> None:
        """Listen on `port`; 0 picks a free port, kept for later restarts."""
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", port)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Stop listening and drop every open connection."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for writer in list(self._writers):
            writer.close()
        self._writers.clear()
        if server is not None:
            await server.wait_closed()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        upstream_reader, upstream_writer = await asyncio.open_connection(*self._target)
        self._writers.update({writer, upstream_writer})
        await asyncio.gather(
            self._pipe(reader, upstream_writer),
            self._pipe(upstream_reader, writer),
        )

    async def _pipe(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(ConnectionError, asyncio.CancelledError):
            while data := await reader.read(65536):
                writer.write(data)
                await writer.drain()
        writer.close()
