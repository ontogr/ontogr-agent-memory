"""Operating-system locks for memory store ownership."""

from pathlib import Path
import os


class ServerLock:
    """Hold an exclusive lock on a path until release or process exit."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._descriptor: int | None = None

    def acquire(self) -> None:
        """Acquire the lock without waiting."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        """Release the lock if acquired."""
        if self._descriptor is None:
            return
        descriptor, self._descriptor = self._descriptor, None
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
