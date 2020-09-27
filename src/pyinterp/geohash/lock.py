"""
Lock handling used to synchronize resources
-------------------------------------------
"""
from typing import Optional, Union
import abc
import os
import pathlib
import sys
import threading
import time


class LockError(Exception):
    """Exception thrown by this module"""
    pass


class Lock:
    """Handle a lock file by opening the file in exclusive mode: if the file
    already exists, access to this file will fail.

    Args:
        path (str): Path to the lock
    """
    OPEN_MODE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_TRUNC

    def __init__(self, path: str) -> None:
        self.path = path
        self.stream = None

    def __getstate__(self):
        return self.path

    def __setstate__(self, path):
        self.__init__(path)

    def acquire(self,
                timeout: Optional[float] = None,
                delay: Optional[float] = None):
        """Acquire a lock

        Args:
            timeout (float, optional): Maximum timeout for a lock acquisition.
            delay (float, optional): Waiting time between attempts.

        Raises:
            LockError: If a lock has not been obtained before the specified
                timeout
        """
        end_time = time.time() + (timeout or sys.maxsize)
        delay = delay or 0.1
        while True:
            try:
                self.stream = os.open(self.path, self.OPEN_MODE)
                return
            except (IOError, OSError):
                if time.time() > end_time:
                    raise LockError
                else:
                    time.sleep(delay)

    def locked(self) -> bool:
        """Test the existence of the lock.

        Returns:
            bool: True if the lock exists
        """
        return self.stream is not None

    def release(self) -> None:
        """Release the lock."""
        if self.stream is not None:
            os.close(self.stream)
            self.stream = None
        try:
            os.remove(self.path)
        # The file is already deleted and that's what we want.
        except OSError:
            pass

    def __enter__(self) -> bool:
        self.acquire()
        return True

    def __exit__(self, *args) -> None:
        if self.locked():
            self.release()


class Synchronizer(abc.ABC):  # pragma: no cover
    """Interface of Synchronizer"""
    @abc.abstractclassmethod
    def __enter__(self) -> bool:
        ...

    @abc.abstractclassmethod
    def __exit__(self, t, v, tb) -> None:
        ...


class PuppetSynchronizer(Synchronizer):
    """Provides synchronization using thread locks."""
    def __enter__(self) -> bool:
        return True

    def __exit__(self, t, v, tb) -> None:
        pass


class ThreadSynchronizer(Synchronizer):
    """Provides synchronization using thread locks."""
    def __init__(self):
        self.lock = threading.Lock()

    def __enter__(self) -> bool:
        return self.lock.acquire()

    def __exit__(self, t, v, tb) -> None:
        self.lock.release()


class ProcessSynchronizer(Synchronizer):
    """Provides synchronization using file locks

    Args:
        path (pathlib.Path, str): The file used for locking/unlocking
    """
    def __init__(self,
                 path: Union[pathlib.Path, str],
                 timeout: Optional[float] = None):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self.path = path
        self.lock = Lock(str(path))
        self.timeout = timeout

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} {self.path!r}")

    def __enter__(self) -> bool:
        self.lock.acquire(self.timeout)
        return True

    def __exit__(self, t, v, tb) -> None:
        self.lock.release()
