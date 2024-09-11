import os
import time
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import posix_ipc

DEFAULT_VERBOSE_LEVEL = 2


def isPosixCompliant():
    # Check os.name for 'posix'
    if os.name == "posix":
        return True

    # Additional checks for POSIX systems
    # On Windows, os.name is 'nt' and platform.system() is 'Windows'
    system = platform.system()
    if system in ["Linux", "Darwin", "FreeBSD", "Unix"]:
        return True

    return False


class SharedMemoryHelper:
    def __init__(self, name: str, shape: Tuple[int, ...], verbose: int = -1):
        if not isPosixCompliant():
            raise Exception("This program requires a POSIX-compliant operating system")
        self.name: str = name
        self.shape: Tuple[int, ...] = shape
        self.verbose: int = verbose if verbose >= 0 else DEFAULT_VERBOSE_LEVEL
        # self.verbose: int = DEFAULT_VERBOSE_LEVEL

        self._shm: Optional[shared_memory.SharedMemory] = None
        self._sem: Optional[posix_ipc.Semaphore] = None
        self._array: Optional[np.ndarray] = None

        self.ownShm: bool = False
        self.ownSem: bool = False
        # cache
        self.prevProbedArray_cache: Optional[np.ndarray] = None

    def shmExists(self) -> bool:
        try:
            shm = shared_memory.SharedMemory(name=self.name, create=False)
            shm.close()
            return True
        except FileNotFoundError:
            return False

    def createShm(self):
        if self.arraySize == 0:
            return
        size = int(np.prod(self.shape) * np.dtype(np.float32).itemsize)
        if self.shmExists():
            shared_memory.SharedMemory(name=self.name, create=False).unlink()
            self.vPrint(f"Clear preexisting shared memory {self.name}", 1)
        self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=size)
        self.ownShm = True
        self.vPrint(f"Created shared memory {self.name} with shape {self.shape}", 1)

    def connectShm(self) -> shared_memory.SharedMemory:
        if self.arraySize == 0:
            return
        timeCnt = 0
        waitTime = 0.1
        while True:
            try:
                self._shm = shared_memory.SharedMemory(name=self.name, create=False)
                self.vPrint(f"\nConnected to shared memory {self.name}", 1)
                self.ownShm = False
                break
            except FileNotFoundError:
                self.vPrint(
                    f"\rWaiting for shared memory {self.name} ... [ {timeCnt:.1f} seconds ]",
                    1,
                    end="",
                )
                time.sleep(waitTime)
                timeCnt += waitTime

        return self._shm

    def clearShm(self):
        if self.shmExists() and self._shm is not None:
            self.array[:] = 0.0

    def semExists(self) -> bool:
        try:
            sem = posix_ipc.Semaphore(
                self.semName,
                flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
                mode=0o666,
                initial_value=0,
            )
            sem.close()
            sem.unlink()
            return False
        except posix_ipc.ExistentialError:
            return True

    def createSem(self):
        if self.arraySize == 0:
            return
        if self.semExists():
            posix_ipc.Semaphore(self.semName, flags=0).unlink()
            self.vPrint(f"Clear preexisting semaphore {self.semName}", 1)
        self._sem = posix_ipc.Semaphore(
            self.semName,
            flags=posix_ipc.O_CREAT | posix_ipc.O_EXCL,
            mode=0o666,
            initial_value=0,
        )
        self.ownSem = True
        self.vPrint(f"Created semaphore {self.semName}", 1)

    def connectSem(self) -> posix_ipc.Semaphore:
        if self.arraySize == 0:
            return
        timeCnt = 0
        waitTime = 0.1
        while True:
            if self.semExists():
                self._sem = posix_ipc.Semaphore(
                    self.semName, flags=posix_ipc.O_RDWR, mode=0o666
                )
                self.ownSem = False
                self.vPrint(f"\nConnected to Semaphore {self.semName}", 1)
                break
            else:
                self.vPrint(
                    f"\rWaiting for Semaphore {self.semName} ... [ {timeCnt:.1f} seconds ]",
                    1,
                    end="",
                )
                time.sleep(waitTime)
                timeCnt += waitTime
        return self._sem

    def clearSem(self) -> None:
        while self.sem.value > 0:
            self.sem.acquire()

    def createTunnel(self) -> None:
        self.createShm()
        self.createSem()

    def connectTunnel(self) -> None:
        self.connectShm()
        self.connectSem()

    @property
    def semName(self) -> str:
        return self.name + "_sem"

    @property
    def sem(self) -> posix_ipc.Semaphore:
        if self._sem is None:
            return self.connectSem()
        return self._sem

    @property
    def shm(self) -> shared_memory.SharedMemory:
        if self._shm is None:
            return self.connectShm()
        return self._shm

    @property
    def buffer(self) -> memoryview:
        return self.shm.buf

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.ndarray(self.shape, dtype=np.float32, buffer=self.buffer)
        return self._array

    @property
    def arraySize(self) -> int:
        return int(np.prod(self.shape))

    def __del__(self):
        try:
            self.close()
            self.unlink()
        except Exception as e:
            pass
            # self.vPrint(f"Exception during __del__: {e}", 1)

    def __str__(self):
        return f"SharedMemoryHolder({self.name}, {self.shape})"

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shm"] = None
        state["_sem"] = None
        state["_array"] = None
        if self.arraySize != 0:
            state["prev_array"] = self.array.copy()
            state["prev_sem"] = self.probeSem()
        for k, v in state.items():
            if k.endswith("_cache"):
                state[k] = None
        return state

    def __setstate__(self, state: Dict):
        prev_array = state.pop("prev_array", None)
        prev_sem = state.pop("prev_sem", None)
        self.__dict__.update(state)
        if self.ownShm:
            self.createShm()
        else:
            self.connectShm()
        if self.ownSem:
            self.createSem()
        else:
            self.connectSem()
        # Recover the data
        if prev_array is not None and prev_sem is not None:
            np.copyto(self.array, prev_array)
            for i in range(prev_sem):
                self.sem.release()

    def __getitem__(self, index) -> float:
        return float(self.array[index])

    def __setitem__(self, index, value) -> None:
        array = self.probeArray()
        array[index] = float(value)
        self.sendArray(array)

    def vPrint(self, content, verbose=2, **kwargs):
        if self.verbose >= verbose:
            print(content, **kwargs)

    def fetchArray(
        self, consume=1, timeout: float = 1000.0, interruptCallback=None
    ) -> np.ndarray:
        if self.shm is None or self.sem is None:
            raise ValueError("Shared memory or semaphore not initialized")
        self.vPrint(f"Fetching array from {self.name}", 2)
        timeCnt = 0
        while timeCnt < timeout:
            if interruptCallback is not None and interruptCallback():
                self.vPrint("Fetch from {} interrupted".format(self.name), 1)
                raise InterruptedError
            if self.sem.value >= consume:
                for i in range(consume):
                    self.sem.acquire()  # Consume a certain number of sem
                self.vPrint(
                    f"\nFetched array from {self.name}: {list(self.array)}, Sem {self.sem.value+1}->{self.sem.value}",
                    2,
                )
                break
            else:
                time.sleep(0.001)
                timeCnt += 0.001
                self.vPrint(
                    f"\rFetching shared memory {self.name} ... [ {timeCnt:.1f} seconds ]",
                    2,
                    end="",
                )
        if timeCnt >= timeout:
            errStr = f"Timeout while waiting to acquire semaphore {self.semName}"
            self.vPrint(errStr, 1)
            raise TimeoutError(errStr)
        return self.array.copy()

    def sendArray(self, array: Union[np.ndarray, List[float]], offset: int = 0):
        if self.shm is None or self.sem is None:
            raise ValueError("Shared memory or semaphore not initialized")
        self.vPrint(f"Sending array in {self.name}", 2)
        if offset == 0 and len(array) == len(self.array):
            np.copyto(self.array, array)
        else:
            for idx, value in enumerate(array):
                self.array[offset + idx] = value
        self.sem.release()
        self.vPrint(
            f"Sent array in {self.name}: {list(array)}, Sem: {self.sem.value-1}->{self.sem.value}",
            2,
        )

    def probeSem(self) -> int:
        return int(self.sem.value)

    def probeArray(self, timeout: float = 1000.0) -> np.ndarray:
        if self.shm is None or self.sem is None:
            raise ValueError("Shared memory or semaphore not initialized")
        timeCnt = 0
        while timeCnt < timeout:
            if self.sem.value > 0:
                self.vPrint("", 2)
                if (
                    self.prevProbedArray_cache is not None
                    and (self.prevProbedArray_cache != self.array).any()
                ):
                    self.vPrint(
                        f"Probed array from {self.name}: {list(self.array)}, Sem: {self.sem.value}",
                        2,
                    )
                break
            else:
                time.sleep(0.001)
                timeCnt += 0.001
        if timeCnt >= timeout:
            self.vPrint(f"Timeout while waiting for semaphore {self.semName}", 1)
            raise TimeoutError(f"Timeout while waiting for semaphore {self.semName}")
        self.prevProbedArray_cache = self.array.copy()
        return self.prevProbedArray_cache

    def close(self):
        if self._shm is not None:
            self._shm.close()
        if self._sem is not None:
            self._sem.close()

    def unlink(self):
        if self._shm is not None and self.ownShm:
            self._shm.unlink()
        if self._sem is not None and self.ownSem:
            self._sem.unlink()
