from typing import Dict, Iterable, List, Tuple
from .SharedMemoryHelper import SharedMemoryHelper


class SharedMemoryManager:
    def __init__(
        self,
        name="",
        config: Iterable[Tuple[str, Tuple[int, ...], int]] = [],
        noPrefix: bool = False,
    ):
        self.name = name
        self.noPrefix = noPrefix
        self.sharedMemoryMap: Dict[str, SharedMemoryHelper] = {}
        if config:
            for item in config:
                self.newShm(*item)

    def __getitem__(self, key: str):
        return self.get(key)

    def newShm(self, shmName: str, shape: Tuple[int, ...], verbose: int):
        full_name = f"{self.name}_{shmName}" if not self.noPrefix else shmName
        self.sharedMemoryMap[shmName] = SharedMemoryHelper(full_name, shape, verbose)

    def createTunnels(self, shmList: List[str] = None):
        if shmList is None:
            shmList = self.keys()
        for shmName in shmList:
            if shmName in self.sharedMemoryMap:
                self.sharedMemoryMap[shmName].createTunnel()

    def connectTunnels(self, shmList: List[str] = None):
        if shmList is None:
            shmList = self.keys()
        for shmName in shmList:
            if shmName in self.sharedMemoryMap:
                self.sharedMemoryMap[shmName].connectTunnel()

    def clear(self, shmList: List[str] = None):
        if shmList is None:
            shmList = self.keys()
        for shmName in shmList:
            if shmName in self.sharedMemoryMap:
                self.sharedMemoryMap[shmName].clearSem()
                self.sharedMemoryMap[shmName].clearShm()

    def close(self, shmList: List[str] = None):
        if shmList is None:
            shmList = self.keys()
        for shmName in shmList:
            if shmName in self.sharedMemoryMap:
                self.sharedMemoryMap[shmName].close()

    def unlink(self, shmList: List[str] = None):
        if shmList is None:
            shmList = self.keys()
        for shmName in shmList:
            if shmName in self.sharedMemoryMap:
                self.sharedMemoryMap[shmName].unlink()

    def get(self, name: str):
        if name in self.sharedMemoryMap:
            return self.sharedMemoryMap[name]
        else:
            raise RuntimeError(f"Shared memory segment not found: {name}")

    def keys(self):
        return list(self.sharedMemoryMap.keys())
