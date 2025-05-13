from abc import ABC, abstractmethod


class Dataset(ABC):
    def __init__(self, root_dir: str):
        self.root_dir: str = root_dir
        self.samples = []

    @abstractmethod
    def traverse_data(self) -> None:
        raise NotImplementedError
