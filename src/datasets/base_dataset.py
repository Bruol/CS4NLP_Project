from abc import ABC, abstractmethod
from typing import Iterator, Dict

class BaseDataset(ABC):
    """
    Abstract base class for all datasets.
    """

    def __init__(self, dataset_name: str):
        """
        Initializes the dataset.

        Args:
            dataset_name (str): The name of the dataset.
        """
        self.dataset_name = dataset_name

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, any]]:
        """
        Returns an iterator over the samples in the dataset.

        Yields:
            Iterator[Dict[str, any]]: A dictionary representing a single sample.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        pass

    def __str__(self) -> str:
        return f"Dataset({self.dataset_name})" 