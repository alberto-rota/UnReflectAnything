"""
MultiDataset implementation for combining multiple datasets.
"""

import random
from typing import List, Optional, Tuple, Union

import torch
from torch.utils.data import ConcatDataset

from logger import get_logger

from .base import Mono3D_Dataset

logger = get_logger(__name__).set_context("MULTI_DATASET")


class MultiDataset(ConcatDataset):
    """
    Extends the ConcatDataset adding custom methods for inspection and summarization.

    This class combines multiple Mono3D_Dataset instances into a single dataset with
    unified sampling, curriculum learning, and inspection capabilities.
    """

    def __init__(
        self, set_of_datasets: List[Mono3D_Dataset], shuffle: bool = True
    ) -> None:
        """
        Initialize the MultiDataset.

        Args:
            set_of_datasets: List of Mono3D_Dataset instances
            shuffle: Whether to shuffle the combined sampler
        """
        super().__init__(set_of_datasets)

        # Store aggregate statistics
        self.numvideos = sum([dataset.numvideos for dataset in set_of_datasets])
        self.numframes = sum([dataset.numframes for dataset in set_of_datasets])
        self.shuffle = shuffle

        # Calculate proportion of videos and frames from each dataset
        self.fracvideos = {}
        for dataset in self.datasets:
            if self.numvideos == 0:
                self.fracvideos[dataset.name] = 0
            else:
                self.fracvideos[dataset.name] = dataset.numvideos / self.numvideos

        self.fracframes = {}
        for dataset in self.datasets:
            if self.numframes == 0:
                self.fracframes[dataset.name] = 0
            else:
                self.fracframes[dataset.name] = dataset.numframes / self.numframes

        self.lens = [len(dataset) for dataset in self.datasets]

        # Create a combined sampler from all datasets
        self._create_combined_sampler()

        # Store curriculum learning information
        self.max_steps_frameskip = (
            max([len(dataset.frameskip_set) for dataset in self.datasets])
            if self.datasets
            else 0
        )

    def _create_combined_sampler(self) -> None:
        """Create a combined sampler from all datasets."""
        multi_sampler = []
        offset = 0
        for dataset in self.datasets:
            # Add the current dataset's sampler indices with the appropriate offset
            multi_sampler.extend([s + offset for s in dataset.sampler])
            # Update the offset for the next dataset
            offset += len(dataset)

        if self.shuffle:
            random.shuffle(multi_sampler)
        self.sampler = multi_sampler

    def _get_ds_from_idx(self, idx: int) -> Union[Tuple[int, int], Tuple[None, None]]:
        """
        Get the dataset index and the local index within that dataset from the global index.

        Args:
            idx: The global index across all datasets.

        Returns:
            A tuple containing:
                - int: The index of the dataset in the `self.datasets` list.
                - int: The local index within the identified dataset.
                If the global index is out of range, returns (None, None).
        """
        for d, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                return d, idx
            idx -= len(dataset)  # Decrement by the size of the current dataset
        return None, None

    def inspect(self, idx: Optional[int] = None) -> None:
        """
        Inspect an item at the specified global index within the datasets.

        Args:
            idx: The global index to inspect. If None, inspect a random item.
        """
        dsidx, localidx = self._get_ds_from_idx(idx)
        self.datasets[dsidx].inspect(localidx)

    def step_frameskip_curriculum(self):
        """Advance the frameskip curriculum step for all datasets."""
        for dataset in self.datasets:
            dataset.step_frameskip_curriculum()

    def __getitem__(self, idx: int) -> object:
        """
        Get an item at the specified global index within the datasets.

        Args:
            idx (int): The global index across all datasets.

        Returns:
            object: The item at the specified global index.
        """
        dsidx, localidx = self._get_ds_from_idx(idx)
        return self.datasets[dsidx][localidx]

    def reset_sampler(self):
        """Reset the sampler for all datasets and recreate the combined sampler."""
        for dataset in self.datasets:
            dataset.reset_sampler()
        self._create_combined_sampler()

    def samplesummary(self):
        """
        Print a summary of a single sample from the dataset.

        Includes shape and statistics for the target, source, and transformation tensors.
        """
        try:
            from utilities import sp  # Import formatting function
        except ImportError:
            # Define a simple fallback if the original function isn't available
            sp = lambda shape: f"{shape}"

        # Extract a sample from the dataset
        sample = next(iter(self))
        framestack, Ts2t = sample["framestack"], sample["Ts2t"]

        # Ensure proper dimensionality
        if len(framestack.shape) == 4:
            framestack = framestack.unsqueeze(0)

        # Get source and target frames
        source, target = framestack[0, :-1, ...], framestack[0, -1, ...]

        # Extract shape dimensions
        CHANNELS, HEIGHT, WIDTH = target.shape

        # Print summary statistics
        logger.info(
            f"Sample target shape: {sp(target.shape)} - Range: [{torch.min(target):.2f} - {torch.max(target):.2f}] "
            f"{torch.mean(target):.2f}\u00b1{torch.std(target):.2f}"
        )
        logger.info(
            f"Sample source shape: {sp(source.shape)} - Range: [{torch.min(source):.2f} - {torch.max(source):.2f}] "
            f"{torch.mean(source):.2f}\u00b1{torch.std(source):.2f}"
        )
        logger.info(
            f"Sample Ts2t shape   : {sp(Ts2t.shape)} - Range: [{torch.min(Ts2t):.2f} - {torch.max(Ts2t):.2f}] "
            f"{torch.mean(Ts2t):.2f}\u00b1{torch.std(Ts2t):.2f}"
        )
