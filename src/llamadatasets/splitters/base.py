"""
Dataset splitters for dividing datasets into train, validation, and test sets
"""
from typing import Dict, List, Any, Tuple, Optional, Union, Callable
import random
import math
import numpy as np
from abc import ABC, abstractmethod

from llamadatasets.core.dataset import Dataset


class BaseSplitter(ABC):
    """
    Abstract base class for dataset splitters.
    
    Dataset splitters divide a dataset into multiple subsets (typically train,
    validation, and test sets).
    """
    
    @abstractmethod
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset into multiple subsets.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary mapping split names to datasets
        """
        pass


class RandomSplitter(BaseSplitter):
    """
    Splitter that randomly divides examples into train/validation/test sets.
    """
    
    def __init__(self, 
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the random splitter.
        
        Args:
            train_size: Proportion for training set (0 to 1)
            val_size: Proportion for validation set (0 to 1)
            test_size: Proportion for test set (0 to 1)
            seed: Random seed for reproducibility
        """
        # Validate that proportions sum to 1
        total = train_size + val_size + test_size
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"Split proportions must sum to 1, got {total}")
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
    
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset randomly.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary with 'train', 'val', and 'test' splits
        """
        # Get all examples and shuffle
        examples = dataset.to_dict_list()
        
        if self.seed is not None:
            random.seed(self.seed)
        
        random.shuffle(examples)
        
        # Calculate split indices
        n = len(examples)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)
        
        # Split the dataset
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        return {
            'train': Dataset(train_examples),
            'val': Dataset(val_examples),
            'test': Dataset(test_examples)
        }


class StratifiedSplitter(BaseSplitter):
    """
    Splitter that maintains the same class distribution across splits.
    """
    
    def __init__(self, 
                 label_column: str,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the stratified splitter.
        
        Args:
            label_column: Column containing class labels
            train_size: Proportion for training set (0 to 1)
            val_size: Proportion for validation set (0 to 1)
            test_size: Proportion for test set (0 to 1)
            seed: Random seed for reproducibility
        """
        # Validate that proportions sum to 1
        total = train_size + val_size + test_size
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"Split proportions must sum to 1, got {total}")
        
        self.label_column = label_column
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
    
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset stratified by label.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary with 'train', 'val', and 'test' splits
        """
        # Group examples by label
        examples = dataset.to_dict_list()
        labels_to_examples = {}
        
        for example in examples:
            if self.label_column not in example:
                raise KeyError(f"Label column '{self.label_column}' not found in example")
            
            label = example[self.label_column]
            if label not in labels_to_examples:
                labels_to_examples[label] = []
            
            labels_to_examples[label].append(example)
        
        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)
        
        # Shuffle each group
        for label in labels_to_examples:
            random.shuffle(labels_to_examples[label])
        
        # Initialize split examples
        train_examples = []
        val_examples = []
        test_examples = []
        
        # Split each group and add to respective splits
        for label, group in labels_to_examples.items():
            n = len(group)
            train_end = int(n * self.train_size)
            val_end = train_end + int(n * self.val_size)
            
            train_examples.extend(group[:train_end])
            val_examples.extend(group[train_end:val_end])
            test_examples.extend(group[val_end:])
        
        # Shuffle again to mix examples from different classes
        random.shuffle(train_examples)
        random.shuffle(val_examples)
        random.shuffle(test_examples)
        
        return {
            'train': Dataset(train_examples),
            'val': Dataset(val_examples),
            'test': Dataset(test_examples)
        }


class TimeSplitter(BaseSplitter):
    """
    Splitter that divides examples based on a timestamp column.
    """
    
    def __init__(self, 
                 timestamp_column: str,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 ascending: bool = True):
        """
        Initialize the time splitter.
        
        Args:
            timestamp_column: Column containing timestamps
            train_size: Proportion for training set (0 to 1)
            val_size: Proportion for validation set (0 to 1)
            test_size: Proportion for test set (0 to 1)
            ascending: Whether to sort timestamps in ascending order
        """
        # Validate that proportions sum to 1
        total = train_size + val_size + test_size
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"Split proportions must sum to 1, got {total}")
        
        self.timestamp_column = timestamp_column
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.ascending = ascending
    
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset based on timestamps.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary with 'train', 'val', and 'test' splits
        """
        # Get all examples and sort by timestamp
        examples = dataset.to_dict_list()
        
        # Check that timestamp column exists
        if not examples or self.timestamp_column not in examples[0]:
            raise KeyError(f"Timestamp column '{self.timestamp_column}' not found in dataset")
        
        # Sort by timestamp
        examples.sort(key=lambda x: x[self.timestamp_column], reverse=not self.ascending)
        
        # Calculate split indices
        n = len(examples)
        train_end = int(n * self.train_size)
        val_end = train_end + int(n * self.val_size)
        
        # Split the dataset
        train_examples = examples[:train_end]
        val_examples = examples[train_end:val_end]
        test_examples = examples[val_end:]
        
        return {
            'train': Dataset(train_examples),
            'val': Dataset(val_examples),
            'test': Dataset(test_examples)
        }


class GroupSplitter(BaseSplitter):
    """
    Splitter that keeps examples with the same group ID in the same split.
    """
    
    def __init__(self, 
                 group_column: str,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 seed: Optional[int] = None):
        """
        Initialize the group splitter.
        
        Args:
            group_column: Column containing group IDs
            train_size: Proportion for training set (0 to 1)
            val_size: Proportion for validation set (0 to 1)
            test_size: Proportion for test set (0 to 1)
            seed: Random seed for reproducibility
        """
        # Validate that proportions sum to 1
        total = train_size + val_size + test_size
        if not math.isclose(total, 1.0, abs_tol=1e-10):
            raise ValueError(f"Split proportions must sum to 1, got {total}")
        
        self.group_column = group_column
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
    
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset while keeping the same group in the same split.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary with 'train', 'val', and 'test' splits
        """
        # Group examples by group ID
        examples = dataset.to_dict_list()
        groups_to_examples = {}
        
        for example in examples:
            if self.group_column not in example:
                raise KeyError(f"Group column '{self.group_column}' not found in example")
            
            group = example[self.group_column]
            if group not in groups_to_examples:
                groups_to_examples[group] = []
            
            groups_to_examples[group].append(example)
        
        # Get unique groups and shuffle them
        groups = list(groups_to_examples.keys())
        
        if self.seed is not None:
            random.seed(self.seed)
        
        random.shuffle(groups)
        
        # Calculate split indices for groups
        n_groups = len(groups)
        train_end = int(n_groups * self.train_size)
        val_end = train_end + int(n_groups * self.val_size)
        
        # Assign groups to splits
        train_groups = groups[:train_end]
        val_groups = groups[train_end:val_end]
        test_groups = groups[val_end:]
        
        # Collect examples for each split
        train_examples = []
        val_examples = []
        test_examples = []
        
        for group in train_groups:
            train_examples.extend(groups_to_examples[group])
        
        for group in val_groups:
            val_examples.extend(groups_to_examples[group])
        
        for group in test_groups:
            test_examples.extend(groups_to_examples[group])
        
        return {
            'train': Dataset(train_examples),
            'val': Dataset(val_examples),
            'test': Dataset(test_examples)
        }


class CustomSplitter(BaseSplitter):
    """
    Splitter that uses a custom function to split the dataset.
    """
    
    def __init__(self, split_func: Callable[[Dataset], Dict[str, Dataset]]):
        """
        Initialize the custom splitter.
        
        Args:
            split_func: Function that takes a dataset and returns a dictionary of splits
        """
        self.split_func = split_func
    
    def split(self, dataset: Dataset) -> Dict[str, Dataset]:
        """
        Split the dataset using the custom function.
        
        Args:
            dataset: The input dataset to split
            
        Returns:
            Dict[str, Dataset]: Dictionary mapping split names to datasets
        """
        return self.split_func(dataset) 