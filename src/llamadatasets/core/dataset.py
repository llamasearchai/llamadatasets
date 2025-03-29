"""
Dataset is the core data structure of LlamaDatasets
"""
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, TypeVar, Generic
import random
import json
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

T = TypeVar('T')


class Dataset(Generic[T]):
    """
    Dataset represents a collection of examples that can be indexed, filtered, and transformed.
    """
    
    def __init__(self, data: List[T]):
        """
        Initialize a Dataset with a list of examples.
        
        Args:
            data: List of examples (usually dictionaries)
        """
        self._data = data
        self._index = list(range(len(data)))
    
    def __len__(self) -> int:
        """
        Get the number of examples in the dataset.
        
        Returns:
            int: The number of examples
        """
        return len(self._data)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[T, List[T]]:
        """
        Get an example or slice of examples by index.
        
        Args:
            idx: Integer index or slice

        Returns:
            The example or list of examples
        """
        if isinstance(idx, slice):
            indices = self._index[idx]
            return [self._data[i] for i in indices]
        else:
            return self._data[self._index[idx]]
    
    def filter(self, filter_fn: Callable[[T], bool]) -> 'Dataset[T]':
        """
        Filter the dataset using a predicate function.
        
        Args:
            filter_fn: Function that takes an example and returns True to keep it or False to filter it out

        Returns:
            Dataset: A new filtered dataset
        """
        filtered_data = [self._data[i] for i in self._index if filter_fn(self._data[i])]
        return Dataset(filtered_data)
    
    def map(self, map_fn: Callable[[T], Any]) -> 'Dataset':
        """
        Apply a function to each example in the dataset.
        
        Args:
            map_fn: Function to apply to each example

        Returns:
            Dataset: A new dataset with transformed examples
        """
        mapped_data = [map_fn(self._data[i]) for i in self._index]
        return Dataset(mapped_data)
    
    def transform(self, transformers: List[Any]) -> 'Dataset':
        """
        Apply a sequence of transformers to the dataset.
        
        Args:
            transformers: List of transformer objects with a transform method

        Returns:
            Dataset: A new transformed dataset
        """
        data = self
        for transformer in transformers:
            data = transformer.transform(data)
        return data
    
    def sample(self, n: int = 1, seed: Optional[int] = None) -> Union[T, List[T]]:
        """
        Sample n random examples from the dataset.
        
        Args:
            n: Number of examples to sample
            seed: Random seed for reproducibility

        Returns:
            The sampled example or list of examples
        """
        if seed is not None:
            random.seed(seed)
        
        if n == 1:
            idx = random.choice(self._index)
            return self._data[idx]
        else:
            n = min(n, len(self))
            sampled_indices = random.sample(self._index, n)
            return [self._data[i] for i in sampled_indices]
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert the dataset to a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The dataset as a DataFrame
        """
        return pd.DataFrame([self._data[i] for i in self._index])
    
    def to_dict_list(self) -> List[Dict[str, Any]]:
        """
        Convert the dataset to a list of dictionaries.
        
        Returns:
            List[Dict[str, Any]]: The dataset as a list of dictionaries
        """
        # If the data is already a list of dictionaries, just return it
        # Otherwise, try to convert each item to a dictionary
        if all(isinstance(self._data[i], dict) for i in self._index):
            return [self._data[i] for i in self._index]
        else:
            return [dict(item) if hasattr(item, '__dict__') else item for item in [self._data[i] for i in self._index]]
    
    def to_numpy(self, columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert the dataset to a numpy array.
        
        Args:
            columns: List of column names to include (if data consists of dictionaries)

        Returns:
            np.ndarray: The dataset as a numpy array
        """
        df = self.to_pandas()
        if columns:
            return df[columns].to_numpy()
        else:
            return df.to_numpy()
    
    def get_column(self, column: str) -> List[Any]:
        """
        Get all values for a specific column/key in the dataset.
        
        Args:
            column: The column/key name

        Returns:
            List[Any]: List of values for the specified column
        """
        values = []
        for i in self._index:
            item = self._data[i]
            if isinstance(item, dict) and column in item:
                values.append(item[column])
            elif hasattr(item, column):
                values.append(getattr(item, column))
            else:
                raise KeyError(f"Column '{column}' not found in dataset item")
        return values
    
    def head(self, n: int = 5) -> 'Dataset[T]':
        """
        Get the first n examples of the dataset.
        
        Args:
            n: Number of examples to return

        Returns:
            Dataset: A new dataset with the first n examples
        """
        n = min(n, len(self))
        return Dataset([self._data[i] for i in self._index[:n]])
    
    def iter_batches(self, batch_size: int = 32) -> Iterator[List[T]]:
        """
        Iterate over the dataset in batches.
        
        Args:
            batch_size: Size of each batch

        Yields:
            List[T]: Batch of examples
        """
        for i in range(0, len(self), batch_size):
            indices = self._index[i:i + batch_size]
            yield [self._data[j] for j in indices]
    
    def split(self, ratio: float = 0.8, seed: Optional[int] = None) -> tuple['Dataset[T]', 'Dataset[T]']:
        """
        Split the dataset into two parts based on a ratio.
        
        Args:
            ratio: Proportion of data in the first split (0.0 to 1.0)
            seed: Random seed for reproducibility

        Returns:
            tuple: (first_split, second_split)
        """
        from llamadatasets.splitters import RandomSplitter
        splitter = RandomSplitter(train_ratio=ratio, val_ratio=1-ratio, test_ratio=0.0, seed=seed)
        train_data, val_data, _ = splitter.split(self)
        return train_data, val_data
    
    def save(self, path: str, format: str = "json") -> None:
        """
        Save the dataset to a file.
        
        Args:
            path: Path to save the file
            format: Format of the file ("json", "csv", or "parquet")
        """
        data = [self._data[i] for i in self._index]
        
        if format.lower() == "json":
            with open(path, "w") as f:
                json.dump(data, f)
        elif format.lower() == "csv":
            df = pd.DataFrame(data)
            df.to_csv(path, index=False)
        elif format.lower() == "parquet":
            df = pd.DataFrame(data)
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load(cls, path: str, format: Optional[str] = None) -> 'Dataset':
        """
        Load a dataset from a file.
        
        Args:
            path: Path to the file
            format: Format of the file ("json", "csv", or "parquet"), inferred from path if None

        Returns:
            Dataset: The loaded dataset
        """
        from llamadatasets import DataLoader
        
        if format is None:
            # Infer format from file extension
            if path.endswith(".json"):
                format = "json"
            elif path.endswith(".csv"):
                format = "csv"
            elif path.endswith(".parquet"):
                format = "parquet"
            else:
                raise ValueError(f"Cannot infer format from path: {path}")
        
        if format.lower() == "json":
            loader = DataLoader.from_json(path)
        elif format.lower() == "csv":
            loader = DataLoader.from_csv(path)
        elif format.lower() == "parquet":
            loader = DataLoader.from_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
 