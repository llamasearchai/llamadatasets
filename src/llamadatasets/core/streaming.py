"""
StreamingDataset for efficient processing of large datasets
"""
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, Generator
import os
import json
import logging
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StreamingDataset:
    """
    StreamingDataset loads and processes data in chunks for memory-efficient handling of large datasets
    """
    
    def __init__(self, 
                 source_type: str,
                 source_path: str,
                 chunk_size: int = 10000,
                 progress_bar: bool = True,
                 **kwargs):
        """
        Initialize a StreamingDataset.
        
        Args:
            source_type: Type of data source ("csv", "json", "parquet", "database")
            source_path: Path to the data source (file path or connection string)
            chunk_size: Size of chunks to load at once
            progress_bar: Whether to show a progress bar
            **kwargs: Additional arguments for the specific loader
        """
        self.source_type = source_type
        self.source_path = source_path
        self.chunk_size = chunk_size
        self.progress_bar = progress_bar
        self.kwargs = kwargs
        
        # Initialize stats
        self._initialize_stats()
    
    def _initialize_stats(self) -> None:
        """
        Initialize dataset statistics.
        """
        self.total_rows = 0
        self.columns = []
        
        # Try to get total rows and columns for progress reporting
        try:
            if self.source_type == "csv":
                with open(self.source_path, 'r') as f:
                    # Count lines and get header
                    header = f.readline().strip().split(',')
                    self.columns = [col.strip() for col in header]
                    
                    # Count lines, but limit to avoid scanning entire file
                    count = 1  # header
                    for _ in f:
                        count += 1
                        if count > 1000000:  # Limit to 1M rows for checking
                            break
                    
                    self.total_rows = count - 1  # Subtract header
            
            elif self.source_type == "parquet":
                parquet_file = pq.ParquetFile(self.source_path)
                self.total_rows = parquet_file.metadata.num_rows
                self.columns = parquet_file.schema.names
            
            elif self.source_type == "database":
                # For databases, we'll estimate by running a count query
                import sqlalchemy
                table = self.kwargs.get('table')
                query = self.kwargs.get('query')
                
                if table or query:
                    engine = sqlalchemy.create_engine(self.source_path)
                    with engine.connect() as connection:
                        if table:
                            count_query = f"SELECT COUNT(*) FROM {table}"
                        else:
                            # This is a rough estimate and may not work for all queries
                            count_query = f"SELECT COUNT(*) FROM ({query}) as subquery"
                        
                        try:
                            result = connection.execute(sqlalchemy.text(count_query))
                            self.total_rows = result.scalar()
                        except Exception as e:
                            logger.warning(f"Failed to get row count: {e}")
        
        except Exception as e:
            logger.warning(f"Failed to initialize dataset stats: {e}")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate through the dataset one example at a time.
        
        Yields:
            Dict[str, Any]: A single example from the dataset
        """
        for batch in self.iter_batches(batch_size=1):
            yield batch[0]
    
    def iter_batches(self, batch_size: int = 1000) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate through the dataset in batches.
        
        Args:
            batch_size: Size of batches to yield

        Yields:
            List[Dict[str, Any]]: A batch of examples
        """
        # Determine which iterator to use based on source type
        if self.source_type == "csv":
            iterator = self._iter_csv_batches(batch_size)
        elif self.source_type == "json":
            iterator = self._iter_json_batches(batch_size)
        elif self.source_type == "parquet":
            iterator = self._iter_parquet_batches(batch_size)
        elif self.source_type == "database":
            iterator = self._iter_database_batches(batch_size)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        
        # Wrap with progress bar if requested
        if self.progress_bar and self.total_rows > 0:
            with tqdm(total=self.total_rows) as pbar:
                for batch in iterator:
                    pbar.update(len(batch))
                    yield batch
        else:
            yield from iterator
    
    def _iter_csv_batches(self, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through a CSV file in batches.
        
        Args:
            batch_size: Size of batches to yield

        Yields:
            List[Dict[str, Any]]: A batch of examples
        """
        for chunk in pd.read_csv(self.source_path, chunksize=self.chunk_size, **self.kwargs):
            # Process the chunk in smaller batches
            records = chunk.to_dict('records')
            for i in range(0, len(records), batch_size):
                yield records[i:i + batch_size]
    
    def _iter_json_batches(self, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through a JSON file in batches.
        
        Args:
            batch_size: Size of batches to yield

        Yields:
            List[Dict[str, Any]]: A batch of examples
        """
        # For JSON, we currently load the whole file into memory
        # This could be improved for line-delimited JSON
        with open(self.source_path, 'r') as f:
            data = json.load(f, **self.kwargs)
        
        # Ensure data is a list
        if isinstance(data, dict):
            if 'data' in data and isinstance(data['data'], list):
                data = data['data']
            else:
                data = [data]
        
        # Yield in batches
        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]
    
    def _iter_parquet_batches(self, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through a Parquet file in batches.
        
        Args:
            batch_size: Size of batches to yield

        Yields:
            List[Dict[str, Any]]: A batch of examples
        """
        # Open the Parquet file
        parquet_file = pq.ParquetFile(self.source_path)
        
        for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
            # Convert to pandas and then to records
            df = batch.to_pandas()
            records = df.to_dict('records')
            
            # Yield in smaller batches
            for i in range(0, len(records), batch_size):
                yield records[i:i + batch_size]
    
    def _iter_database_batches(self, batch_size: int) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate through a database query in batches.
        
        Args:
            batch_size: Size of batches to yield

        Yields:
            List[Dict[str, Any]]: A batch of examples
        """
        import sqlalchemy
        
        query = self.kwargs.get('query')
        if not query:
            raise ValueError("Query must be provided for database source")
        
        engine = sqlalchemy.create_engine(self.source_path)
        
        # Use pandas to handle chunking
        with engine.connect() as connection:
            for chunk in pd.read_sql(query, connection, chunksize=self.chunk_size):
                records = chunk.to_dict('records')
                
                # Yield in smaller batches
                for i in range(0, len(records), batch_size):
                    yield records[i:i + batch_size]
    
    def map(self, map_fn: Callable[[Dict[str, Any]], Any], batch_size: int = 1000, show_progress: bool = True) -> 'StreamingDataset':
        """
        Apply a function to each example in the dataset.
        
        Args:
            map_fn: Function to apply to each example
            batch_size: Size of batches to process at once
            show_progress: Whether to show a progress bar

        Returns:
            StreamingDataset: A transformed streaming dataset
        """
        raise NotImplementedError("Mapping for StreamingDataset is not implemented yet")
    
    def to_dataset(self, max_examples: Optional[int] = None) -> 'Dataset':
        """
        Convert this streaming dataset to a regular in-memory Dataset.
        Warning: This loads the entire dataset into memory.
        
        Args:
            max_examples: Maximum number of examples to load

        Returns:
            Dataset: An in-memory Dataset
        """
        from llamadatasets.core.dataset import Dataset
        
        # Load examples into memory
        examples = []
        for batch in self.iter_batches(batch_size=1000):
            examples.extend(batch)
            if max_examples and len(examples) >= max_examples:
                examples = examples[:max_examples]
                break
        
        return Dataset(examples)
    
    def head(self, n: int = 5) -> 'Dataset':
        """
        Get the first n examples of the dataset.
        
        Args:
            n: Number of examples to return

        Returns:
            Dataset: A new dataset with the first n examples
        """
        from llamadatasets.core.dataset import Dataset
        
        examples = []
        for batch in self.iter_batches(batch_size=n):
            examples.extend(batch)
            if len(examples) >= n:
                break
        
        return Dataset(examples[:n])
    
    def __repr__(self) -> str:
        return f"StreamingDataset(source_type='{self.source_type}', source_path='{self.source_path}', total_rows={self.total_rows})" 