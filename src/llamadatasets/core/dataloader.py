"""
DataLoader provides methods to load data from various sources
"""
from typing import List, Dict, Any, Optional, Union, Callable, Type
import os
import json
import logging
import hashlib
import time
from datetime import datetime, timedelta
import pandas as pd

from pydantic import BaseModel, Field

from llamadatasets.core.dataset import Dataset
from llamadatasets.core.streaming import StreamingDataset

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """
    Configuration for caching behavior
    """
    enabled: bool = True
    location: str = "/tmp/llamadatasets_cache"
    expiration: Optional[int] = None  # Seconds until cache expires
    max_size: Optional[int] = None  # Maximum size in bytes
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure cache directory exists
        if self.enabled and not os.path.exists(self.location):
            try:
                os.makedirs(self.location, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create cache directory: {e}")
                self.enabled = False


class DataLoader:
    """
    DataLoader provides methods to load data from various sources
    """
    
    def __init__(self, 
                 source_type: str,
                 source_path: Optional[str] = None,
                 streaming: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 cache_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize a DataLoader.
        
        Args:
            source_type: Type of data source ("csv", "json", "parquet", "database", etc.)
            source_path: Path to the data source (file path or connection string)
            streaming: Whether to load data in streaming mode
            cache_config: Configuration for caching behavior
            cache_key: Key for caching (if None, generated from source_path and kwargs)
            **kwargs: Additional arguments for the specific loader
        """
        self.source_type = source_type
        self.source_path = source_path
        self.streaming = streaming
        self.cache_config = cache_config or CacheConfig(enabled=False)
        self.cache_key = cache_key
        self.kwargs = kwargs
    
    @classmethod
    def from_csv(cls, 
                 path: str, 
                 streaming: bool = False,
                 cache_config: Optional[CacheConfig] = None,
                 cache_key: Optional[str] = None,
                 **kwargs) -> 'DataLoader':
        """
        Create a loader for CSV files.
        
        Args:
            path: Path to the CSV file
            streaming: Whether to load data in streaming mode
            cache_config: Configuration for caching behavior
            cache_key: Key for caching
            **kwargs: Additional arguments for pandas.read_csv

        Returns:
            DataLoader: A configured DataLoader instance
        """
        return cls(
            source_type="csv",
            source_path=path,
            streaming=streaming,
            cache_config=cache_config,
            cache_key=cache_key,
            **kwargs
        )
    
    @classmethod
    def from_json(cls, 
                  path: str, 
                  streaming: bool = False,
                  cache_config: Optional[CacheConfig] = None,
                  cache_key: Optional[str] = None,
                  **kwargs) -> 'DataLoader':
        """
        Create a loader for JSON files.
        
        Args:
            path: Path to the JSON file
            streaming: Whether to load data in streaming mode
            cache_config: Configuration for caching behavior
            cache_key: Key for caching
            **kwargs: Additional arguments for json.load

        Returns:
            DataLoader: A configured DataLoader instance
        """
        return cls(
            source_type="json",
            source_path=path,
            streaming=streaming,
            cache_config=cache_config,
            cache_key=cache_key,
            **kwargs
        )
    
    @classmethod
    def from_parquet(cls, 
                     path: str, 
                     streaming: bool = False,
                     cache_config: Optional[CacheConfig] = None,
                     cache_key: Optional[str] = None,
                     **kwargs) -> 'DataLoader':
        """
        Create a loader for Parquet files.
        
        Args:
            path: Path to the Parquet file
            streaming: Whether to load data in streaming mode
            cache_config: Configuration for caching behavior
            cache_key: Key for caching
            **kwargs: Additional arguments for pandas.read_parquet

        Returns:
            DataLoader: A configured DataLoader instance
        """
        return cls(
            source_type="parquet",
            source_path=path,
            streaming=streaming,
            cache_config=cache_config,
            cache_key=cache_key,
            **kwargs
        )
    
    @classmethod
    def from_database(cls, 
                      connection_string: str,
                      query: str,
                      streaming: bool = False,
                      cache_config: Optional[CacheConfig] = None,
                      cache_key: Optional[str] = None,
                      **kwargs) -> 'DataLoader':
        """
        Create a loader for database queries.
        
        Args:
            connection_string: Database connection string
            query: SQL query to execute
            streaming: Whether to load data in streaming mode
            cache_config: Configuration for caching behavior
            cache_key: Key for caching
            **kwargs: Additional arguments for SQLAlchemy

        Returns:
            DataLoader: A configured DataLoader instance
        """
        return cls(
            source_type="database",
            source_path=connection_string,
            streaming=streaming,
            cache_config=cache_config,
            cache_key=cache_key,
            query=query,
            **kwargs
        )
    
    def load(self) -> Union[Dataset, StreamingDataset]:
        """
        Load the data from the source.
        
        Returns:
            Union[Dataset, StreamingDataset]: The loaded dataset
        """
        # Check cache first if enabled
        if self.cache_config.enabled:
            cached_data = self._load_from_cache()
            if cached_data is not None:
                logger.info(f"Loaded data from cache for {self.source_type} source")
                return cached_data
        
        # Load data based on source type
        if self.source_type == "csv":
            data = self._load_csv()
        elif self.source_type == "json":
            data = self._load_json()
        elif self.source_type == "parquet":
            data = self._load_parquet()
        elif self.source_type == "database":
            data = self._load_database()
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        
        # Save to cache if enabled
        if self.cache_config.enabled:
            self._save_to_cache(data)
        
        return data
    
    def _load_csv(self) -> Union[Dataset, StreamingDataset]:
        """
        Load data from a CSV file.
        
        Returns:
            Union[Dataset, StreamingDataset]: The loaded dataset
        """
        if self.streaming:
            # Return a streaming dataset
            return StreamingDataset(
                source_type="csv",
                source_path=self.source_path,
                **self.kwargs
            )
        else:
            # Load the entire CSV into memory
            df = pd.read_csv(self.source_path, **self.kwargs)
            data = df.to_dict('records')
            return Dataset(data)
    
    def _load_json(self) -> Union[Dataset, StreamingDataset]:
        """
        Load data from a JSON file.
        
        Returns:
            Union[Dataset, StreamingDataset]: The loaded dataset
        """
        if self.streaming:
            # Return a streaming dataset
            return StreamingDataset(
                source_type="json",
                source_path=self.source_path,
                **self.kwargs
            )
        else:
            # Load the entire JSON into memory
            with open(self.source_path, 'r') as f:
                data = json.load(f, **self.kwargs)
            
            # Ensure data is a list
            if isinstance(data, dict):
                # Handle case where JSON is an object with a data field
                if 'data' in data and isinstance(data['data'], list):
                    data = data['data']
                # Handle case where JSON is a dictionary of records
                else:
                    data = [data]
            
            return Dataset(data)
    
    def _load_parquet(self) -> Union[Dataset, StreamingDataset]:
        """
        Load data from a Parquet file.
        
        Returns:
            Union[Dataset, StreamingDataset]: The loaded dataset
        """
        if self.streaming:
            # Return a streaming dataset
            return StreamingDataset(
                source_type="parquet",
                source_path=self.source_path,
                **self.kwargs
            )
        else:
            # Load the entire Parquet file into memory
            df = pd.read_parquet(self.source_path, **self.kwargs)
            data = df.to_dict('records')
            return Dataset(data)
    
    def _load_database(self) -> Union[Dataset, StreamingDataset]:
        """
        Load data from a database query.
        
        Returns:
            Union[Dataset, StreamingDataset]: The loaded dataset
        """
        import sqlalchemy
        
        query = self.kwargs.get('query')
        if not query:
            raise ValueError("Query must be provided for database source")
        
        if self.streaming:
            # Return a streaming dataset
            return StreamingDataset(
                source_type="database",
                source_path=self.source_path,
                query=query,
                **self.kwargs
            )
        else:
            # Load the entire query result into memory
            engine = sqlalchemy.create_engine(self.source_path)
            with engine.connect() as connection:
                df = pd.read_sql(query, connection)
            data = df.to_dict('records')
            return Dataset(data)
    
    def _get_cache_key(self) -> str:
        """
        Get the cache key, generating one if not provided.
        
        Returns:
            str: The cache key
        """
        if self.cache_key:
            return self.cache_key
        
        # Generate a cache key based on source path and kwargs
        key_components = [
            self.source_type,
            str(self.source_path),
            json.dumps(self.kwargs, sort_keys=True)
        ]
        
        key_str = "_".join(key_components)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cache_path(self) -> str:
        """
        Get the path to the cache file.
        
        Returns:
            str: The cache file path
        """
        key = self._get_cache_key()
        return os.path.join(self.cache_config.location, f"{key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """
        Check if the cache is valid (not expired).
        
        Args:
            cache_path: Path to the cache file

        Returns:
            bool: True if the cache is valid, False otherwise
        """
        if not os.path.exists(cache_path):
            return False
        
        # Check expiration if set
        if self.cache_config.expiration:
            cache_time = os.path.getmtime(cache_path)
            cache_age = time.time() - cache_time
            if cache_age > self.cache_config.expiration:
                return False
        
        # Check max size if set
        if self.cache_config.max_size:
            cache_size = os.path.getsize(cache_path)
            if cache_size > self.cache_config.max_size:
                return False
        
        return True
    
    def _load_from_cache(self) -> Optional[Dataset]:
        """
        Load data from cache if available and valid.
        
        Returns:
            Optional[Dataset]: The cached dataset or None if not available
        """
        cache_path = self._get_cache_path()
        
        if not self._is_cache_valid(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return Dataset(data)
        except Exception as e:
            logger.warning(f"Failed to load from cache: {e}")
            return None
    
    def _save_to_cache(self, dataset: Dataset) -> None:
        """
        Save dataset to cache.
        
        Args:
            dataset: The dataset to cache
        """
        cache_path = self._get_cache_path()
        
        try:
            # Convert dataset to list of dictionaries
            data = dataset.to_dict_list()
            
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            logger.info(f"Saved data to cache at {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save to cache: {e}")
    
    def __repr__(self) -> str:
        return f"DataLoader(source_type='{self.source_type}', source_path='{self.source_path}', streaming={self.streaming})" 