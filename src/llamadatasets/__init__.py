"""
LlamaDatasets: A library for dataset management and processing for LlamaSearch.ai applications
"""

__version__ = "0.1.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai" = "Nik Jois"
__email__ = "nikjois@llamasearch.ai" = "Nik Jois"
__license__ = "MIT"

# Import core components
from llamadatasets.core import CacheConfig, DataLoader, Dataset, StreamingDataset

# Import generators
from llamadatasets.generators import (
    BaseTextGenerator,
    RandomTextGenerator,
    TemplateTextGenerator,
)

# Import splitters
from llamadatasets.splitters import (
    BaseSplitter,
    CustomSplitter,
    GroupSplitter,
    RandomSplitter,
    StratifiedSplitter,
    TimeSplitter,
)

# Import transformers
from llamadatasets.transformers import (
    BaseTransformer,
    ChainTransformer,
    ColumnTransformer,
    FunctionTransformer,
    StopWordsRemoverTransformer,
    TextCleanerTransformer,
    TextLemmatizerTransformer,
    TextStemmerTransformer,
    TokenizerTransformer,
)

__all__ = [
    # Core
    'Dataset',
    'DataLoader',
    'CacheConfig',
    'StreamingDataset',
    
    # Transformers
    'BaseTransformer',
    'FunctionTransformer',
    'ColumnTransformer',
    'ChainTransformer',
    'TextCleanerTransformer',
    'TokenizerTransformer',
    'StopWordsRemoverTransformer',
    'TextStemmerTransformer',
    'TextLemmatizerTransformer',
    
    # Splitters
    'BaseSplitter',
    'RandomSplitter',
    'StratifiedSplitter',
    'TimeSplitter',
    'GroupSplitter',
    'CustomSplitter',
    
    # Generators
    'BaseTextGenerator',
    'RandomTextGenerator',
    'TemplateTextGenerator'
] 