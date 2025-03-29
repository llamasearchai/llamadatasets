"""
LlamaDatasets: A library for dataset management and processing for LlamaSearch.ai applications
"""

__version__ = "0.1.0"
__author__ = "LlamaSearch.ai"
__license__ = "MIT"

# Import core components
from llamadatasets.core import (
    Dataset,
    DataLoader,
    CacheConfig,
    StreamingDataset
)

# Import transformers
from llamadatasets.transformers import (
    BaseTransformer,
    FunctionTransformer,
    ColumnTransformer,
    ChainTransformer,
    TextCleanerTransformer,
    TokenizerTransformer,
    StopWordsRemoverTransformer,
    TextStemmerTransformer,
    TextLemmatizerTransformer
)

# Import splitters
from llamadatasets.splitters import (
    BaseSplitter,
    RandomSplitter,
    StratifiedSplitter,
    TimeSplitter,
    GroupSplitter,
    CustomSplitter
)

# Import generators
from llamadatasets.generators import (
    BaseTextGenerator,
    RandomTextGenerator,
    TemplateTextGenerator
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