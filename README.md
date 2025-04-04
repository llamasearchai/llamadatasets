# LlamaDatasets

A comprehensive dataset management and processing library for LlamaSearch.ai applications.

## Features

- **Dataset Loaders**: Load datasets from various formats (CSV, JSON, Parquet, etc.)
- **Data Transformation**: Tools for preprocessing, cleaning, and transforming datasets
- **Dataset Splitting**: Methods for splitting datasets for training, validation, and testing
- **Synthetic Data Generation**: Utilities for generating synthetic data for testing and development
- **Data Validation**: Validate datasets against schemas and requirements
- **Integration with LlamaSearch.ai Ecosystem**: Seamless integration with other LlamaSearch.ai tools
- **Streaming Data Processing**: Efficient processing of large datasets using streaming
- **Caching Mechanisms**: Caching strategies for optimizing data loading and processing

## Installation

### Using pip

```bash
pip install llamadatasets
```

### From source

```bash
git clone https://llamasearch.ai
cd llamadatasets
pip install -e .
```

## Quick Start

```python
from llamadatasets import Dataset, DataLoader
from llamadatasets.transformers import TextNormalizer, Tokenizer
from llamadatasets.splitters import RandomSplitter

# Load a dataset
data_loader = DataLoader.from_csv("path/to/data.csv")
dataset = data_loader.load()

# Apply transformations
normalizer = TextNormalizer(lowercase=True, remove_punctuation=True)
tokenizer = Tokenizer(tokenizer_type="word")

transformed_dataset = dataset.transform([normalizer, tokenizer])

# Split the dataset
splitter = RandomSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
train_data, val_data, test_data = splitter.split(transformed_dataset)

# Use the datasets
print(f"Training examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)}")

# Sample some data
print(train_data.sample(3))
```

## Core Components

### Dataset

The `Dataset` class is the central data structure that represents a collection of examples:

```python
from llamadatasets import Dataset

# Create a dataset from a list of dictionaries
data = [
    {"text": "This is an example", "label": "positive"},
    {"text": "Another example", "label": "negative"},
    # ...
]

dataset = Dataset(data)

# Access data
print(dataset[0])  # {'text': 'This is an example', 'label': 'positive'}

# Filter data
positive_examples = dataset.filter(lambda example: example["label"] == "positive")

# Map a function to each example
processed = dataset.map(lambda example: {
    "text": example["text"].lower(),
    "label": example["label"]
})
```

### DataLoaders

DataLoaders provide methods to load data from various sources:

```python
from llamadatasets import DataLoader

# Load from CSV
csv_loader = DataLoader.from_csv("data.csv")
csv_dataset = csv_loader.load()

# Load from JSON
json_loader = DataLoader.from_json("data.json")
json_dataset = json_loader.load()

# Load from Parquet
parquet_loader = DataLoader.from_parquet("data.parquet")
parquet_dataset = parquet_loader.load()

# Load from a database
db_loader = DataLoader.from_database(
    connection_string="postgresql://user:pass@localhost:5432/db",
    query="SELECT * FROM data_table"
)
db_dataset = db_loader.load()
```

### Transformers

Transformers process and modify datasets:

```python
from llamadatasets.transformers import (
    TextNormalizer, Tokenizer, OneHotEncoder, StandardScaler
)

# Configure transformers
normalizer = TextNormalizer(lowercase=True, remove_punctuation=True)
tokenizer = Tokenizer(max_length=100)
encoder = OneHotEncoder(columns=["category"])
scaler = StandardScaler(columns=["numeric_feature"])

# Apply transformers in sequence
transformed_dataset = dataset.transform([
    normalizer,
    tokenizer,
    encoder,
    scaler
])
```

### Splitters

Splitters divide datasets into training, validation, and test sets:

```python
from llamadatasets.splitters import (
    RandomSplitter, StratifiedSplitter, TimeSeriesSplitter
)

# Random splitting
random_splitter = RandomSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
train_data, val_data, test_data = random_splitter.split(dataset)

# Stratified splitting (preserves label distribution)
stratified_splitter = StratifiedSplitter(
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    stratify_column="label", seed=42
)
train_data, val_data, test_data = stratified_splitter.split(dataset)

# Time series splitting
timeseries_splitter = TimeSeriesSplitter(
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    time_column="date"
)
train_data, val_data, test_data = timeseries_splitter.split(dataset)
```

### Synthetic Data Generators

Generate synthetic data for testing and development:

```python
from llamadatasets.generators import (
    TextGenerator, TabularDataGenerator, TimeSeriesGenerator
)

# Generate synthetic text data
text_generator = TextGenerator(num_examples=100, length_range=(10, 50))
text_dataset = text_generator.generate()

# Generate synthetic tabular data
schema = {
    "age": {"type": "int", "range": (18, 90)},
    "income": {"type": "float", "range": (20000, 150000)},
    "category": {"type": "category", "values": ["A", "B", "C"]}
}
tabular_generator = TabularDataGenerator(schema=schema, num_examples=200)
tabular_dataset = tabular_generator.generate()

# Generate synthetic time series data
timeseries_generator = TimeSeriesGenerator(
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="D",
    patterns=["trend", "seasonality"]
)
timeseries_dataset = timeseries_generator.generate()
```

## Advanced Usage

### Handling Large Datasets

```python
from llamadatasets import DataLoader, StreamingDataset

# Load large dataset in streaming mode
loader = DataLoader.from_csv("very_large_file.csv", streaming=True)
streaming_dataset = loader.load()

# Process in chunks
for batch in streaming_dataset.iter_batches(batch_size=1000):
    # Process each batch
    processed_batch = process_function(batch)
    # Do something with the processed batch
    save_results(processed_batch)
```

### Caching

```python
from llamadatasets import DataLoader, CacheConfig

# Configure caching
cache_config = CacheConfig(
    enabled=True,
    location="/tmp/llamadatasets_cache",
    expiration=3600  # Cache expires after 1 hour
)

# Use caching with data loading
loader = DataLoader.from_csv(
    "data.csv",
    cache_config=cache_config,
    cache_key="my_dataset_v1"
)

# First call loads from file, subsequent calls use cache
dataset = loader.load()
```

## Examples

See the [examples](examples/) directory for more detailed examples, including:

- Text classification datasets
- Time series forecasting
- Image datasets
- Recommendation systems data
- Natural language processing datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT 