#!/usr/bin/env python
"""
Streaming dataset example for handling large datasets efficiently
"""
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

from llamadatasets import DataLoader, FunctionTransformer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_large_csv(filename, num_rows=100000):
    """
    Create a large CSV file for demonstrating streaming
    """
    # Create a sample dataset with many rows
    logger.info(f"Creating large CSV file with {num_rows} rows...")

    # Generate data in chunks to avoid memory issues
    chunk_size = 10000
    for chunk_idx in range(0, num_rows, chunk_size):
        end_idx = min(chunk_idx + chunk_size, num_rows)
        chunk_size_actual = end_idx - chunk_idx

        # Create chunk data
        data = {
            "id": list(range(chunk_idx + 1, end_idx + 1)),
            "text": [
                f"This is sample text {i} for streaming example with some random content {np.random.rand()}"
                for i in range(chunk_idx + 1, end_idx + 1)
            ],
            "category": np.random.choice(
                ["business", "tech", "health", "entertainment"], chunk_size_actual
            ),
            "timestamp": [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for _ in range(chunk_size_actual)
            ],
            "value1": np.random.normal(100, 15, chunk_size_actual).round(2),
            "value2": np.random.normal(50, 10, chunk_size_actual).round(2),
            "value3": np.random.normal(25, 5, chunk_size_actual).round(2),
        }

        # Create a DataFrame for this chunk
        df = pd.DataFrame(data)

        # Write to CSV (append mode after first chunk)
        if chunk_idx == 0:
            df.to_csv(filename, index=False, mode="w")
        else:
            df.to_csv(filename, index=False, mode="a", header=False)

        logger.info(
            f"  Wrote chunk {chunk_idx//chunk_size + 1}/{(num_rows-1)//chunk_size + 1}"
        )

    logger.info(f"Created large CSV file at {filename}")
    return filename


def main():
    """
    Main example function
    """
    # Create a directory for sample data if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
    os.makedirs(sample_dir, exist_ok=True)

    # Create sample file (a smaller size for demonstration)
    csv_file = os.path.join(sample_dir, "large_sample.csv")
    create_large_csv(csv_file, num_rows=50000)  # Use a smaller size for testing

    # Example 1: Memory usage comparison - Standard loading vs Streaming
    logger.info("\nExample 1: Memory usage comparison")

    # Standard loading (loads everything into memory)
    logger.info("Standard loading (loads entire dataset into memory):")
    start_time = time.time()
    loader = DataLoader.from_csv(csv_file)
    dataset = loader.load()
    load_time = time.time() - start_time
    logger.info(f"  Loaded {len(dataset)} examples in {load_time:.2f} seconds")
    logger.info(f"  First example: {dataset[0]}")

    # Streaming loading
    logger.info("\nStreaming loading (processes in chunks):")
    start_time = time.time()
    loader = DataLoader.from_csv(csv_file, streaming=True)
    streaming_dataset = loader.load()
    load_time = time.time() - start_time
    logger.info(f"  Initialized streaming dataset in {load_time:.2f} seconds")

    # Count rows by iterating
    start_time = time.time()
    count = 0
    for _ in streaming_dataset:
        count += 1
    iter_time = time.time() - start_time
    logger.info(f"  Iterated through {count} examples in {iter_time:.2f} seconds")

    # Example 2: Processing data in batches
    logger.info("\nExample 2: Processing data in batches")

    # Define a transformation to calculate derived values
    def process_example(example):
        # Add a derived field
        example["avg_value"] = (
            example["value1"] + example["value2"] + example["value3"]
        ) / 3
        return example

    # Initialize transformer
    transformer = FunctionTransformer(process_example)

    # Process in batches
    start_time = time.time()

    # Statistics counters
    example_count = 0
    sum_avg_value = 0

    # Process each batch
    logger.info("Processing in batches of 1000:")
    for i, batch in enumerate(streaming_dataset.iter_batches(batch_size=1000)):
        # Apply transformation to the batch
        transformed_batch = transformer.batch_transform(batch)

        # Update statistics
        for example in transformed_batch:
            example_count += 1
            sum_avg_value += example["avg_value"]

        # Log progress occasionally
        if (i + 1) % 10 == 0:
            logger.info(f"  Processed {example_count} examples so far...")

    process_time = time.time() - start_time

    # Calculate final statistics
    avg_of_avgs = sum_avg_value / example_count if example_count > 0 else 0

    logger.info(f"Processed {example_count} examples in {process_time:.2f} seconds")
    logger.info(f"Average of the 'avg_value' field: {avg_of_avgs:.2f}")

    # Example 3: Getting a sample of the data for inspection
    logger.info("\nExample 3: Getting a sample of the data for inspection")

    # Get the first 5 examples
    head_dataset = streaming_dataset.head(5)
    logger.info(f"First 5 examples:")
    for i, example in enumerate(head_dataset):
        logger.info(
            f"  Example {i+1}: id={example['id']}, category={example['category']}"
        )

    # Example 4: Converting a subset to a regular dataset
    logger.info("\nExample 4: Converting a subset to a regular dataset")

    # Convert the first 1000 examples to a regular dataset
    start_time = time.time()
    small_dataset = streaming_dataset.to_dataset(max_examples=1000)
    convert_time = time.time() - start_time

    logger.info(
        f"Converted first 1000 examples to regular dataset in {convert_time:.2f} seconds"
    )

    # Save this sample to a file
    output_file = os.path.join(sample_dir, "sample_subset.json")
    small_dataset.save(output_file, format="json")
    logger.info(f"Saved sample of 1000 examples to {output_file}")


if __name__ == "__main__":
    main()
