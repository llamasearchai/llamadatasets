#!/usr/bin/env python
"""
Basic usage example for llamadatasets package
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

from llamadatasets import (
    DataLoader, 
    CacheConfig,
    TextCleanerTransformer,
    TokenizerTransformer,
    StopWordsRemoverTransformer,
    RandomSplitter
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_csv(filename):
    """
    Create a sample CSV file for demonstration
    """
    # Create a sample dataset
    data = {
        'id': list(range(1, 101)),
        'text': [
            f"This is sample text {i} with some variation and punctuation!!!" 
            for i in range(1, 101)
        ],
        'category': np.random.choice(['business', 'tech', 'health', 'entertainment'], 100),
        'timestamp': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S') 
            for _ in range(100)
        ],
        'score': np.random.uniform(0, 10, 100).round(2)
    }
    
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logger.info(f"Created sample CSV file at {filename}")
    return filename


def create_sample_json(filename):
    """
    Create a sample JSON file for demonstration
    """
    # Create a sample dataset
    data = []
    for i in range(1, 101):
        data.append({
            'id': i,
            'text': f"This is sample JSON text {i} with more information and details.",
            'category': np.random.choice(['business', 'tech', 'health', 'entertainment']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'score': round(np.random.uniform(0, 10), 2),
            'metadata': {
                'source': 'example',
                'version': '1.0',
                'tags': ['sample', 'demo', 'test']
            }
        })
    
    # Save to JSON
    with open(filename, 'w') as f:
        json.dump(data, f)
    
    logger.info(f"Created sample JSON file at {filename}")
    return filename


def main():
    """
    Main example function
    """
    # Create a directory for sample data if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample files
    csv_file = os.path.join(sample_dir, 'sample.csv')
    json_file = os.path.join(sample_dir, 'sample.json')
    create_sample_csv(csv_file)
    create_sample_json(json_file)
    
    # Example 1: Basic CSV loading
    logger.info("Example 1: Basic CSV loading")
    loader = DataLoader.from_csv(csv_file)
    dataset = loader.load()
    logger.info(f"Loaded CSV dataset with {len(dataset)} examples")
    logger.info(f"First example: {dataset[0]}")
    
    # Example 2: Loading with caching
    logger.info("\nExample 2: Loading with caching")
    cache_config = CacheConfig(
        enabled=True,
        location=os.path.join(sample_dir, 'cache'),
        expiration=3600  # 1 hour
    )
    loader = DataLoader.from_json(
        json_file,
        cache_config=cache_config
    )
    dataset = loader.load()
    logger.info(f"Loaded JSON dataset with {len(dataset)} examples")
    logger.info(f"First example: {dataset[0]}")
    
    # Example 3: Data transformations
    logger.info("\nExample 3: Data transformations")
    
    # Clean text
    text_cleaner = TextCleanerTransformer(
        columns='text',
        target_columns='cleaned_text',
        lower=True,
        remove_punctuation=True
    )
    dataset = text_cleaner(dataset)
    logger.info(f"After text cleaning, example has cleaned_text: {dataset[0]['cleaned_text']}")
    
    # Tokenize
    tokenizer = TokenizerTransformer(
        columns='cleaned_text',
        target_columns='tokens'
    )
    dataset = tokenizer(dataset)
    logger.info(f"After tokenization, example has tokens: {dataset[0]['tokens']}")
    
    # Remove stop words
    stop_words_remover = StopWordsRemoverTransformer(
        column='tokens',
        target_column='filtered_tokens'
    )
    dataset = stop_words_remover(dataset)
    logger.info(f"After stop words removal, example has filtered_tokens: {dataset[0]['filtered_tokens']}")
    
    # Example 4: Filtering and mapping
    logger.info("\nExample 4: Filtering and mapping")
    
    # Filter for tech category
    tech_dataset = dataset.filter(lambda x: x['category'] == 'tech')
    logger.info(f"After filtering for tech category, dataset has {len(tech_dataset)} examples")
    
    # Map to create a new field
    def add_token_count(example):
        example['token_count'] = len(example['filtered_tokens'])
        return example
    
    dataset_with_counts = dataset.map(add_token_count)
    logger.info(f"After mapping, example has token_count: {dataset_with_counts[0]['token_count']}")
    
    # Example 5: Dataset splitting
    logger.info("\nExample 5: Dataset splitting")
    
    # Random split
    splitter = RandomSplitter(
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=42
    )
    splits = splitter.split(dataset)
    
    logger.info(f"Split dataset into:")
    logger.info(f"  - Train: {len(splits['train'])} examples")
    logger.info(f"  - Validation: {len(splits['val'])} examples")
    logger.info(f"  - Test: {len(splits['test'])} examples")
    
    # Example 6: Save modified dataset
    logger.info("\nExample 6: Save modified dataset")
    output_file = os.path.join(sample_dir, 'processed_data.json')
    splits['train'].save(output_file, format='json')
    logger.info(f"Saved processed training data to {output_file}")


if __name__ == "__main__":
    main() 