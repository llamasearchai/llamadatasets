#!/usr/bin/env python
"""
Synthetic data generation example using generators
"""
import os
import logging
import json
import random
from datetime import datetime

from llamadatasets import (
    RandomTextGenerator,
    TemplateTextGenerator,
    TextCleanerTransformer,
    TokenizerTransformer,
    StratifiedSplitter
)

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main example function
    """
    # Create a directory for sample data if it doesn't exist
    sample_dir = os.path.join(os.path.dirname(__file__), 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Example 1: Random Text Generator
    logger.info("Example 1: Random Text Generator")
    
    # Configure a generator for random product reviews
    random_generator = RandomTextGenerator(
        min_words=10,
        max_words=100,
        word_length_range=(3, 12),
        include_punctuation=True,
        categories=['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'],
        include_metadata=True,
        seed=42
    )
    
    # Generate random examples
    random_dataset = random_generator.generate(num_examples=50)
    logger.info(f"Generated {len(random_dataset)} random examples")
    
    # Show a sample
    logger.info("Sample random text:")
    for i in range(3):
        example = random_dataset[i]
        logger.info(f"  Example {i+1} ({example['category']}): {example['text'][:100]}...")
    
    # Save the dataset
    output_file = os.path.join(sample_dir, 'random_reviews.json')
    random_dataset.save(output_file, format='json')
    logger.info(f"Saved random dataset to {output_file}")
    
    # Example 2: Template-based Generator
    logger.info("\nExample 2: Template-based Generator")
    
    # Define templates and variables for customer support queries
    templates = [
        "I need help with my {product}. It {issue} and I've tried {solution} but it didn't work.",
        "My {product} is {issue}. I've had it for {timeperiod} and this is the first time it happened.",
        "How do I {action} on my {product}? I've looked at the manual but it doesn't cover this.",
        "When I try to {action}, my {product} {issue}. Is this covered by the warranty?",
        "I purchased a {product} {timeperiod} ago and it's already {issue}. Can I get a refund?"
    ]
    
    variables = {
        "product": [
            "smartphone", "laptop", "tablet", "smartwatch", "headphones",
            "TV", "game console", "wireless earbuds", "router", "smart speaker"
        ],
        "issue": [
            "won't turn on", "keeps crashing", "overheats", "has a cracked screen",
            "won't charge", "makes strange noises", "has poor battery life",
            "is extremely slow", "doesn't connect to WiFi", "has display issues"
        ],
        "solution": [
            "resetting it", "updating the firmware", "replacing the battery",
            "clearing the cache", "reinstalling the operating system",
            "contacting customer support", "checking online forums",
            "following the troubleshooting guide", "asking a friend for help"
        ],
        "action": [
            "reset the password", "update the software", "connect to Bluetooth",
            "set up cloud backup", "transfer files", "sync with my other devices",
            "activate voice control", "configure parental controls", 
            "extend the warranty", "claim the insurance"
        ],
        "timeperiod": [
            "one week", "a month", "three months", "six months", "a year",
            "two years", "just a few days", "less than a month", "nearly a year"
        ]
    }
    
    # Add additional fields
    fields = {
        "priority": ["low", "medium", "high", "urgent"],
        "customer_id": lambda: f"CUST-{random.randint(10000, 99999)}",
        "date_submitted": lambda: datetime.now().strftime("%Y-%m-%d"),
        "response_time_hours": lambda: round(random.uniform(0, 72), 1)
    }
    
    # Initialize the template generator
    template_generator = TemplateTextGenerator(
        templates=templates,
        variables=variables,
        fields=fields,
        seed=123
    )
    
    # Generate examples
    template_dataset = template_generator.generate(num_examples=100)
    logger.info(f"Generated {len(template_dataset)} template-based examples")
    
    # Show a sample
    logger.info("Sample template-based texts:")
    for i in range(3):
        example = template_dataset[i]
        logger.info(f"  Example {i+1} (Priority: {example['priority']}): {example['text']}")
        logger.info(f"    Customer ID: {example['customer_id']}, Response Time: {example['response_time_hours']} hours")
    
    # Apply transformations to the dataset
    logger.info("\nApplying transformations to the template dataset")
    
    # Clean text
    text_cleaner = TextCleanerTransformer(
        columns='text',
        target_columns='cleaned_text'
    )
    template_dataset = text_cleaner(template_dataset)
    
    # Tokenize
    tokenizer = TokenizerTransformer(
        columns='cleaned_text',
        target_columns='tokens'
    )
    template_dataset = tokenizer(template_dataset)
    
    # Show a transformed example
    logger.info("Sample after transformation:")
    logger.info(f"  Original: {template_dataset[0]['text']}")
    logger.info(f"  Cleaned: {template_dataset[0]['cleaned_text']}")
    logger.info(f"  Tokens: {template_dataset[0]['tokens'][:10]}...")
    
    # Split the dataset by priority
    logger.info("\nSplitting dataset by priority")
    splitter = StratifiedSplitter(
        label_column='priority',
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        seed=42
    )
    
    splits = splitter.split(template_dataset)
    
    # Show distribution in each split
    for split_name, split_dataset in splits.items():
        priority_counts = {}
        for example in split_dataset:
            priority = example['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        logger.info(f"  {split_name.capitalize()} split ({len(split_dataset)} examples): {priority_counts}")
    
    # Save the processed dataset
    output_file = os.path.join(sample_dir, 'support_queries.json')
    template_dataset.save(output_file, format='json')
    logger.info(f"Saved template-based dataset to {output_file}")
    
    # Create a dataset with mixed sources
    logger.info("\nCreating a combined dataset from multiple sources")
    
    # Add a source field to each dataset
    for example in random_dataset:
        example['source'] = 'random'
    
    for example in template_dataset:
        example['source'] = 'template'
    
    # Combine datasets
    from llamadatasets import Dataset
    combined_examples = random_dataset.to_dict_list() + template_dataset.to_dict_list()
    combined_dataset = Dataset(combined_examples)
    
    logger.info(f"Combined dataset has {len(combined_dataset)} examples from multiple sources")
    
    # Save the combined dataset
    output_file = os.path.join(sample_dir, 'combined_dataset.json')
    combined_dataset.save(output_file, format='json')
    logger.info(f"Saved combined dataset to {output_file}")


if __name__ == "__main__":
    main() 