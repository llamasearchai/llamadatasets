"""
Synthetic text data generators for creating test and training datasets
"""
from typing import List, Dict, Any, Optional, Union, Callable
import random
import string
import numpy as np
from abc import ABC, abstractmethod

from llamadatasets.core.dataset import Dataset


class BaseTextGenerator(ABC):
    """
    Abstract base class for text data generators
    """
    
    @abstractmethod
    def generate_example(self) -> Dict[str, Any]:
        """
        Generate a single example.
        
        Returns:
            Dict[str, Any]: A dictionary representing a single example
        """
        pass
    
    def generate(self, num_examples: int) -> Dataset:
        """
        Generate multiple examples and return as a dataset.
        
        Args:
            num_examples: Number of examples to generate
            
        Returns:
            Dataset: A dataset containing the generated examples
        """
        examples = [self.generate_example() for _ in range(num_examples)]
        return Dataset(examples)


class RandomTextGenerator(BaseTextGenerator):
    """
    Generator that creates random text data with configurable properties
    """
    
    def __init__(self, 
                 min_words: int = 5,
                 max_words: int = 50,
                 word_length_range: tuple = (2, 10),
                 include_punctuation: bool = True,
                 categories: Optional[List[str]] = None,
                 include_metadata: bool = False,
                 seed: Optional[int] = None):
        """
        Initialize the random text generator.
        
        Args:
            min_words: Minimum number of words in generated text
            max_words: Maximum number of words in generated text
            word_length_range: Range of word lengths (min, max)
            include_punctuation: Whether to include punctuation
            categories: List of categories to choose from (if None, no category field)
            include_metadata: Whether to include random metadata
            seed: Random seed for reproducibility
        """
        self.min_words = min_words
        self.max_words = max_words
        self.word_length_range = word_length_range
        self.include_punctuation = include_punctuation
        self.categories = categories
        self.include_metadata = include_metadata
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Punctuation to choose from
        self.punctuation = ['.', ',', '!', '?', ';', ':']
    
    def _generate_word(self) -> str:
        """
        Generate a random word.
        
        Returns:
            str: A random word
        """
        length = random.randint(self.word_length_range[0], self.word_length_range[1])
        return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))
    
    def _generate_text(self) -> str:
        """
        Generate random text.
        
        Returns:
            str: Random text with the specified properties
        """
        num_words = random.randint(self.min_words, self.max_words)
        words = []
        
        for i in range(num_words):
            word = self._generate_word()
            
            # Capitalize first word or after punctuation
            if i == 0 or (words and words[-1].endswith(tuple(self.punctuation))):
                word = word.capitalize()
            
            words.append(word)
            
            # Add punctuation occasionally
            if self.include_punctuation and i < num_words - 1 and random.random() < 0.1:
                words[-1] += random.choice(self.punctuation)
        
        # Add final punctuation
        if self.include_punctuation:
            words[-1] += random.choice(['.', '!', '?'])
        
        return ' '.join(words)
    
    def generate_example(self) -> Dict[str, Any]:
        """
        Generate a single example with random text.
        
        Returns:
            Dict[str, Any]: A dictionary with random text and optional fields
        """
        example = {
            'id': random.randint(1, 1000000),
            'text': self._generate_text()
        }
        
        # Add a category if specified
        if self.categories:
            example['category'] = random.choice(self.categories)
        
        # Add random metadata if specified
        if self.include_metadata:
            example['metadata'] = {
                'source': f'generator-{random.randint(1, 5)}',
                'created_at': f'2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
                'score': round(random.uniform(0, 10), 2)
            }
        
        return example


class TemplateTextGenerator(BaseTextGenerator):
    """
    Generator that creates text data from templates with variable substitution
    """
    
    def __init__(self, 
                 templates: List[str],
                 variables: Dict[str, List[str]],
                 fields: Optional[Dict[str, Union[List[Any], Callable]]] = None,
                 seed: Optional[int] = None):
        """
        Initialize the template text generator.
        
        Args:
            templates: List of text templates with {variable} placeholders
            variables: Dictionary mapping variable names to possible values
            fields: Additional fields to include in examples (name -> list of values or callable)
            seed: Random seed for reproducibility
        """
        self.templates = templates
        self.variables = variables
        self.fields = fields or {}
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Validate that all variables in templates exist in variables dict
        for template in templates:
            for var_name in self._extract_variables(template):
                if var_name not in variables:
                    raise ValueError(f"Variable '{var_name}' in template not found in variables dict")
    
    def _extract_variables(self, template: str) -> List[str]:
        """
        Extract variable names from a template string.
        
        Args:
            template: Template string with {variable} placeholders
            
        Returns:
            List[str]: List of variable names
        """
        import re
        return re.findall(r'\{(\w+)\}', template)
    
    def _fill_template(self, template: str) -> str:
        """
        Fill a template with random values for its variables.
        
        Args:
            template: Template string with {variable} placeholders
            
        Returns:
            str: Template with variables replaced by values
        """
        result = template
        for var_name in self._extract_variables(template):
            value = random.choice(self.variables[var_name])
            result = result.replace(f"{{{var_name}}}", value)
        
        return result
    
    def generate_example(self) -> Dict[str, Any]:
        """
        Generate a single example using templates.
        
        Returns:
            Dict[str, Any]: A dictionary with text generated from templates
        """
        template = random.choice(self.templates)
        text = self._fill_template(template)
        
        example = {
            'id': random.randint(1, 1000000),
            'text': text
        }
        
        # Add additional fields
        for field_name, field_values in self.fields.items():
            if callable(field_values):
                # Call the function to get a value
                example[field_name] = field_values()
            else:
                # Choose a random value from the list
                example[field_name] = random.choice(field_values)
        
        return example 