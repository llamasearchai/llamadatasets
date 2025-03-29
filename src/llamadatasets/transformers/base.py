"""
Base class for data transformers
"""
from typing import Dict, Any, Callable, List, Union, Optional
from abc import ABC, abstractmethod


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.
    
    Data transformers modify dataset examples, either changing existing fields
    or adding new ones.
    """
    
    @abstractmethod
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single example.
        
        Args:
            example: The input example to transform
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        pass
    
    def batch_transform(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform a batch of examples.
        
        The default implementation applies transform to each example individually.
        Override this method for more efficient batch processing.
        
        Args:
            examples: List of input examples
            
        Returns:
            List[Dict[str, Any]]: List of transformed examples
        """
        return [self.transform(example) for example in examples]
    
    def __call__(self, example_or_batch: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply the transformer to an example or batch of examples.
        
        Args:
            example_or_batch: Either a single example or a list of examples
            
        Returns:
            Union[Dict[str, Any], List[Dict[str, Any]]]: Transformed example(s)
        """
        if isinstance(example_or_batch, list):
            return self.batch_transform(example_or_batch)
        else:
            return self.transform(example_or_batch)


class FunctionTransformer(BaseTransformer):
    """
    A transformer that applies a given function to examples.
    """
    
    def __init__(self, func: Callable[[Dict[str, Any]], Dict[str, Any]], batch_func: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None):
        """
        Initialize the transformer with a transformation function.
        
        Args:
            func: Function that transforms a single example
            batch_func: Optional function for batch processing
        """
        self.func = func
        self.batch_func = batch_func
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transformation function to an example.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        return self.func(example)
    
    def batch_transform(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the transformation to a batch of examples.
        
        Args:
            examples: List of input examples
            
        Returns:
            List[Dict[str, Any]]: List of transformed examples
        """
        if self.batch_func is not None:
            return self.batch_func(examples)
        else:
            return super().batch_transform(examples)


class ColumnTransformer(BaseTransformer):
    """
    A transformer that applies a function to specific columns.
    """
    
    def __init__(self, 
                 columns: Union[str, List[str]], 
                 func: Callable[[Any], Any],
                 target_columns: Optional[Union[str, List[str]]] = None,
                 skip_missing: bool = False):
        """
        Initialize the transformer.
        
        Args:
            columns: Column(s) to transform
            func: Function to apply to the column values
            target_columns: Optional target column(s) for the output (if None, overwrites input columns)
            skip_missing: Whether to skip missing columns instead of raising an error
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        
        if target_columns is None:
            self.target_columns = self.columns
        else:
            self.target_columns = [target_columns] if isinstance(target_columns, str) else target_columns
            
        if len(self.target_columns) != len(self.columns):
            raise ValueError("target_columns must have the same length as columns")
            
        self.func = func
        self.skip_missing = skip_missing
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transformation function to specified columns.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        result = dict(example)  # Create a copy
        
        for i, col in enumerate(self.columns):
            target_col = self.target_columns[i]
            
            if col not in example:
                if self.skip_missing:
                    continue
                else:
                    raise KeyError(f"Column '{col}' not found in example")
                
            # Apply the transformation and store in the target column
            result[target_col] = self.func(example[col])
            
        return result


class ChainTransformer(BaseTransformer):
    """
    A transformer that chains multiple transformers together.
    """
    
    def __init__(self, transformers: List[BaseTransformer]):
        """
        Initialize the transformer with a list of transformers to chain.
        
        Args:
            transformers: List of transformers to apply in sequence
        """
        self.transformers = transformers
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply each transformer in sequence.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        result = example
        for transformer in self.transformers:
            result = transformer.transform(result)
        return result
    
    def batch_transform(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply each transformer in sequence to the batch.
        
        Args:
            examples: List of input examples
            
        Returns:
            List[Dict[str, Any]]: List of transformed examples
        """
        result = examples
        for transformer in self.transformers:
            result = transformer.batch_transform(result)
        return result 