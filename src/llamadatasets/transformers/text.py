"""
Text transformers for processing text data in datasets
"""
from typing import Dict, Any, List, Optional, Union, Callable
import re
import string
import unicodedata
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

from llamadatasets.transformers.base import BaseTransformer, ColumnTransformer


class TextCleanerTransformer(ColumnTransformer):
    """
    Transformer that cleans text data by removing unwanted characters,
    normalizing whitespace, etc.
    """
    
    def __init__(self, 
                 columns: Union[str, List[str]],
                 target_columns: Optional[Union[str, List[str]]] = None,
                 lower: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_whitespace: bool = True,
                 normalize_unicode: bool = True,
                 remove_html: bool = False,
                 skip_missing: bool = False):
        """
        Initialize the text cleaner transformer.
        
        Args:
            columns: Column(s) containing text to clean
            target_columns: Optional target column(s) for the cleaned text
            lower: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
            remove_whitespace: Whether to normalize whitespace
            normalize_unicode: Whether to normalize Unicode characters
            remove_html: Whether to remove HTML tags
            skip_missing: Whether to skip missing columns
        """
        self.lower = lower
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_whitespace = remove_whitespace
        self.normalize_unicode = normalize_unicode
        self.remove_html = remove_html
        
        # HTML pattern
        self.html_pattern = re.compile(r'<.*?>')
        
        # Call parent init with a function that applies all selected transformations
        super().__init__(
            columns=columns,
            func=self._clean_text,
            target_columns=target_columns,
            skip_missing=skip_missing
        )
    
    def _clean_text(self, text: str) -> str:
        """
        Apply all configured cleaners to the text.
        
        Args:
            text: The input text to clean
            
        Returns:
            str: The cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Apply each cleaning step if enabled
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        if self.remove_html:
            text = self.html_pattern.sub('', text)
        
        if self.lower:
            text = text.lower()
        
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.remove_numbers:
            text = ''.join([c for c in text if not c.isdigit()])
        
        if self.remove_whitespace:
            text = ' '.join(text.split())
        
        return text


class TokenizerTransformer(ColumnTransformer):
    """
    Transformer that tokenizes text into words or subword units.
    """
    
    def __init__(self, 
                 columns: Union[str, List[str]],
                 target_columns: Optional[Union[str, List[str]]] = None,
                 tokenizer: str = 'word',
                 skip_missing: bool = False,
                 language: str = 'english',
                 custom_tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize the tokenizer transformer.
        
        Args:
            columns: Column(s) containing text to tokenize
            target_columns: Optional target column(s) for tokenized output
            tokenizer: Type of tokenizer ('word', 'char', 'subword')
            skip_missing: Whether to skip missing columns
            language: Language for NLTK tokenizer
            custom_tokenizer: Optional custom tokenization function
        """
        self.tokenizer_type = tokenizer
        self.language = language
        self.custom_tokenizer = custom_tokenizer
        
        # Try to download NLTK resources if needed
        try:
            if tokenizer == 'word':
                nltk.download('punkt', quiet=True)
        except:
            pass  # If download fails, we'll use a fallback
        
        # Call parent init with the appropriate tokenizer function
        super().__init__(
            columns=columns,
            func=self._tokenize_text,
            target_columns=target_columns,
            skip_missing=skip_missing
        )
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using the configured tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Use custom tokenizer if provided
        if self.custom_tokenizer is not None:
            return self.custom_tokenizer(text)
        
        # Otherwise use the selected tokenizer type
        if self.tokenizer_type == 'word':
            try:
                return word_tokenize(text, language=self.language)
            except:
                # Fallback to simple splitting
                return text.split()
        elif self.tokenizer_type == 'char':
            return list(text)
        elif self.tokenizer_type == 'subword':
            # Simple whitespace and punctuation split
            tokens = []
            for word in text.split():
                # Split by punctuation but keep it
                for char in string.punctuation:
                    if char in word:
                        parts = word.split(char)
                        for i, part in enumerate(parts):
                            if part:
                                tokens.append(part)
                            if i < len(parts) - 1:
                                tokens.append(char)
                        break
                else:
                    tokens.append(word)
            return tokens
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")


class StopWordsRemoverTransformer(BaseTransformer):
    """
    Transformer that removes stop words from tokenized text.
    """
    
    def __init__(self, 
                 column: str,
                 target_column: Optional[str] = None,
                 language: str = 'english',
                 additional_stopwords: Optional[List[str]] = None,
                 skip_missing: bool = False):
        """
        Initialize the stop words remover.
        
        Args:
            column: Column containing tokenized text
            target_column: Optional target column for filtered tokens
            language: Language for stop words list
            additional_stopwords: Optional additional stop words
            skip_missing: Whether to skip missing columns
        """
        self.column = column
        self.target_column = target_column or column
        self.language = language
        self.skip_missing = skip_missing
        
        # Get stop words
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words(language))
        except:
            # Fallback to a small set of common English stop words
            self.stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 
                                  'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 
                                  'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 
                                  'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                                  'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 
                                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
                                  'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 
                                  'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
                                  'into', 'through', 'during', 'before', 'after', 'above', 'below', 
                                  'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                                  'under', 'again', 'further', 'then', 'once', 'here', 'there', 
                                  'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 
                                  'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
                                  'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 
                                  't', 'can', 'will', 'just', 'don', 'should', 'now'])
        
        # Add any additional stopwords
        if additional_stopwords:
            self.stop_words.update(additional_stopwords)
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove stop words from tokenized text.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        result = dict(example)  # Create a copy
        
        if self.column not in example:
            if self.skip_missing:
                return result
            else:
                raise KeyError(f"Column '{self.column}' not found in example")
        
        tokens = example[self.column]
        
        # Ensure tokens is a list
        if not isinstance(tokens, list):
            raise ValueError(f"Expected list of tokens in column '{self.column}', got {type(tokens)}")
        
        # Remove stop words
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        result[self.target_column] = filtered_tokens
        return result


class TextStemmerTransformer(BaseTransformer):
    """
    Transformer that stems tokenized text.
    """
    
    def __init__(self, 
                 column: str,
                 target_column: Optional[str] = None,
                 skip_missing: bool = False):
        """
        Initialize the stemmer transformer.
        
        Args:
            column: Column containing tokenized text
            target_column: Optional target column for stemmed tokens
            skip_missing: Whether to skip missing columns
        """
        self.column = column
        self.target_column = target_column or column
        self.skip_missing = skip_missing
        self.stemmer = PorterStemmer()
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Stem tokens in the specified column.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        result = dict(example)  # Create a copy
        
        if self.column not in example:
            if self.skip_missing:
                return result
            else:
                raise KeyError(f"Column '{self.column}' not found in example")
        
        tokens = example[self.column]
        
        # Ensure tokens is a list
        if not isinstance(tokens, list):
            raise ValueError(f"Expected list of tokens in column '{self.column}', got {type(tokens)}")
        
        # Apply stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        
        result[self.target_column] = stemmed_tokens
        return result


class TextLemmatizerTransformer(BaseTransformer):
    """
    Transformer that lemmatizes tokenized text.
    """
    
    def __init__(self, 
                 column: str,
                 target_column: Optional[str] = None,
                 skip_missing: bool = False):
        """
        Initialize the lemmatizer transformer.
        
        Args:
            column: Column containing tokenized text
            target_column: Optional target column for lemmatized tokens
            skip_missing: Whether to skip missing columns
        """
        self.column = column
        self.target_column = target_column or column
        self.skip_missing = skip_missing
        
        # Initialize lemmatizer
        try:
            nltk.download('wordnet', quiet=True)
            self.lemmatizer = WordNetLemmatizer()
        except:
            # If WordNet is not available, use a simple identity function
            self.lemmatizer = type('', (), {'lemmatize': lambda self, word, pos='n': word})()
    
    def transform(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lemmatize tokens in the specified column.
        
        Args:
            example: The input example
            
        Returns:
            Dict[str, Any]: The transformed example
        """
        result = dict(example)  # Create a copy
        
        if self.column not in example:
            if self.skip_missing:
                return result
            else:
                raise KeyError(f"Column '{self.column}' not found in example")
        
        tokens = example[self.column]
        
        # Ensure tokens is a list
        if not isinstance(tokens, list):
            raise ValueError(f"Expected list of tokens in column '{self.column}', got {type(tokens)}")
        
        # Apply lemmatization
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        result[self.target_column] = lemmatized_tokens
        return result 