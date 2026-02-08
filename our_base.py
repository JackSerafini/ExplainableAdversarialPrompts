# base.py

from typing import Optional, List, Dict, Tuple, Any, Callable, Set
import base64
import numpy as np
import pandas as pd
import os
import json
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
import random
import torch
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def default_output_handler(message: str) -> None:
    """Prints messages without newline."""
    print(message, end='', flush=True)

def encode_image_to_base64(image_path: str) -> str:
    """
    Converts an image file to a base64-encoded string.
    
    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Base64-encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_text_before_last_underscore(token):
    return token.rsplit('_', 1)[0]

class TextVectorizer:
    """Base class for text vectorization"""
    
    def vectorize(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class ModelBase(ABC):
    """Base class for all models (text and vision)"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, api_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.client = None
        
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from model"""
        pass

    def generate_with_tools(self,
                           prompt: str,
                           tools: List[Dict],
                           tool_executor: Optional[Callable[[str, Dict], str]] = None,
                           max_iterations: int = 10) -> Tuple[str, Dict[str, int]]:
        """
        Generate response with tool calling support.

        Args:
            prompt: The user prompt
            tools: List of tool definitions (API format)
            tool_executor: Callable(tool_name, args) -> result string
            max_iterations: Max tool call iterations

        Returns:
            Tuple of (response_text, tool_usage_counts)

        Note: Subclasses should override this for proper tool support.
        Default implementation just calls generate() without tools.
        """
        # Default: no tool support, just generate
        return self.generate(prompt), {}

class HuggingFaceEmbeddings(TextVectorizer):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        """
        Initialize HuggingFace sentence embeddings vectorizer - much simpler implementation
        
        Args:
            model_name: Name of the sentence-transformer model from HuggingFace
            device: Device to run model on ('cpu' or 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialize_model()
        
    def _initialize_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            # Load model - SentenceTransformer handles all the complexity
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except ImportError:
            raise ImportError("sentence-transformers package not installed. Please install with 'pip install sentence-transformers'")
            
    def vectorize(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using sentence-transformers - much simpler"""
        if not self.model:
            self._initialize_model()
            
        # SentenceTransformer handles batching, padding, etc. automatically
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity between vectors"""
        # Sentence-transformers models already return normalized vectors
        return np.dot(comparison_vectors, base_vector)

class TfidfTextVectorizer(TextVectorizer):
    def __init__(self):
        self.vectorizer = None
        
    def vectorize(self, texts: List[str]) -> np.ndarray:
        self.vectorizer = TfidfVectorizer().fit(texts)
        return self.vectorizer.transform(texts).toarray()
        
    def calculate_similarity(self, base_vector: np.ndarray, comparison_vectors: np.ndarray) -> np.ndarray:
        return cosine_similarity(
            base_vector.reshape(1, -1), comparison_vectors
        ).flatten()

class LocalModel(ModelBase):
    """Local model implementation supporting both text and vision using HuggingFace models"""
    
    def __init__(self, 
                model_name: str,
                model_type: str = "text",  # "text" or "vision"
                max_new_tokens: int = 100,
                temperature: float = 0.5,
                device: str = "auto",
                dtype: Optional[str] = "bfloat16",
                **model_kwargs):
        """
        Initialize local model
        
        Args:
            model_name: HuggingFace model name/path
            model_type: "text" or "vision"
            max_new_tokens: Maximum new tokens to generate
            temperature: Generation temperature
            device: Device to run model on ("auto", "cuda", "cpu")
            dtype: Torch data type for model
            **model_kwargs: Additional kwargs for model initialization
        """
        super().__init__(model_name)
        self.device = device
        self.model_type = model_type
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.dtype = getattr(torch, dtype) if dtype else None
        self.model_kwargs = model_kwargs
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the appropriate model and tokenizer/processor"""
        try:
            if self.model_type == "text":
                from transformers import AutoTokenizer, AutoModelForCausalLM
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    dtype=self.dtype,
                    device_map=self.device,
                    **self.model_kwargs
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            raise ImportError(f"Error initializing model: {str(e)}")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Text prompt
            
        Returns:
            Generated text response
        """
        try:
            # Text-only processing
            # inputs = self.tokenizer(prompt, return_tensors="pt")
            # CUSTOM TOKENIZER
            inputs = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]

            # outputs = self.model.generate(
            #     **inputs,
            #     max_new_tokens=self.max_new_tokens,
            #     temperature=self.temperature,
            #     pad_token_id=self.tokenizer.eos_token_id
            # )
            # CUSTOM GENERATION
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=self.temperature,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # Qua andiamo a prenderci esclusivamente i token generati dal modello
            # -> in outputs ci sono gli inputs con gli outputs appended
            generated_tokens = outputs[0][input_length:]
            return self.tokenizer.decode( # Converts token IDs back into text
                generated_tokens,
                skip_special_tokens=True # Removes special tokens like: <s>, <|eos|>, ...
            ).strip() # Removes leading/trailing whitespace
                
        except Exception as e:
            raise RuntimeError(f"Error generating response: {str(e)}")

class BaseSHAP(ABC):
    """Base class for SHAP implementations"""
    
    def __init__(self, 
                 model: ModelBase,
                 vectorizer: Optional[TextVectorizer] = None,
                 debug: bool = False):
        self.model = model
        self.vectorizer = vectorizer
        self.debug = debug
        self.results_df = None
        self.shapley_values = None

    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled"""
        if self.debug:
            print(message)

    def _calculate_baseline(self, content: Any, **kwargs) -> str:
        """Calculate baseline model response"""
        # return self.model.generate(**self._prepare_generate_args(content, **kwargs))
        # CUSTOM RETURN
        return self.model.generate(self._prepare_generate_args(content, **kwargs))

    @abstractmethod
    def _prepare_generate_args(self, content: Any, **kwargs) -> Dict:
        """Prepare arguments for model.generate()"""
        pass

    def _generate_random_combinations(self,
                                    samples: List[Any],
                                    k: int,
                                    exclude_combinations_set: Set[Tuple[int, ...]]) -> List[Tuple[List, Tuple[int, ...]]]:
        """
        Generate random combinations efficiently using binary representation
        """
        n = len(samples)
        sampled_indexes_set = set()  # Track only indexes (hashable) for deduplication
        sampled_combinations = []  # Store actual combinations in a list
        max_attempts = k * 10  # Prevent infinite loops in case of duplicates
        attempts = 0

        while len(sampled_combinations) < k and attempts < max_attempts:
            attempts += 1
            rand_int = random.randint(1, 2 ** n - 2)
            bin_str = bin(rand_int)[2:].zfill(n)
            combination = [samples[i] for i in range(n) if bin_str[i] == '1']
            indexes = tuple([i + 1 for i in range(n) if bin_str[i] == '1'])
            if indexes not in exclude_combinations_set and indexes not in sampled_indexes_set:
                sampled_indexes_set.add(indexes)
                sampled_combinations.append((combination, indexes))

        if len(sampled_combinations) < k:
            self._debug_print(f"Warning: Could only generate {len(sampled_combinations)} unique combinations out of requested {k}")
        return sampled_combinations

    def _get_result_per_combination(self, 
                                content: Any, 
                                sampling_ratio: float,
                                max_combinations: Optional[int] = 1000) -> Dict[str, Tuple[str, Tuple[int, ...]]]:
        """
        Get model responses for combinations
        
        Args:
            content: Content to analyze
            sampling_ratio: Ratio of non-essential combinations to sample (0-1)
            max_combinations: Maximum number of combinations (must be >= n for n tokens)
        """
        # Dividi il prompt secondo lo Splitter
        samples = self._get_samples(content)
        n = len(samples)

        if n > 1000:
            print("Warning: the number of samples is greater than 1000; execution will be slow.")

        # Always start with essential combinations (each missing one sample)
        essential_combinations = []
        essential_combinations_set = set()
        for i in range(n):
            combination = samples[:i] + samples[i + 1:]
            indexes = tuple([j + 1 for j in range(n) if j != i])
            essential_combinations.append((combination, indexes))
            essential_combinations_set.add(indexes)

        num_essential = len(essential_combinations)
        if max_combinations is not None and max_combinations < num_essential:
            print(f"Warning: max_combinations ({max_combinations}) is less than the number of essential combinations "
                  f"({num_essential}). Will use all essential combinations despite the limit.")
            max_combinations = num_essential

        # Calculate how many additional combinations we can/should generate
        remaining_budget = float('inf')
        if max_combinations is not None:
            remaining_budget = max(0, max_combinations - num_essential)

        # If using sampling ratio, calculate possible additional combinations without generating them
        if sampling_ratio < 1.0:
            # Get theoretical number of total combinations
            theoretical_total = 2 ** n - 1
            theoretical_additional = theoretical_total - num_essential
            # Calculate desired number based on ratio
            desired_additional = int(theoretical_additional * sampling_ratio)
            # Take minimum of sampling ratio and max_combinations limits
            num_additional = min(desired_additional, remaining_budget)
        else:
            num_additional = remaining_budget

        num_additional = int(num_additional)  # Ensure integer

        # Generate additional random combinations if needed
        additional_combinations = []
        if num_additional > 0:
            additional_combinations = self._generate_random_combinations(
                samples, num_additional, essential_combinations_set
            )

        # Process all combinations
        all_combinations = essential_combinations + additional_combinations

        responses = {}
        for idx, (combination, indexes) in enumerate(tqdm(all_combinations, desc="Processing combinations")):
            args = self._prepare_combination_args(combination, content)
            # response = self.model.generate(**args)
            # CUSTOM RESPONSES
            response = self.model.generate(args)

            key = self._get_combination_key(combination, indexes)
            responses[key] = (response, indexes)

        return responses

    def _get_df_per_combination(self, responses: Dict[str, Tuple[str, Tuple[int, ...]]], baseline_text: str) -> pd.DataFrame:
        """Create DataFrame with combination results"""
        df = pd.DataFrame(
            [(key.split('_')[0], response[0], response[1])
             for key, response in responses.items()],
            columns=['Content', 'Response', 'Indexes']
        )

        all_texts = [baseline_text] + df["Response"].tolist()
        vectors = self.vectorizer.vectorize(all_texts)
        base_vector = vectors[0]
        comparison_vectors = vectors[1:]
        
        similarities = self.vectorizer.calculate_similarity(base_vector, comparison_vectors)
        df["Similarity"] = similarities

        return df

    def _calculate_shapley_values(self, df: pd.DataFrame, content: Any) -> Dict[str, float]:
        """Calculate Shapley values"""
        samples = self._get_samples(content)
        shapley_values = {}

        def normalize_shapley_values(values: Dict[str, float], power: float = 1) -> Dict[str, float]:
            min_value = min(values.values())
            shifted_values = {k: v - min_value for k, v in values.items()}
            powered_values = {k: v ** power for k, v in shifted_values.items()}
            total = sum(powered_values.values())
            if total == 0:
                return {k: 1 / len(powered_values) for k in powered_values}
            return {k: v / total for k, v in powered_values.items()}

        for i, sample in enumerate(samples, start=1):
            with_sample = np.average(
                df[df["Indexes"].apply(lambda x: i in x)]["Similarity"].values
            )
            without_sample = np.average(
                df[df["Indexes"].apply(lambda x: i not in x)]["Similarity"].values
            )

            shapley_values[f"{sample}_{i}"] = with_sample - without_sample

        return normalize_shapley_values(shapley_values)

    @abstractmethod
    def _get_samples(self, content: Any) -> List[Any]:
        """Get samples from content for analysis"""
        pass

    @abstractmethod
    def _prepare_combination_args(self, combination: List[Any], original_content: Any) -> Dict:
        """Prepare model arguments for a combination"""
        pass

    @abstractmethod
    def _get_combination_key(self, combination: List[Any], indexes: Tuple[int, ...]) -> str:
        """Get unique key for combination"""
        pass

    def save_results(self, output_dir: str, metadata: Optional[Dict] = None) -> None:
        """Save analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results_df is not None:
            self.results_df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
            
        if self.shapley_values is not None:
            with open(os.path.join(output_dir, "shapley_values.json"), 'w') as f:
                json.dump(self.shapley_values, f, indent=2)
                
        if metadata:
            with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)