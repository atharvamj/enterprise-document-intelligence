"""
LLM interface for LLaMA 3 integration.
"""

from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from loguru import logger


class LLMInterface:
    """
    Interface for LLaMA 3 language model.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM interface.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_config = config['llm']
        self.model_name = self.llm_config['model_name']
        self.max_length = self.llm_config['max_length']
        self.temperature = self.llm_config['temperature']
        self.top_p = self.llm_config['top_p']
        self.top_k = self.llm_config['top_k']
        
        # Device configuration
        device = self.llm_config.get('device', 'auto')
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing LLM: {self.model_name} on {self.device}")
        
        # Load model with quantization if configured
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the language model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Configure quantization
            quantization_config = None
            if self.llm_config.get('use_4bit_quantization', False):
                logger.info("Using 4-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.llm_config.get('use_8bit_quantization', False):
                logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load model
            model_kwargs = {
                'device_map': 'auto' if self.device == 'cuda' else None,
                'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32,
            }
            
            if quantization_config:
                model_kwargs['quantization_config'] = quantization_config
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == 'cpu' and not quantization_config:
                self.model = self.model.to(self.device)
            
            logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            logger.warning("You may need to install the model or use a different model name")
            logger.warning("For local models, set model_name to the path of your model")
            raise
    
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                top_k: Optional[int] = None) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Use defaults from config if not specified
        max_new_tokens = max_new_tokens or self.llm_config.get('max_new_tokens', 512)
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        top_k = top_k or self.top_k
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                  max_length=self.max_length)
            
            # Move to device
            if self.device != 'cpu':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from output
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def format_prompt(self, system_prompt: str, user_message: str, 
                     context: Optional[str] = None) -> str:
        """
        Format prompt for LLaMA 3 chat format.
        
        Args:
            system_prompt: System instruction
            user_message: User query
            context: Retrieved context (optional)
            
        Returns:
            Formatted prompt
        """
        # LLaMA 3 chat format
        messages = []
        
        # Add system message
        if system_prompt:
            messages.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>")
        
        # Add context if provided
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {user_message}"
        else:
            user_content = user_message
        
        messages.append(f"<|start_header_id|>user<|end_header_id|>\n\n{user_content}<|eot_id|>")
        messages.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        return "".join(messages)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model info
        """
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'model_name': self.model_name,
            'device': self.device,
            'max_length': self.max_length,
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'quantized': self.llm_config.get('use_4bit_quantization') or 
                        self.llm_config.get('use_8bit_quantization')
        }
