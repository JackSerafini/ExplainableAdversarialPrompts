import torch
from prompt import PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM

class AdversarialPromptGenerator:
    def __init__(
        self, 
        prompt:str=PROMPT, 
        tokenizer_model_name:str="meta-llama/Llama-3.2-1B-Instruct"     # HF model name
    ):
        assert isinstance(prompt, str), "prompt must be a string"
        assert isinstance(tokenizer_model_name, str), "tokenizer_model_name must be a string"

        self.prompt = prompt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)

    def get_from(self, adversarial_suffix_path:str):
        assert isinstance(adversarial_suffix_path, str), "adversarial_suffix_path must be a string"

        adv_suffixes = torch.load(adversarial_suffix_path)
        adv_suffixes_decoded = self.tokenizer.batch_decode(adv_suffixes)
        return [self.prompt + adv_suffix for adv_suffix in adv_suffixes_decoded]