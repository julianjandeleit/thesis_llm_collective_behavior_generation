

from dataclasses import dataclass
from pandas import DataFrame
from typing import Optional, Callable
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import DEFAULT_GENERATE_PROMPT, load_default_model
import random
import numpy as np

@dataclass
class CustomInference:
    text: Optional[str] = None # raw text pre formating for llm
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    temperature: Optional[float] = None
    seq_len: float = 2000
    generate_prompt: Callable[[str, str, 'CustomInference'], str] = DEFAULT_GENERATE_PROMPT

    def __post_init__(self):
        # load mathstral 7B by default
        if self.model is None or self.tokenizer is None:
            model, tokenizer, lora, bnb = load_default_model()
            self.model = model
            self.tokenizer = tokenizer

        if self.text is None:
            self.text = 'The environment features a rectangle with dimensions 5.57 x 2.19 x 1.75.There are the following lights in the arena: ((-0.25, 0.57)). 16 robots are evenly spaced around the central point, spanning a radius of 0.96 m. In the surroundings, there exists a circle at [-0.71, 0.76] with a radius of 0.25 meters, exhibiting a white color, and another at [0.78, -1.25] with a radius of 0.27 meters in black. The objective for the robots is to transfer items from the white initial location to the black circle. '


    def inference(self):
        """ assumes generate_prompt has a 5 chars long closing token for llm input and removes it without checking"""
        random.seed(42)
        np.random.seed(42)
        txt = self.generate_prompt(self.text, "", self)[3:-5] # 3 strips the first <s> token as this will be generated again as well. -5 removes enclosing token

        inputs = self.tokenizer(
            txt,
            return_tensors="pt",  # Return PyTorch tensors
        #    padding=True,          # Pad to the longest sequence
        #    truncation=True,       # Truncate to the model's max length
            return_attention_mask=True  # Return the attention mask
        )

        # Access input_ids and attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        do_sample = True if self.temperature is not None else False
        add_dct = {"temperature": self.temperature, "do_sample": do_sample}
        text_tokens = self.model.generate(
            input_ids.to(self.model.device), 
            attention_mask=attention_mask.to(self.model.device),
            min_length=1,  # Set to a positive value to ensure some output
            max_new_tokens=self.seq_len,  # Ensure this is directly passed
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,  # Use eos_token_id to stop generation
            **add_dct
        )

        text = self.tokenizer.decode(text_tokens[0])
        #text = text_tokens[0]
        return text