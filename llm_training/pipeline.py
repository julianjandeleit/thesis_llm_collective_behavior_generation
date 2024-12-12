import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

secret_value_0 = os.getenv('HF_TOKEN')

from huggingface_hub import login
login(token = secret_value_0)

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig, get_peft_model
import pandas as pd
from swarm_descriptions.mission_elements import get_generators, MissionParams
from swarm_descriptions.configfiles import config_to_string
from swarm_descriptions.utils import truncate_floats
import random
import pyarrow as pa
import pyarrow.dataset as ds
import pickle
import numpy as np
import random
import re
import pathlib
import yaml
from sklearn.model_selection import train_test_split
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments

random.seed(42)
np.random.seed(42)


def load_foundation_model_and_tokenizer():
    lora_config = LoraConfig(
        r=3, # smaller lora dimension? original 16
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_name = "mistralai/Mathstral-7B-v0.1"

    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


    model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                            quantization_config=bnb_config,)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token = tokenizer.unk_token

    model = get_peft_model(model, lora_config)
    
    return model, tokenizer, lora_config, bnb_config

def load_dataset(path_automode_descriptions_evaluated="../ressources/automode_descriptions_evaluated.pickle"):
    with open(path_automode_descriptions_evaluated,"rb") as file:
        dataset = pickle.load(file)

    dataset["type"] = dataset["parameters"].map(lambda x: type(x.objective_params).__name__)
    dataset["original_index"] = dataset.index
    dataset = dataset.dropna()

    dataset["z-scores"] = dataset["scores"].map(lambda x: (x - np.mean(x)) / np.std(x))
    dataset["coeff_of_var"] = dataset["scores"].map(lambda x: x+abs(np.min(x)) + 1).map(lambda x: np.std(x)/np.mean(x))



    dataset.head()
    dataset["llm_input"] = dataset["description"]#.map(lambda x: encode_number(x))
    dataset["llm_output"] = dataset["behavior_tree"]#.map(lambda x: encode_number(x))
    return dataset

def encode_number(text):
    def f(match):
        num = match.group(0)  # The entire matched number
        i = match.group(1)    # The integer part of the number
        li = len(i)           # Length of the integer part
        d = match.group(3)    # The decimal part of the number (if any)
        ld = len(d) if d else 0  # Length of the decimal part, default to 0 if None
        
        if d:
            prefix = f'<sn>{li}.{ld}<mn>'
        else:
            prefix = f'<sn>{li}<mn>'
        
        return prefix + num + '<en>'
    
    pattern = r'(\d+)(\.(\d+))?'  # Regular expression pattern to match numbers
    return re.sub(pattern, f, text)

def decode_number(text):
    pattern = r'<sn>[\d\.]+<mn>'  # Pattern to match the processed number format
    text = re.sub(pattern, '', text)  # Remove the <sn> and <mn> tags
    text = re.sub(r'<en>', '', text)   # Remove the <en> tag
    return text

def prepare_dataset_for_training(dataset, tokenizer, generate_prompt):

    #dataset["tokens"] = dataset.apply(lambda x: generate_prompt(x, tokenizer),axis=1)
    #dataset["text"] = dataset.apply(lambda x: tokenizer.decode(x.tokens), axis=1)
    dataset["text"] = dataset.apply(lambda x: generate_prompt(x, tokenizer), axis=1)
    
    dataset = dataset.filter(["text","original_index"])
    dataset.head()

    generated_train_dataset, generated_val_dataset = train_test_split(dataset, test_size=0.2)

    def to_dataset(df):
        dataset = ds.dataset(pa.Table.from_pandas(df).to_batches())

        hg_dataset = Dataset(pa.Table.from_pandas(df))
        return hg_dataset

    generated_train_dataset = to_dataset(generated_train_dataset.head(2500))
    generated_val_dataset = to_dataset(generated_val_dataset.head(500))
    return generated_train_dataset, generated_val_dataset
    
def inference_model(bnb_config, lora_config, text, tokenizer, model_path="sft_config", seq_len=2000):
    print(f"loading model from {model_path}")
    _meval = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config)
   # _meval = AutoModelForCausalLM.from_pretrained(model_path, load_in_4bit=True, device_map="auto")
    #_meval = get_peft_model(_meval, lora_config)

    # %%

    #inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
    #inputs = tokenizer.encode(generated_val_dataset[0]["text"], return_tensors="pt", padding=True)

    #attention_mask = inputs["attention_mask"]
    inputs = tokenizer(
        text,
        return_tensors="pt",  # Return PyTorch tensors
    #    padding=True,          # Pad to the longest sequence
    #    truncation=True,       # Truncate to the model's max length
        return_attention_mask=True  # Return the attention mask
    )

    # Access input_ids and attention_mask
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    text_tokens = _meval.generate(
        input_ids.to(_meval.device), 
        attention_mask=attention_mask.to(_meval.device),
        min_length=1,  # Set to a positive value to ensure some output
        max_new_tokens=seq_len,  # Ensure this is directly passed
        do_sample=False,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id  # Use eos_token_id to stop generation
    )

    text = tokenizer.decode(text_tokens[0])
    #text = text_tokens[0]
    return text

def generate_prompt(sample, tokenizer):
        messages = [
            {"role": "user", "content": sample["llm_input"]+"\nGenerate the behavior tree that achieves the objective of this mission."},
            {"role": "assistant", "content": str(sample["llm_output"])},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False) # wraps text with special tokens depending on role (assitant or user)
        return encode_number(text) #text["input_ids"][1:]


class MLPipeline:
    
    def __init__(self) -> None:
        # these do not produce anything useful and are mainly for testing
        self.sft_default_config = {
    "max_seq_length": 2000,
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,  # 4
    "optim": "paged_adamw_32bit",
    "save_steps": 0,
    "logging_steps": 25,
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "bf16": False,
    "max_grad_norm": 0.3,
    "max_steps": -1,
    "warmup_ratio": 0.03,
    "group_by_length": True,
    "lr_scheduler_type": "cosine",
    "report_to": "none",
    "eval_strategy": "epoch"
}
        
    def prepare_model(self):
        model, tokenizer, lora_config, bnb_config = load_foundation_model_and_tokenizer()
        self.model = model
        self.tokenizer = tokenizer
        self.lora_config = lora_config
        self.bnb_config = bnb_config
        
    def prepare_dataset(self, dataset_path="../ressources/automode_descriptions_evaluated.pickle", generate_prompt=None):
        df = load_dataset(path_automode_descriptions_evaluated=dataset_path)
        generated_train_dataset, generated_val_dataset = prepare_dataset_for_training(df, self.tokenizer, generate_prompt)
        self.train_dataset = generated_train_dataset
        self.val_dataset = generated_val_dataset
        
    def train_model(self, sft_config_params, save_path="sft_trained"):
        """ requires all prepare steps """
            # Set up the training arguments
        training_args_dict = {
    "output_dir": save_path,
    "evaluation_strategy": "epoch",
    "eval_steps": 1,
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
    "dataset_text_field": "text",
    "output_dir": "logs"
}
        sft_arguments = SFTConfig(**training_args_dict,
                                **sft_config_params)

        print(f"example sample {self.train_dataset[0]['text']}")

        # Initialize the SFTTrainer with the SFTConfig
        sft_trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            args=sft_arguments,
            packing=False,
        )

        # Note: If you need to set padding_side, you can do it in the config or directly in the tokenizer.
        # For example:
        self.tokenizer.padding_side = 'right'  # Ensure padding is set to 'right' if needed
        self.sft_trainer = sft_trainer
        self.sft_trainer.train()
        
        
        inputs = self.tokenizer(
        self.train_dataset[0]['text'].split("[/INST]")[0]+"[/INST]",
        return_tensors="pt",  # Return PyTorch tensors
    #    padding=True,          # Pad to the longest sequence
    #    truncation=True,       # Truncate to the model's max length
        return_attention_mask=True  # Return the attention mask
        )

        # Access input_ids and attention_mask
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        text_tokens = self.model.generate(
            inputs['input_ids'].to(self.model.device), 
            attention_mask=attention_mask.to(self.model.device),
            min_length=1,  # Set to a positive value to ensure some output
            max_new_tokens=1500,  # Ensure this is directly passed
            do_sample=False,
            top_p=1.0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id  # Use eos_token_id to stop generation
        )

        text = self.tokenizer.decode(text_tokens[0])
        print(f"last generated output is {text}")

        import shutil
        try:
            shutil.rmtree(save_path)
        except:
            print(f"{save_path} dir not present, will be created")
        print(f"saving to {save_path}")
        self.sft_trainer.save_model(output_dir=save_path)

        # Assuming self.sft_trainer.state.log_history is your dictionary
        log_history = self.sft_trainer.state.log_history

        # Specify the filename for the pickle file
        filename = pathlib.Path(save_path) / 'loss_history.pkl'

        # Write the dictionary to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(log_history, file)
        #self.model.save_pretrained(save_path, from_pt=True)
        with open(pathlib.Path(save_path) / 'sft_params.yaml', 'w') as yaml_file:
            yaml.dump(sft_config_params, yaml_file, default_flow_style=False)  # 
        
        
    def train_pipeline(self, dataset_path, generate_prompt, save_path, sft_config_params):
        self.prepare_model()
        self.prepare_dataset(dataset_path, generate_prompt)
        self.train_model(sft_config_params, save_path)
        
    def inference(self, text: str, model_path="sft_trained", seq_len=2000):
        """ requires prepare_model """
        res = inference_model(self.bnb_config, self.lora_config, text, self.tokenizer, model_path, seq_len)
        return res