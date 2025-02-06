import os
from typing import Tuple
from dotenv import load_dotenv

from pipeline.dpo import CustomDPOTrainer
from pipeline.inference import CustomInference

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
from peft import PeftModel
from transformers import TrainingArguments
from .sft import CustomSFTTrainer, DEFAULT_SFT_CONFIG

random.seed(42)
np.random.seed(42)

class MLPipeline:
    """will load HF_TOKEN from .env if present"""
    
    def __init__(self) -> None:

        self.lora_config = LoraConfig(
        r=3, # smaller lora dimension? original 16
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        )

        self.bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
    )
        
    def load_model_from_path(self, path="sft_trained") -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        
        _meval = AutoModelForCausalLM.from_pretrained(path, quantization_config=self.bnb_config)
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token = tokenizer.unk_token
        
        return _meval, tokenizer
    
    def load_model_for_dpo_training(self, model_path):
        # Load the base model again
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
        )

        # Load the adapter twice with different names
        model = PeftModel.from_pretrained(model, model_path, adapter_name="train_adapt")
        model.load_adapter(model_path, adapter_name="reference_adapt")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
        
    def load_dpo_trained_model(self, model_path, adapter="train_adapt"):
        # https://discuss.huggingface.co/t/correct-way-to-save-load-adapters-and-checkpoints-in-peft/77836/8
        # Load the base model again
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.bnb_config,
        )

        # Load the adapter twice with different names
        model = PeftModel.from_pretrained(model, model_path)
        model.load_adapter(model_path+f"/{adapter}", adapter_name="train_adapt")
        model.set_adapter(adapter)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
        
    def train_dpo(self, model,tokenizer, lora_conf, dpo_dataframe, save_path, filter_nan_scores_bt=False):

        # sft on both possible outcomes in dataset for stability.
        if filter_nan_scores_bt:
            # if original columns were nan (no working bt generated) don't sft on it, only use the bt score at dpo rl to tell whats wrong
            df_sft_train = dpo_dataframe.dropna(subset=["scores_bt1", "scores_bt1"]) # dont use
        else:
            df_sft_train = dpo_dataframe
        
        sft_conf = DEFAULT_SFT_CONFIG.copy()
        sft_conf["num_train_epochs"] = 1
        sft_trained_A, _,_ = CustomSFTTrainer(dataset= df_sft_train, model=model, sft_config=sft_conf, tokenizer=tokenizer, test_size=None, llmin_col="llmin", llmout_col="llmout_A").train()
        sft_trained_B, _,_ = CustomSFTTrainer(dataset= df_sft_train, model=sft_trained_A, sft_config=sft_conf, tokenizer=tokenizer, test_size=None, llmin_col="llmin", llmout_col="llmout_B").train()

        trainer = CustomDPOTrainer(dataset = dpo_dataframe, model = sft_trained_B, lora = lora_conf, bnb=self.bnb_config, tokenizer=tokenizer)
        trained_model, hf_trainer, dataset_train = trainer.train()
        
        dpo_trainer = hf_trainer

        if save_path is not None:
            # print(type(trained_model))
            # print(trained_model.peft_config)
            #trained_model.save_pretrained(save_path)

            dpo_trainer.save_model(save_path)
            #dpo_trainer.save_model(output_dir=save_path)

            log_history = dpo_trainer.state.log_history

            # Specify the filename for the pickle file
            filename = pathlib.Path(save_path) / 'loss_history.pkl'

            # Write the dictionary to a pickle file
            with open(filename, 'wb') as file:
                pickle.dump(log_history, file)
        
        return trained_model, hf_trainer, dataset_train        
        
        
    def train_sft(self, dataset_path, generate_prompt, save_path, sft_config_params):
        
        with open(dataset_path,"rb") as file:
            dataset = pickle.load(file)

        dataset["type"] = dataset["parameters"].map(lambda x: type(x.objective_params).__name__)
        dataset["original_index"] = dataset.index
        dataset = dataset.dropna()

        dataset["z-scores"] = dataset["scores"].map(lambda x: (x - np.mean(x)) / np.std(x))
        dataset["coeff_of_var"] = dataset["scores"].map(lambda x: x+abs(np.min(x)) + 1).map(lambda x: np.std(x)/np.mean(x))

        dataset["llm_input"] = dataset["description"]#.map(lambda x: encode_number(x))
        dataset["llm_output"] = dataset["behavior_tree"]#.map(lambda x: encode_number(x))
     
        import shutil
        try:
            shutil.rmtree(save_path)
        except:
            print(f"{save_path} dir not present, will be created")
        print(f"saving to {save_path}")

        trainer = CustomSFTTrainer(dataset=dataset, generate_prompt=generate_prompt, sft_config=sft_config_params, llmin_col="llm_input", llmout_col="llm_output")
        
        _trained_model, hf_trainer, dataset = trainer.train()

        hf_trainer.save_model(output_dir=save_path)
        # Assuming self.sft_trainer.state.log_history is your dictionary
        log_history = hf_trainer.state.log_history

        # Specify the filename for the pickle file
        filename = pathlib.Path(save_path) / 'loss_history.pkl'

        # Write the dictionary to a pickle file
        with open(filename, 'wb') as file:
            pickle.dump(log_history, file)
        
        with open(pathlib.Path(save_path) / 'sft_params.yaml', 'w') as yaml_file:
            yaml.dump(sft_config_params, yaml_file, default_flow_style=False)  # 

        return _trained_model, hf_trainer, dataset

        
    def inference(self, model, tokenizer, text: str, seq_len=2000, temperature=None):
        """ requires prepare_model """
        
        inferencer = CustomInference(text, model, tokenizer, temperature=temperature, seq_len=seq_len)
        res = inferencer.inference()

        return res