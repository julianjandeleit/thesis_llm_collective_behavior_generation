import torch
import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

def load_default_model():
    """ loads Mathstral-7B-v0.1"""
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
    tokenizer.padding_side = 'right'

    model = get_peft_model(model, lora_config)

    return model, tokenizer, lora_config, bnb_config

def DEFAULT_GENERATE_PROMPT(llmin, llmout, baseclass):
        messages = [
            {"role": "user", "content": llmin+"\nGenerate the behavior tree that achieves the objective of this mission."},
            {"role": "assistant", "content": llmout},
        ]

        text = baseclass.tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False,add_special_tokens=False) # wraps text with special tokens depending on role (assitant or user)
        return text


def DESCRIPTIVE_GENERATE_PROMPT(llmin, llmout, baseclass):
        messages = [
            {"role": "user", "content": llmin+"\n\nGenerate the behavior tree that achieves the objective of this mission.\nThis is how the desired behavior tree syntax is structured:The first element is always the node identifier (like --nroot or --n0), followed by the type of node. 1 and 1 define selector nodes, 2 and 3 sequence nodes, 4 is a decorator node, 5 an action and 6 a condition. Different versions of the same node type are specified by --c[nodeID] parameter for conditions and --a[nodeID] parameter for actions. Conditions range from 0-6 with the order black floor, gray floor, white floor, neighbors count, inverted neighbors count, fixed probability and light. Depending on the condition, they have additional attributes like probability (p) or neighbors and epsilon (p, w). Actions range from 0-5 in the order exploration, stop, phototaxis, anti-phototaxis, attraction and repulsion. They all have a success probability parameter p. Additional parameters for some actions are steps range (rwm), attraction (att) and repulsion rep.\nNow generate the robot controller."},
            {"role": "assistant", "content": llmout},
        ]

        text = baseclass.tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False,add_special_tokens=False) # wraps text with special tokens depending on role (assitant or user)
        return text