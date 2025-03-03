#%%
from pipeline.pipeline import MLPipeline
from pipeline.utils import DEFAULT_GENERATE_PROMPT

import argparse
import os

def nowrap_prompt(llmin, llmout, baseclass):
        messages = [
            {"role": "user", "content": llmin},
            {"role": "assistant", "content": llmout},
        ]

        text = baseclass.tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False,add_special_tokens=False) # wraps text with special tokens depending on role (assitant or user)
        return text

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some text and a model directory.')

    # Add arguments
    parser.add_argument('--text', type=str, required=True, help='The text to process.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--nowrap', action='store_true', help='Specify if the text should be wrapped with prompt template.')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the model directory exists
    if not os.path.isdir(args.model):
        print(f"Error: The specified model directory '{args.model}' does not exist.")
        return

    mlp = MLPipeline()
    #model, tokenizer = mlp.load_dpo_trained_model(args.model)
    model, tokenizer = mlp.load_model_from_path(args.model)
    #print(f"loaded model with adapters: {model.active_adapters}")

    def perform_inference(txt, nowrap=True):
        if nowrap:
             gp = nowrap_prompt
        else:
             gp = None
        #    txt = txt_prompt(txt, "", mlp.tokenizer)[:-5]
        out = mlp.inference(model, tokenizer, txt, seq_len=1000, generate_prompt=gp)
        res = None
        try:
            res = out.split("[/INST]")[1]
            res = res.split("</s>")[0]
        except IndexError as e:
            res = out
        return res, txt

    res, text = perform_inference(args.text, args.nowrap)
    print("Input:\n"+text)
    print("Output:\n"+res)

if __name__ == '__main__':
    main()

  