#%%
from pipeline.pipeline import MLPipeline
MODEL_PATH = "../llm_training/demo_train_2024-12-12_16_automode_evaluated_concat_s14n600_s15n600_notbtstartend"
        
def txt_prompt(llmin, llmout, tokenizer):
        #f"\nNUMNODES={int(len(llmout.split(' '))/2.0)}\n"+
        # f"\nsyntax example: {stx}\n"
        # Specify the tree inside |BTSTART|<TREE>|BTEND| by starting the tree with --nroot.
        messages = [
            {"role": "user", "content": llmin+"\nGenerate the behavior tree that achieves the objective of this mission."},
            {"role": "assistant", "content": llmout},
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, truncation=True, return_dict=False) # wraps text with special tokens depending on role (assitant or user)
        return text

    

import argparse
import os

def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some text and a model directory.')

    # Add arguments
    parser.add_argument('--text', type=str, required=True, help='The text to process.')
    parser.add_argument('--model', type=str, required=True, help='Path to the model directory.')
    parser.add_argument('--wrap', action='store_true', help='Specify if the text should be wrapped (default: True).')

    # Parse the arguments
    args = parser.parse_args()

    # Check if the model directory exists
    if not os.path.isdir(args.model):
        print(f"Error: The specified model directory '{args.model}' does not exist.")
        return

    mlp = MLPipeline()
    mlp.prepare_model()

    def perform_inference(txt, wrap=True):
        if wrap:
            txt = txt_prompt(txt, "", mlp.tokenizer)[:-5]
        out = mlp.inference(txt, args.model, seq_len=1000)
        res = None
        try:
            res = out.split("[/INST]")[1]
            res = res.split("</s>")[0]
        except IndexError as e:
            res = out
        return res, txt

    res, text = perform_inference(args.text, args.wrap)
    print(text)
    print(res)

if __name__ == '__main__':
    main()

  