#%%
from pipeline.pipeline import MLPipeline
SCRIPT_PATH="./run_argos_with_vis.sh"
MODEL_PATH = "../ressources/dpo_ft_model_24-12-27_seed17"
DF_PATH = "../ressources/llm_evaluated_s17_n600_24-12-20.pickle"
NUM_SCORES_PER_RUN=10

#%% 
import pandas as pd
df = pd.read_pickle(DF_PATH).reset_index()
df['llm_dpo_ft_scores'] = [[] for _ in range(len(df))]
df["llm_dpo_ft_scores"] = df["llm_dpo_ft_scores"].astype(object)
# %%

script_name = ""
import tempfile
import subprocess
import os
import re
import math
import numpy as np
from tqdm.auto import tqdm

def evaluate_configuration(argos,behavior_tree,script_path="./run_argos_with_vis.sh",tmpfile="/tmp/vis.argos"):
    """expects argos to include commented out visualization"""
    
    res = None
        # Create a temporary file for the argos file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.argos') as temp_file:
        # Write the behavior_tree entry to the temporary file
        temp_file.write(argos.encode('utf-8'))
        temp_file_path = temp_file.name  # Get the path of the temporary file
        
    behavior_tree_args = behavior_tree.split()

    # Prepare the command to run, including the temporary file and behavior_tree arguments
    command = [script_path, "--no-vis", temp_file_path, tmpfile] + behavior_tree_args
    try:
        # Run the command and capture the output
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        
        # Print the executed command
        #print(f"Executed: {' '.join(command)}")
        
        # Get the output from the command
        output = result.stdout
        
        # Print the output for debugging
        #print("Command Output:")
        #print(output)
        
        # Extract the number from the line starting with "Score"
        score_line = next((line for line in output.splitlines() if line.startswith("Score")), None)
        #print(f"score line {score_line=}")
        if score_line:
            # Use regex to extract the number from the score line
            score = re.search(r'-?\d+(\.\d+)?', score_line)
            if score:
                res = float(score.group())
            else:
                print("No score number found in the score line.")
        else:
            print("No line starting with 'Score' found in the output.")
            
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
    finally:
        pass#os.remove(temp_file_path)
        
    return res
        
        
mlp = MLPipeline()
mlp.prepare_model() # need both currently
mlp.prepare_model_from_path(path=MODEL_PATH)
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
    
def perform_inference(txt):
    txt = txt_prompt(txt, "", mlp.tokenizer)[:-5]
    out = mlp.inference(txt, seq_len=1000)
    res = None
    try:
        res = out.split("[/INST]")[1]
        res = res.split("</s>")[0]
    except:
        res = None
    #print(res)
    return res
    
    
progress_bar = tqdm(total=len(df))
# Iterate through the DataFrame
for index, row in df.iterrows():
    progress_bar.set_postfix(current_item = index)
    behavior_tree = perform_inference(row["description"])
    mission = row["argos"]
    
    # Check if behavior_tree is not None
    if type(behavior_tree) != str or len(behavior_tree) == 0:
        print("could not evaluate behavior_tree, it is empty")
        progress_bar.update(1)
        continue

    # Execute the command
    scores = []
    for i in range(NUM_SCORES_PER_RUN):
        score = evaluate_configuration(mission, behavior_tree, script_path=SCRIPT_PATH)
        if score is not None:
            scores.append(score)
    
    
    df.at[index,"llm_dpo_ft_behavior_tree"] = behavior_tree
    #print(scores,df.at[index,"llm_scores"])
    df.at[index,"llm_dpo_ft_scores"] = scores
    df.at[index, "llm_dpo_ft_avg_score"] = np.mean(scores).item()  if len(scores) > 0 else None
    progress_bar.write(f"evaluated {index}, score: "+str(df.at[index, "llm_dpo_ft_avg_score"]))
    progress_bar.update(1)
# %%

df.to_pickle(f"llm_dpo_ft_evaluated.pickle")

# %%
