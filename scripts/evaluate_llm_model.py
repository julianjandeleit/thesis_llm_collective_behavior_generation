#%%
from pipeline.pipeline import MLPipeline
from pipeline.utils import DEFAULT_GENERATE_PROMPT, DESCRIPTIVE_GENERATE_PROMPT
PROMPT = DESCRIPTIVE_GENERATE_PROMPT
SCRIPT_PATH="./run_argos_with_vis.sh" # used for evaluation in simulation
MODEL_PATH = "../llm_training/trained_sft" # where fine-tuned model is saved
DF_PATH = "../ressources/final_experiments/automode_datasets/df_increasing_size_validate.pickle" # scenarios with automode baseline to validate on (same format as for training)
NUM_SCORES_PER_RUN=10

#%% 
import pandas as pd
df = pd.read_pickle(DF_PATH).reset_index(drop=True)
df['llm_scores'] = [[] for _ in range(len(df))]
df["llm_scores"] = df["llm_scores"].astype(object)
#df = df.head(10)
# %%

script_name = ""
import tempfile
import subprocess
import os
import re
import math
import numpy as np
from tqdm.auto import tqdm

def evaluate_configuration(argos, behavior_tree, script_path="./run_argos_with_vis.sh", tmpfile="/tmp/vis.argos"):
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
model, tokenizer = mlp.load_model_from_path(path=MODEL_PATH)
    
def perform_inference(txt):
    out = mlp.inference(model,tokenizer, txt, seq_len=1000, generate_prompt=PROMPT)
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
    
    
    df.at[index,"llm_behavior_tree"] = behavior_tree
    #print(scores,df.at[index,"llm_scores"])
    df.at[index,"llm_scores"] = scores
    df.at[index, "llm_avg_score"] = np.mean(scores).item()  if len(scores) > 0 else None
    progress_bar.write(f"evaluated {index}, score: "+str(df.at[index, "llm_avg_score"]))
    progress_bar.update(1)
# %%

df.to_pickle(f"llm_evaluated.pickle")

# %%
