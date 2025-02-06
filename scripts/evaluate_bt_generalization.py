#%%
import pandas as pd
import numpy as np
import pickle
import random
random.seed(42)
np.random.seed(42)
# Step 1: Read the data from a semicolon-separated file
behavior_tree = '--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.2877 --n01 5 --a01 4 --att01 3.565 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.3689 --n11 5 --a11 2 --p11 0'
dataset_path = "../ressources/final_experiments/result_aggregation_exp/llm_evaluated_only_black.pickle"

# Step 3: Read the existing DataFrame from a pickle file
with open(dataset_path, "rb") as file:
#with open("testdata.pkl", "rb") as file:
    df_existing = pickle.load(file)

# Step 4: Merge the DataFrames on the index
merged_df = df_existing

# Display the merged DataFrame
print(merged_df)
#%% evaluate controller

script_name = "./run_argos_with_vis.sh"
import tempfile
import subprocess
import os
import re
from prepare_irace_experiments import append_vis_part
import math
NUM_SCORES_PER_RUN=10
merged_df['btscores'] = [None] * len(merged_df) 
# Iterate through the DataFrame
for index, row in merged_df.iterrows():
    print("evaluating " +str(index))
    row = append_vis_part(row)
    mission = row["argos"]

    # Create a temporary file for the argos file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.argos') as temp_file:
        # Write the behavior_tree entry to the temporary file
        temp_file.write(mission.encode('utf-8'))
        temp_file_path = temp_file.name  # Get the path of the temporary file

    # Prepare the command to run
    behavior_tree_args = behavior_tree.split()

    # Prepare the command to run, including the temporary file and behavior_tree arguments
    command = [script_name, "--no-vis", temp_file_path, "/tmp/vis.argos"] + behavior_tree_args
    #print(" ".join(command))

    # Execute the command
    scores = []
    for i in range(NUM_SCORES_PER_RUN):
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
                    #print(f"Extracted Score: {score.group()}")
                    scores.append(float(score.group()))
                else:
                    print("No score number found in the score line.")
            else:
                print("No line starting with 'Score' found in the output.")
                
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command: {e}")

    # Optionally, remove the temporary file after execution
    merged_df.at[index,"argos"] = mission
    merged_df.at[index,"btscores"] = scores
    merged_df.at[index, "bt_avg_score"] = np.mean(scores).item()
    print(merged_df.at[index, "bt_avg_score"])
    os.remove(temp_file_path)
#%%
# write to file
merged_df.to_pickle("../ressources/automode_bt_evaluated.pickle")
