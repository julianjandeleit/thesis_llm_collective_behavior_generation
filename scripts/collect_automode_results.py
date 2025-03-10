#%%
import pandas as pd
import numpy as np
import pickle
import random
random.seed(42)
np.random.seed(42)
# Step 1: Read the data from a semicolon-separated file
file_path = '../ressources/outfile_seed20_n600_25-12-14.txt'  # automode bt file path
dataset_path = "../ressources/dataset_seed20_n600_24-12-24.pickle" # sampled dataset with scenarios

# Initialize a list to hold parsed data
parsed_data = []

# Read the file and parse the data
with open(file_path, 'r') as file:
    for line in file:
        if line.strip():  # Check if the line is not empty
            experiment, behavior_tree = line.split(';')
            experiment_number = int(experiment.split('_')[1])  # Extract the number
            parsed_data.append((experiment_number, behavior_tree.strip()))



# Step 2: Create a DataFrame from the parsed data
df_trees = pd.DataFrame(parsed_data, columns=['experiment_number', 'behavior_tree'])
df_trees.set_index('experiment_number', inplace=True)

# Step 3: Read the existing DataFrame from a pickle file
with open(dataset_path, "rb") as file:
#with open("testdata.pkl", "rb") as file:
    df_existing = pickle.load(file)

# Step 4: Merge the DataFrames on the index
merged_df = df_existing.merge(df_trees, left_index=True, right_index=True, how='left')

# Display the merged DataFrame
print(merged_df)
#%% evaluate controller

script_name = "./run_argos_with_vis.sh" # this script is executed to actually run simulaation
import tempfile
import subprocess
import os
import re
from prepare_irace_experiments import append_vis_part
import math
NUM_SCORES_PER_RUN=10
merged_df['scores'] = [None] * len(merged_df) 
# Iterate through the DataFrame
for index, row in merged_df.iterrows():
    print("evaluating " +str(index))
    row = append_vis_part(row)
    behavior_tree = row['behavior_tree']
    mission = row["argos"]
    
    # Check if behavior_tree is not None
    if type(behavior_tree) == float:
        # in this case there was no line present in outfile
        # in case of empty string its present empty string -> output doesnt contain bt
        mt = type(row["parameters"].objective_params).__name__
        print(f"bt is {behavior_tree} (type {mt})")
    if type(behavior_tree) != str or len(behavior_tree) == 0:
        continue
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
    merged_df.at[index,"scores"] = scores
    merged_df.at[index, "avg_score"] = np.mean(scores).item()
    print(merged_df.at[index, "avg_score"])
    os.remove(temp_file_path)
#%%
# write to file
merged_df.to_pickle("../ressources/automode_descriptions_evaluated.pickle")
