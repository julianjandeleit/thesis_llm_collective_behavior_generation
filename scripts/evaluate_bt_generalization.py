import pandas as pd
import numpy as np
import pickle
import subprocess
import tempfile
import os
import re

# Initialize random seed for reproducibility
np.random.seed(42)

# Step 1: Read the data from a pickle file
dataset_path = "../ressources/final_experiments/result_aggregation_exp/llm_evaluated_only_black.pickle"
with open(dataset_path, "rb") as file:
    df_existing = pickle.load(file)

# Define behavior trees based on "type"
behavior_trees = {
    "Aggregation": "--nroot 3 --nchildroot 4 --n0 0 --nchild0 2 --n00 6 --c00 5 --p00 0.6627 --n01 5 --a01 3 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 0 --p10 0.9127 --n11 5 --a11 4 --att11 4.3675 --p11 0 --n2 0 --nchild2 2 --n20 6 --c20 5 --p20 0.6156 --n21 5 --a21 2 --p21 0 --n3 0 --nchild3 2 --n30 6 --c30 5 --p30 0.0786 --n31 5 --a31 1 --p31 0",
    "Foraging": "--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 0 --p00 0.9995 --n01 5 --a01 0 --rwm01 1 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 2 --p10 0.9999 --n11 5 --a11 2 --p11 0",
    "Distribution": "--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 1 --p00 0.9995 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.9999 --n11 5 --a11 5 --rep11 4.9999 --p11 0",
    "Connection": "--nroot 3 --nchildroot 2 --n0 0 --nchild0 2 --n00 6 --c00 2 --p00 0.9995 --n01 5 --a01 2 --p01 0 --n1 0 --nchild1 2 --n10 6 --c10 5 --p10 0.9999 --n11 5 --a11 4 --att11 4.6862 --p11 0",
    # Add more types as necessary
}

behavior_trees["Aggregation"] = behavior_trees["Foraging"]

# Prepare to store scores in the DataFrame
NUM_SCORES_PER_RUN = 10
df_existing['btscores'] = [None] * len(df_existing)

# Evaluate each row
for index, row in df_existing.iterrows():
    print(f"Evaluating {index} with type {row['type']}")
    
    # Select the behavior tree based on the 'type' column
    behavior_tree = behavior_trees.get(row["type"], None)  # Default to type1 if not found
    print(behavior_tree, row["type"])
    
    mission = row["argos"]

    # Create a temporary file for the argos file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.argos') as temp_file:
        temp_file.write(mission.encode('utf-8'))
        temp_file_path = temp_file.name  # Get the path of the temporary file

    # Prepare the command to run
    behavior_tree_args = behavior_tree.split()
    command = ["./run_argos_with_vis.sh", "--no-vis", temp_file_path, "/tmp/vis.argos"] + behavior_tree_args

    # Execute the command and capture scores
    scores = []
    for _ in range(NUM_SCORES_PER_RUN):
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            output = result.stdout

            # Extract the score from the output using regex
            score_line = next((line for line in output.splitlines() if line.startswith("Score")), None)
            if score_line:
                score = re.search(r'-?\d+(\.\d+)?', score_line)
                if score:
                    scores.append(float(score.group()))
                else:
                    print("No score number found in the score line.")
            else:
                print("No line starting with 'Score' found in the output.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while executing the command: {e}")

    # Store the results back in the DataFrame
    df_existing.at[index, "btscores"] = scores
    df_existing.at[index, "bt_avg_score"] = np.mean(scores)
    print(f"Avg BT score for index {index}: {df_existing.at[index, 'bt_avg_score']}")

    # Clean up the temporary file
    os.remove(temp_file_path)

# Save the updated DataFrame
df_existing.to_pickle("../ressources/automode_bt_evaluated.pickle")
