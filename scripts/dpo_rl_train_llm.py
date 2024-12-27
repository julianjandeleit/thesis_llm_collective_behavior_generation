#%%
from pipeline.pipeline import MLPipeline
SCRIPT_PATH="./run_argos_with_vis.sh"
MODEL_PATH = "../llm_training/demo_train_2024-12-23_12_automode_evaluated_concat_s14-s18_24-12-23_wtargetlights"
OUTPUT_PATH="dpo_rl_model"
NUM_SCORES_PER_RUN=10
NUM_ROWS_PER_EPOCH=250
NUM_EPOCHS=25
SKELETON_TEMPLATE="../ressources/skeleton.argos"
#%% 
import random
import numpy as np
random.seed(42)
np.random.seed(42)
import pandas as pd
from swarm_descriptions.mission_elements import get_generators, MissionParams
from swarm_descriptions.configfiles import config_to_string
from swarm_descriptions.utils import truncate_floats
from swarm_descriptions.configfiles import ET, Configurator

def sample_dataset(n_rows = 10000, generators = get_generators()) -> pd.DataFrame:
    rows = []
    for _n in tqdm(range(n_rows)):
        mission = MissionParams.sample(*generators)
        conf = config_to_string(mission.configure())
        conf = truncate_floats(conf)
        desc = random.sample(mission.describe(),1)[0]  
        desc = truncate_floats(desc)
        rows.append({"description": desc, "configuration": conf, "parameters": mission})
    dataset = pd.DataFrame(rows)
    
    return dataset
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
#mlp.prepare_model_from_path(path=MODEL_PATH)
mlp.prepare_dpo_model(model_path=MODEL_PATH)
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
    out = mlp.inference(txt, seq_len=1000, temperature=0.41)
    res = None
    try:
        res = out.split("[/INST]")[1]
        res = res.split("</s>")[0]
    except:
        res = None
    #print(res)
    return res

def evaluate_tree(behavior_tree, mission):
    # Check if behavior_tree is not None
    if type(behavior_tree) != str or len(behavior_tree) == 0:
        print("could not evaluate behavior_tree, it is empty")
        return None

    # Execute the command
    scores = []
    for i in range(NUM_SCORES_PER_RUN):
        score = evaluate_configuration(mission, behavior_tree, script_path=SCRIPT_PATH)
        if score is not None:
            scores.append(score)
            
    return np.mean(scores).item()  if len(scores) > 0 else None

def add_config_to_dataset(df: pd.DataFrame, skeleton: ET.ElementTree):
    result = []
    for config_params in df["configuration"]:
        argos_config = config_params_to_argos_config(config_params, skeleton)        
        result.append(argos_config)
    df["argos"] = result
    return df  

def config_params_to_argos_config(params: str, skeleton: ET.ElementTree):
    xml = ET.fromstring(params)
    config_tree = Configurator().convert_config_params(params=xml, skeleton_root=skeleton)
    config = config_tree.getroot()
    config = config_to_string(config)
    xml = ET.fromstring(params)
    config_tree = Configurator().convert_config_params(params=xml, skeleton_root=skeleton)
    config = config_tree.getroot()
    config = config_to_string(config)
    return config

def rescale_score(score, df, category):
    df_cat = df[df.type == category]
    min_score = min(df_cat.scores_bt1.min(), df_cat.scores_bt2.min())
    max_score = max(df_cat.scores_bt1.max(), df_cat.scores_bt2.max())
    #print(score,min_score, max_score,float(score-min_score),float(max_score - min_score))
    if score is None or min_score is None or max_score is None or min_score == max_score:
        return None
    score_scaled = float(score-min_score)/float(max_score - min_score)
    #print(score_scaled)
    
    return score_scaled
dataframes = []
for epoch in range(NUM_EPOCHS):    
    print(f"{epoch}/{NUM_EPOCHS}")
    df = sample_dataset(NUM_ROWS_PER_EPOCH)
    skeleton = ET.parse(SKELETON_TEMPLATE)
    df = add_config_to_dataset(df, skeleton)
    df["type"] = df["parameters"].map(lambda x: type(x.objective_params).__name__)
    df['scores_bt1'] = [None] * len(df)
    df['scores_bt2'] = [None] * len(df) 
    df['bt1'] = [None] * len(df)
    df['bt2'] = [None] * len(df) 
    df['scores_bt1_scaled'] = [None] * len(df)
    df['scores_bt2_scaled'] = [None] * len(df) 

    progress_bar = tqdm(total=len(df))
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        progress_bar.set_postfix(current_item = index)
        behavior_tree = perform_inference(row["description"])
        behavior_tree2 = perform_inference(row["description"])
        # print(behavior_tree)
        # print(behavior_tree2)
        mission = row["argos"]
        category = row["type"]
        
        score1 = evaluate_tree(behavior_tree, mission)
        score2 = evaluate_tree(behavior_tree2, mission)
        df.at[index, "scores_bt1"] = score1
        df.at[index, "scores_bt2"] = score2
        
        
        #exit(1)
        
        df.at[index,"bt1"] = behavior_tree
        df.at[index,"bt2"] = behavior_tree2

        #print(scores,df.at[index,"llm_scores"])
        # df.at[index,"llm_scores"] = scores
        # df.at[index, "llm_avg_score"] = 
        progress_bar.write(f"generated {index}")
        progress_bar.update(1)
    # %%
    #score1_scaled = rescale_score(score1,df,category)
    #score2_scaled = rescale_score(score2,df,category)
    #print(category, score1, score2, score1_scaled, score2_scaled)
    #df.to_pickle("debug.pickle")
    df["scores_bt1_scaled"] = df.apply(lambda row: rescale_score(row["scores_bt1"], df, row["type"]), axis=1)
    df["scores_bt2_scaled"] = df.apply(lambda row: rescale_score(row["scores_bt2"], df, row["type"]), axis=1)
    df = df.dropna(subset=["scores_bt1_scaled", "scores_bt2_scaled"])
    print(f"scores computed and rescaled")
    def choose_and_reject(row):
        if row['scores_bt1_scaled'] > row['scores_bt2_scaled']:
            return pd.Series({
                'chosen': txt_prompt(row["description"], row['bt1'], mlp.tokenizer),
                'rejected': txt_prompt(row["description"], row['bt2'], mlp.tokenizer),
                'score_chosen': row['scores_bt1_scaled'],
                'score_rejected': row['scores_bt2_scaled']
            })
        else:
            return pd.Series({
                'chosen':txt_prompt(row["description"], row['bt2'], mlp.tokenizer),
                'rejected': txt_prompt(row["description"], row['bt1'], mlp.tokenizer),
                'score_chosen': row['scores_bt2_scaled'],
                'score_rejected': row['scores_bt1_scaled']
            })

    result = df.apply(choose_and_reject, axis=1)
    df = pd.concat([df, result], axis=1)

    print(df[["chosen", "rejected", "score_chosen","score_rejected"]].head())

    #%%
     # as this is done everytime the final version should be the one in the directory after exececution, I assume that training the same model twice works 
    mlp.train_dpo(df, save_path=OUTPUT_PATH)

    dataframes.append(df)


# %%
import pickle
for i, df in enumerate(dataframes):
    df['dataset_position'] = i  # Add a new column with the position
    
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df.to_pickle(OUTPUT_PATH+"/dataset.pickle")