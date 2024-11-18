#%%


import pickle
import pandas as pd

with open("../ressources/dataset_seed42_n400.pickle", "rb") as file:
    df = pickle.load(file)
# %%

def append_vis_part(row):
    vispart="""<!--<qt-opengl>
            <camera>
                <placement idx="0"
                           position="0,0,4.5"
                           look_at="0,0,0"
                           lens_focal_length="30" />
            </camera>
        </qt-opengl>-->
        """

    original = row["argos"]
    def insert_between_strings(text, first_string, second_string, new_string):
        # Find the start and end positions of the first and second strings
        start_index = text.find(first_string)
        end_index = text.find(second_string, start_index)

        # Check if both strings are found
        if start_index != -1 and end_index != -1:
            # Calculate the position to insert the new string
            insert_position = end_index  # Insert after the first occurrence of the second string

            # Create the new string with the inserted text
            new_text = text[:insert_position] + new_string + text[insert_position:]
            return new_text
        else:
            return text  # Return the original text if either string is not found
    
    with_vis = insert_between_strings(original, "<visualization>", "</visualization>", vispart)
    row["argos"] = with_vis
    return row

outdir= "generated_irace_datasets"
template = "../ressources/irace_template"

import os
import shutil
import pathlib
os.makedirs(outdir, exist_ok=True)
#%%

scriptpath = pathlib.Path(outdir) / 'slurmscript.sh'
with open(scriptpath, 'w') as file:  # 'w' mode to c
 pass 
# write experiments
for index, row in df.iterrows():
    # experiment
    row = append_vis_part(row)
    dirname = f"experiment_{index}"
    shutil.copytree(template, pathlib.Path(outdir) / dirname)
    argospath = pathlib.Path(outdir) / dirname / "experiments-folder" / "mission.argos"
    with open(argospath, 'w') as file:  # 'w' mode to overwrite the file
        file.write(row["argos"])

    # experiments script
    # Define the file path


# Open the file in append mode
    with open(scriptpath, 'a') as file:  # 'a' mode to append to the file
        line_to_append = f"sbatch --partition single task_irace.sh {dirname} outfile.txt\n"  # Create a line to append
        file.write(line_to_append)  # Write 


shutil.copy("slurm/task_irace.sh",outdir)

    #print(f"Index: {index}, A: {row['argos']}")
    #brea
# %%
