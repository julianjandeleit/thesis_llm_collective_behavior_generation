#%%


import pickle
import pandas as pd


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

#%%
if __name__ == "__main__":
    with open("../ressources/dataset_seed17_n600_24-12-18_target_lights_big.pickle", "rb") as file:
        df = pickle.load(file)
        
    outdir= "generated_irace_datasets"
    template = "../ressources/irace_template"
    #template = "../ressources/irace_template_dev"

    import os
    import shutil
    import pathlib
    if pathlib.Path(outdir).exists():
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)


    # scriptpath = pathlib.Path(outdir) / 'slurmscript.sh'
    # with open(scriptpath, 'w') as file:  # 'w' mode to c
    #  pass 
    # write experiments
    for index, row in df.iterrows():
        # experiment
        row = append_vis_part(row)
        dirname = f"experiment_{index}"
        df.at[index,"dirname"] = dirname
        shutil.copytree(template, pathlib.Path(outdir) / dirname)
        argospath = pathlib.Path(outdir) / dirname / "experiments-folder" / "mission.argos"
        with open(argospath, 'w') as file:  # 'w' mode to overwrite the file
            file.write(row["argos"])

        with open(pathlib.Path(outdir) / dirname / "mission_type.txt", "w") as file:
            file.write(type(row["parameters"].objective_params).__name__)

        # experiments script
        # Define the file path


    # Open the file in append mode
        # with open(scriptpath, 'a') as file:  # 'a' mode to append to the file
        #     line_to_append = f"sbatch --partition single task_irace.sh {dirname} outfile.txt\n"  # Create a line to append
        #     file.write(line_to_append)  # Write 

    # %%
    from datetime import timedelta
    num_experiments = df.shape[0]
    NUM_SLURMTASKS = 50
    TIME_PER_EXPERIMENT = timedelta(minutes=150)
    import math
    experiments_per_task = math.ceil(num_experiments / float(NUM_SLURMTASKS))

    total_seconds = int(TIME_PER_EXPERIMENT.total_seconds()*experiments_per_task)+5*60 # 5 minutes for enroot install 
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Print in the desired format
    timestr = f"{hours}:{minutes:02}:{seconds:02}"

    jobs = []
    job = ""
    slurms = ""
    for index, row in df.iterrows():
        dirname = row["dirname"]
        if index % experiments_per_task == 0:
                if job != "":        
                    jobs.append(job)
                slurms += f"sbatch --partition single slurmjob_{len(jobs)}.sh\n"
                job = """
odir=$(pwd)
export ENROOT_DATA_PATH=$TMPDIR/enrootdir
export ENROOT_CACHE_PATH=$TMPDIR/enrootcache
export ENROOT_RUNTIME_PATH=$TMPDIR/enrootruntime
mkdir -p $ENROOT_DATA_PATH
mkdir -p $ENROOT_CACHE_PATH
mkdir -p $ENROOT_RUNTIME_PATH
cd ~
zstd -d automode.zstd -o $TMPDIR/automode.sqsh
cd $TMPDIR
enroot create -n automode automode.sqsh
ls $ENROOT_DATA_PATH
ls $ENROOT_CACHE_PATH
ls $ENROOT_RUNTIME_PATH
cd $odir                
"""

        job += f"odir=$(pwd)\nmv {dirname} $TMPDIR\nbash $odir/task_irace.sh $TMPDIR/{dirname} outfile.txt\nmv $TMPDIR/{dirname} $odir/{dirname}\n" 



    if job != "":
        jobs.append(job)  # Append the last job if it exists

    # slurm parameters

    slurmparameters = f"""#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time={timestr}
#SBATCH --mem=2500
#SBATCH --job-name=irace_job
#SBATCH --cpus-per-task=8
"""

    # Write each job to a separate file
    for i, job in enumerate(jobs):
        job_filename = f"slurmjob_{i}.sh"
        with open(pathlib.Path(outdir) / job_filename, 'w') as job_file:
            job_file.write(slurmparameters+job)


    # Write all SLURM commands to a separate file
    with open(pathlib.Path(outdir) / "slurm_commands.sh", 'w') as slurm_file:
        slurm_file.write(slurms)


    source_file_path = 'slurm/task_irace_template.sh'  # Replace with your source file path
    destination_file_path =  pathlib.Path(outdir) / 'task_irace.sh'  # Replace with your destination file path
    replacement_string = timestr  # Replace with the string you want to use

    # Load the file, replace [RUNTIME], and write to another file
    with open(source_file_path, 'r') as source_file:
        content = source_file.read()  # Read the entire content of the source file

    # Replace [RUNTIME] with the desired string
    modified_content = content.replace('[RUNTIME]', replacement_string)

    # Write the modified content to the destination file
    with open(destination_file_path, 'w') as destination_file:
        destination_file.write(modified_content)

    #shutil.copy("slurm/task_irace.sh",outdir)

        #print(f"Index: {index}, A: {row['argos']}")
        #brea
    # %%
    print("execute the following to upload to hpc:\nzip -r generated_irace_datasets.zip generated_irace_datasets && scp -r generated_irace_datasets.zip kn_pop515691@bwunicluster.scc.kit.edu:irace_experiments_slurm.zip && rm -r generated_irace_datasets")
    # %%
